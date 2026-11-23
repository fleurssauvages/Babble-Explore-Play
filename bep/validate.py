"""Validation: the robot (Franka arm + Spike-MPPI) runs what was learned on the floating hand.

    python -m bep.validate --scene runs/pair_cubes --spec pair_cubes

Babble, explore and play move the floating hand with the Hand-of-God motor.  The primitives' fields and
roadmaps live in the relational state z, which does not depend on the body, so the robot can execute them
as they are; this phase measures and records how well it does:

1. every primitive run by Spike-MPPI from ``VALIDATE['tests']`` fresh starts (as exploration's executor
   test), next to its Hand-of-God score: the gap between the ideal hand and the robot, per primitive;
2. ``VALIDATE['calls']`` play calls on the robot (chained from random layouts, chosen as in play), recorded
   in ``play_robot.jsonl`` with source "robot" and weight ``VALIDATE['weight']``.  Solving with the robot
   (bep.brain.solve --motor robot) fits the effect models on the floating hand's play and these calls.

Results -> <scene>/validation.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .config import EXPLORE, HOG, VALIDATE, progress, task_seed

_V: dict = {}


def _init(scene: str, threads: int) -> None:
    from .explore.library import load_library
    from .world.objects import use_body
    use_body("robot")
    R, lib = load_library(scene)
    _V.update(R=R, lib=lib, threads=threads)


def _prim_test(task) -> dict:
    """One primitive from one fresh start, on the robot."""
    from .motor.execute import execute_call
    from .motor.mppi import MPPIConfig, SpikeMPPI
    target, seed = task
    prim, field, _ = _V["lib"][target]
    _, env, _, _ = prim.sample_start(np.random.default_rng(seed))
    target_body = getattr(env, "ref", "B") if prim.kind == "object-object" else None
    ctl = SpikeMPPI(env, MPPIConfig(threads=_V["threads"], seed=seed, horizon=HOG["horizon"]))
    try:
        out = execute_call(env, _V["R"], prim, field, ctl, "A", target_body, timeout=EXPLORE["exec_timeout_s"])
    finally:
        ctl.close()
    return {"target": target, "done": out["result"] == "done", "result": out["result"], "time": out["time"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="runs/pair_cubes", help="directory of the scene's learning")
    ap.add_argument("--spec", required=True, help="scene name (bep.world.scenes.SCENES)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tests", type=int, default=VALIDATE["tests"])
    ap.add_argument("--calls", type=int, default=VALIDATE["calls"])
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    from .play.effect import load
    from .play.play import _execute_task, _init as play_init
    scene = Path(args.scene)
    t0 = time.time()
    prims = [p["target"] for p in json.loads((scene / "primitives.json").read_text())]
    report: dict = {"primitives": {}}
    # 1. every primitive from fresh starts
    tasks = [(t, task_seed(args.seed, 1000 * k + i)) for k, t in enumerate(prims) for i in range(args.tests)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(scene), args.threads)) as pool:
        res = []
        for n, r in enumerate(pool.imap_unordered(_prim_test, tasks)):
            res.append(r)
            if n % 8 == 7:
                progress(n + 1, len(tasks) + args.calls, "primitives on the robot")
    for t in prims:
        rs = [r for r in res if r["target"] == t]
        ex = scene / f"prim_{t}" / "executor.json"
        report["primitives"][t] = {
            "robot": round(float(np.mean([r["done"] for r in rs])), 3),
            "hand_of_god": json.loads(ex.read_text())["success"] if ex.exists() else None,
            "results": {k: sum(r["result"] == k for r in rs) for k in sorted({r["result"] for r in rs})}}
        print(t, json.dumps(report["primitives"][t]), flush=True)
    # 2. play calls on the robot
    out = scene / "play_robot.jsonl"
    out.unlink(missing_ok=True)
    history = load(scene / "play.jsonl")
    recs: list = []
    with mp.get_context("spawn").Pool(args.workers, initializer=play_init,
                                      initargs=(str(scene), args.spec, args.threads, "robot")) as pool:
        k = 0
        while len(recs) < args.calls:
            n = int(min(args.workers, max(1, np.ceil((args.calls - len(recs)) / 4))))   # ~4 calls per episode
            batch = [(history + recs, task_seed(args.seed + 7, k + i)) for i in range(n)]
            k += n
            for got, _ in pool.map(_execute_task, batch):
                recs += got
            progress(len(tasks) + min(len(recs), args.calls), len(tasks) + args.calls, "play calls on the robot")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(dict(r, source="robot", weight=VALIDATE["weight"])) + "\n")
    per = {}
    for r in recs:
        per.setdefault(r["primitive"], []).append(r["success"])
    report["robot_play"] = {t: f"{sum(v)}/{len(v)}" for t, v in sorted(per.items())}
    hog = {}
    for r in history:
        hog.setdefault(r["primitive"], []).append(r["success"])
    report["hand_of_god_play"] = {t: f"{sum(v)}/{len(v)}" for t, v in sorted(hog.items())}
    report["wall_s"] = round(time.time() - t0, 1)
    (scene / "validation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
