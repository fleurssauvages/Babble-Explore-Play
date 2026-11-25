"""Assessment of learning directories (epoch snapshots), the same yardstick for every variant.

    python -m bep.assess --runs runs/pair_cubes/epoch_1,runs/pair_cubes/epoch_2 --tasks 3 --task-names tower,restack

Per learning directory:
* play: success rate of each schema over the play calls new in that directory (the calls since the previous
  directory of the same run, for loop epochs), and over all its calls;
* executor: each schema's last executor success (exploration's own test), and (``--exec n``) its success from n
  fixed start states (the same seeds for every directory: the comparable measure);
* tasks (``--tasks n``): bep.brain.solve on
  cubes4 (tower DCBA, restack DCBA>ABCD), n episodes on the floating hand (motor hog, plan once), success rate and median time;
The map's own calibration is reported at every sleep in epochs.jsonl (bep.explore.qmap).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def play_stats(d: Path, prev: Path | None) -> dict:
    lines = (d / "play.jsonl").read_text().splitlines() if (d / "play.jsonl").exists() else []
    n0 = len((prev / "play.jsonl").read_text().splitlines()) if prev is not None and (prev / "play.jsonl").exists() else 0
    out = {}
    for name, part in (("new", lines[n0:]), ("all", lines)):
        c = defaultdict(lambda: [0, 0])
        for ln in part:
            r = json.loads(ln)
            c[r["primitive"]][0] += r["success"]
            c[r["primitive"]][1] += 1
        out[name] = {t: f"{a}/{b} ({a / b:.0%})" for t, (a, b) in sorted(c.items()) if b}
    return out


def executor(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("prim_*")):
        if (p / "executor.json").exists():
            out[p.name[5:]] = round(json.loads((p / "executor.json").read_text())["success"], 3)
            out[p.name[5:] + "_nodes"] = int(len(np.load(p / "archive.npz")["z"]))
    return out


def executor_fixed(d: Path, n: int, pool) -> dict:
    """Each schema's executor (bep.motor.hog_motor.run_primitive_hop) from n fixed fresh starts."""
    from .config import EXPLORE, task_seed
    specs = json.loads((d / "primitives.json").read_text())
    out = {}
    for p in specs:
        t = p["target"]
        folder = d / f"prim_{t}"
        if not (folder / "archive.npz").exists():
            continue
        chain = {q["target"]: str(d / f"prim_{q['target']}") for q in specs if q["stage"] < p["stage"]}
        spec = {"folder": str(folder), "target": t, "kinds": list(p["kinds"]), "probes": str(d / "probes.npz"),
                "chain": chain}
        mt = folder / "metric.json"
        metric = {int(m): w for m, w in json.loads(mt.read_text()).items()} if mt.exists() else {}
        tasks = [dict(spec=spec, archive=str(folder / "archive.npz"), metric=metric, seed=task_seed(999, i),
                      timeout=EXPLORE["exec_timeout_s"], start=None) for i in range(n)]
        res = [r["exec"] for r in pool.map("bep.explore.explore:exec_test", tasks) if r["exec"] is not None]
        out[t] = round(float(np.mean(res)), 3) if res else None
    return out


def tasks(d: Path, n: int, seed: int = 100, task: str = "tower", imagine: bool = True) -> dict:
    cmd = [sys.executable, "-m", "bep.brain.solve", "--scene", "cubes4", "--task", task, *([] if imagine else ["--no-imagine"]),
           "--plan-once", "--motor", "hog", "--learning", str(d), "--episodes", str(n), "--seed", str(seed)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
    if line is None:
        return {"error": (p.stderr or p.stdout)[-400:]}
    s = json.loads(line[len("SUMMARY "):])
    return {"success_rate": s["success_rate"], "median_time_s": s["median_time_s"], "episodes": s["episodes"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="learning directories, comma-separated")
    ap.add_argument("--exec", type=int, default=0, help="fixed executor tests per schema (0: none)")
    ap.add_argument("--tasks", type=int, default=0, help="episodes per task (0: no tasks)")
    ap.add_argument("--task-names", default="tower", help="cubes4 tasks, comma-separated (tower, restack)")
    ap.add_argument("--no-imagine", action="store_true", help="the brain without rehearsals")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dirs = [Path(p) for p in args.runs.split(",")]
    rows = []
    pool = None
    if args.exec:
        from .babble.discovery import scene_info
        from .workers import Pool
        from .world.objects import use_body
        use_body("float")
        pool = Pool(scene_info(json.loads((dirs[0] / "learning.json").read_text())["scene"]))
    for d in dirs:
        prev = None
        if d.name.startswith("epoch_"):
            k = int(d.name.split("_")[1])
            cand = d.parent / f"epoch_{k - 1}"
            prev = cand if cand.exists() else None
        row = {"dir": str(d), "play": play_stats(d, prev), "executor": executor(d)}
        if pool is not None:
            row["executor_fixed"] = executor_fixed(d, args.exec, pool)
        if args.tasks:
            for task in args.task_names.split(","):
                row[task] = tasks(d, args.tasks, task=task, imagine=not args.no_imagine)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if pool is not None:
        pool.close()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
