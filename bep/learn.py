"""The whole learning stage for one scene, in one command: babble -> explore -> play.

    python -m bep.learn --scene pair_cubes            # -> runs/<scene>/

Phase 1  Babble    random Hand-of-God configurations, grasp and motion probes -> relations, and which
                   of them get a primitive (bep.babble.select)
Phase 2  Explore   reverse Go-Explore toward each, state and hand-object primitives first, then the
                   others, seeded by the goals of the first (each stops by itself)
Phase 3  Play      executed calls of every primitive -> effect models
Learning moves only the floating hand, with the Hand-of-God motor (no sampling planner), within the
robot's reachability map; Spike-MPPI comes in only when tasks are executed (bep.brain.solve).
Progress is printed as "Phase k/3 <name>: ..."; timings -> runs/<scene>/timing.json.  The README clips are
rendered afterwards (bep.render), outside the timed phases.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .babble.select import choose_primitives
from .config import EXPLORE
from .world.relations import RelationSet

N_PHASES = 3


def say(phase: int, name: str, msg: str) -> None:
    print(f"Phase {phase}/{N_PHASES} {name}: {msg}", flush=True)


def run_phase(phase: int, name: str, args: list[str], log: Path) -> float:
    """Run one module, relaying its @progress lines; return the wall time."""
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-u", "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            if line.startswith("@progress "):
                d = json.loads(line[len("@progress "):])
                say(phase, name, f"{d['done']} / {d['total']}  {d['msg']}")
        p.wait()
    if p.returncode:
        raise SystemExit(f"{name} failed, see {log}")
    say(phase, name, f"done ({time.time() - t0:.0f} s)")
    return round(time.time() - t0, 1)


def learn(scene: str, out: Path, *, probes: int = 30000, max_rounds: int = 300, cores: int | None = None,
          clips: bool = True, reuse_probe: bool = False) -> dict:
    import numpy as np

    from .explore.archive import Archive
    from .explore.explore import NoGoalSeed, seed_goals
    from .explore.primitive import DiscoveredPrimitive
    from .world.objects import body_kind, use_body
    from .world.scenes import SCENES, scene_kinds, scene_pairs
    cores = cores or os.cpu_count()
    out.mkdir(parents=True, exist_ok=True)
    shapes, fixtures = scene_kinds(SCENES[scene])
    pairs = scene_pairs(SCENES[scene])
    (out / "learning.json").write_text(json.dumps({"scene": scene, "pairs": pairs}, indent=1))
    rel, probe_file = out / "relations", out / "probes.npz"
    timing: dict = {}
    t0 = time.time()
    cmd = ["bep.babble.probe", "--probes", str(probes), "--workers", str(cores), "--shapes", ",".join(shapes) or "-",
           "--pairs", ",".join(f"{x}/{y}" for x, y in pairs),
           "--clips", "0", "--out", str(rel)] + (["--fixtures", ",".join(fixtures)] if fixtures else [])
    full = out / "relations_all"
    if reuse_probe and probe_file.exists() and (full.with_suffix(".json").exists() or rel.with_suffix(".json").exists()):
        say(1, "Babble", "reusing the saved probe")
        timing["probe_s"] = json.loads((out / "timing.json").read_text()).get("probe_s") \
            if (out / "timing.json").exists() else None
    else:
        timing["probe_s"] = run_phase(1, "Babble", cmd, out / "probe.log")
        full.with_suffix(".json").unlink(missing_ok=True)

    R = RelationSet.load(full if full.with_suffix(".json").exists() else rel)
    R.save(full)                                    # every discovered relation, kept for reference
    prims = choose_primitives(R, str(probe_file))
    # the scene's vocabulary: the relations a primitive makes.  Modes (and so the fields' graphs) are made of
    # these only; probe clusters no primitive acts on (e.g. a palm brushing a body) would split every graph
    body = body_kind()
    use_body("float")
    try:
        vocabulary = RelationSet([R.get(p["target"]) for p in prims], R._rest_tilt)
        while prims:
            viable = []
            for p in prims:
                prim = DiscoveredPrimitive(vocabulary, p["target"], kinds=p["kinds"], probes=probe_file)
                try:
                    seed_goals(Archive(), prim, np.random.default_rng(0), limit=1)
                except NoGoalSeed as exc:
                    say(1, "Babble", str(exc))
                else:
                    viable.append(p)
            if len(viable) == len(prims):
                break
            prims = viable
            vocabulary = RelationSet([R.get(p["target"]) for p in prims], R._rest_tilt)
    finally:
        use_body(body)
    if not prims:
        raise SystemExit("No discovered primitive had a reproducible goal state")
    vocabulary.save(rel)
    (out / "primitives.json").write_text(json.dumps(prims, indent=1))
    say(1, "Babble", f"{len(R)} relations discovered, {len(prims)} primitives: "
        + ", ".join(f"{p['target']} ({p['kind']}, {'/'.join(p['kinds'])})" for p in prims))

    timing["explore_s"] = {}
    done: list[str] = []
    for i, p in enumerate(prims):
        chain = ",".join(f"{q}={out / f'prim_{q}'}" for q in done
                         if next(x for x in prims if x["target"] == q)["stage"] < p["stage"])
        target = p["target"]
        folder = out / f"prim_{target}"
        label = f"Explore {i + 1}/{len(prims)} [{target}: {p['kind']}]"
        cmd = ["bep.explore.explore", "--relations", str(rel), "--probes", str(probe_file), "--target", target,
               "--rounds", str(max_rounds), "--workers", str(cores), "--per-worker", str(max(1, 16 // cores))]
        if chain:
            cmd += ["--chain", chain]
        elapsed = run_phase(2, label, [*cmd, "--out", str(folder)], out / f"explore_{target}.log")
        report = json.loads((folder / "executor.json").read_text())
        if report["success"] < EXPLORE["retry_below"]:
            say(2, label, f"trying a new graph after {report['success']:.0%} executor success")
            with tempfile.TemporaryDirectory(prefix=f"retry_{target}_", dir=out) as pending:
                retry = Path(pending)
                elapsed += run_phase(2, f"{label} retry", [*cmd, "--seed", "1", "--out", str(retry)],
                                     out / f"explore_{target}_retry.log")
                alternative = json.loads((retry / "executor.json").read_text())
                attempts = [report["success"], alternative["success"]]
                if alternative["success"] > report["success"]:
                    for file in retry.iterdir():
                        shutil.copy2(file, folder / file.name)
                    report = alternative
                report["attempt_success"] = attempts
                (folder / "executor.json").write_text(json.dumps(report))
                say(2, label, f"retained graph with {report['success']:.0%} executor success")
        timing["explore_s"][target] = round(elapsed, 1)
        done.append(target)

    cmd = ["bep.play.play", "--scene", str(out), "--spec", scene, "--workers", str(cores),
           "--clips", "0", "--out", str(out / "play.jsonl")]
    (out / "play.jsonl").unlink(missing_ok=True)
    timing["play_s"] = run_phase(3, "Play", cmd, out / "play.log")
    timing["total_s"] = round(time.time() - t0, 1)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"Learning done in {timing['total_s'] / 60:.1f} min -> {out}", flush=True)
    if clips:                                       # after learning: the timings above exclude rendering
        t1 = time.time()
        print("Rendering clips ...", flush=True)
        subprocess.run([sys.executable, "-m", "bep.render", "--scene", scene, "--run", str(out)],
                       stdout=subprocess.DEVNULL, check=True)
        print(f"Clips rendered in {time.time() - t1:.0f} s -> {out / 'clips'}", flush=True)
    return timing


def main() -> None:
    from .world.scenes import SCENES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--out", default=None, help="default runs/<scene>")
    ap.add_argument("--probes", type=int, default=30000)
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--no-clips", action="store_true", help="skip rendering the README clips after learning")
    ap.add_argument("--reuse-probe", action="store_true", help="keep the saved probe (phases 2-4 only)")
    args = ap.parse_args()
    learn(args.scene, Path(args.out or f"runs/{args.scene}"), probes=args.probes, max_rounds=args.max_rounds,
          cores=args.cores, clips=not args.no_clips, reuse_probe=args.reuse_probe)


if __name__ == "__main__":
    main()
