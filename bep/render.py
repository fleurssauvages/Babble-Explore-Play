"""Render the README clips of a learned scene, after learning (so the learning times exclude rendering).

    python -m bep.render --scene pair_cubes          # -> runs/<scene>/clips/*.mp4

probe_*.mp4      probes with an interaction (grasp and motion probes)
probing/<rel>.mp4   one probe per discovered relation, a probe that brought it about (also as
                 assets/probing/<scene>_<relation>_<kind>.gif)
explore_*.mp4    each primitive run by Spike-MPPI on its field, from the farthest nodes of its graph
play_*.mp4       executed play episodes (chained primitive calls)
solve_*.mp4      the first episode of each zero-shot task (bep.brain.solve --clip)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def render(scene: str, run: Path, *, probes: int = 4, explore: int = 2, play: int = 2, seed: int = 7) -> None:
    from .babble import probe as pr
    from .explore import explore as ex
    from .play import play as pl
    from .explore.archive import Archive
    from .explore.primitive import DiscoveredPrimitive
    from .world.relations import RelationSet
    from .world.scenes import SCENES, scene_kinds, scene_pairs
    clips = run / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    shapes, fixtures = scene_kinds(SCENES[scene])
    print("probe clips", flush=True)
    pr.record_clips(clips, probes, list(shapes), list(fixtures), seed, scene_pairs(SCENES[scene]))
    R = RelationSet.load(run / "relations")
    print("one probe clip per discovered relation -> assets/probing", flush=True)
    from .report import stitch
    got = pr.record_relation_clips(R, clips / "probing", list(shapes), list(fixtures), scene_pairs(SCENES[scene]),
                                   seed)
    for name, mp4 in got.items():
        r = R.get(name)
        stitch([mp4], Path("assets/probing") / f"{scene}_{name}_{r.kind}.gif", max_s=8.0)
    missing = sorted({r.name for r in R} - set(got))
    print(f"  {len(got)} relation clips" + (f", none found for {missing}" if missing else ""), flush=True)
    for p in json.loads((run / "primitives.json").read_text()):
        print("explore clips", p["target"], flush=True)
        prim = DiscoveredPrimitive(R, p["target"], probes=str(run / "probes.npz"))
        mt = run / f"prim_{p['target']}" / "metric.json"
        if mt.exists():
            prim.set_metric({int(m): w for m, w in json.loads(mt.read_text()).items()})
        ex.record_clips(Archive.load(run / f"prim_{p['target']}" / "archive.npz"), prim,
                        clips / f"explore_{p['target']}", explore)
    print("play clips", flush=True)
    pl.record_clips(SimpleNamespace(scene=str(run), spec=scene, seed=seed), clips, play)


def main() -> None:
    from .world.scenes import SCENES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--run", default=None, help="default runs/<scene>")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    render(args.scene, Path(args.run or f"runs/{args.scene}"))


if __name__ == "__main__":
    main()
