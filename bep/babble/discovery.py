"""Babble blocks and the relations read from them.

``babble_block`` runs one block of probes on the pool (half of them restarting from the novelty archive
once it has events) and adds them to the event store.  ``read_vocabulary`` reads the relations from the
store (bep.babble.events), chooses the primitives (bep.babble.select, as the original pipeline) and keeps the
primitives whose goal can be reproduced; it writes the learning directory files the rest of the pipeline
reads (relations, relations_all, probes.npz, primitives.json, learning.json).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import BABBLE
from ..world.relations import RelationSet
from .events import Events, relations_from_events, save_probes


def scene_info(scene: str) -> dict:
    from ..world.scenes import SCENES, scene_kinds, scene_pairs
    shapes, fixtures = scene_kinds(SCENES[scene])
    return {"name": scene, "shapes": shapes, "fixtures": fixtures, "pairs": scene_pairs(SCENES[scene])}


def babble_block(pool, ev: Events, rng, seed: int, n: int | None = None, schema_bases=None) -> dict:
    """One block of probes; returns its statistics (novelty: share of its events with rarely seen keys).
    ``schema_bases(k)``: k states the schemas reached, a share BABBLE['schema_p'] of the restarts start there."""
    n = n or BABBLE["block"]
    start_row, t0 = len(ev), time.time()
    first = len(ev.recs)
    ids = list(range(first, first + n))
    restart = [rng.random() < BABBLE["restart_p"] for _ in ids] if len(ev) else [False] * n
    k = sum(restart)
    from_schemas = schema_bases(int(round(BABBLE["schema_p"] * k))) if schema_bases is not None else []
    bases = from_schemas + ev.restart_bases(rng, k - len(from_schemas))
    rng.shuffle(bases)
    it = iter(bases)
    per = 8
    tasks = []
    for a in range(0, n, per):
        chunk = ids[a:a + per]
        tasks.append({"seed": seed, "ids": chunk,
                      "bases": [next(it, None) if r else None for r in restart[a:a + per]]})
    recs = [r for out in pool.map("bep.babble.probe:probe", tasks) for r in out]
    ev.add(recs)
    return {"probes": len(recs), "events": len(ev) - start_row, "novelty": round(ev.novelty(start_row), 4),
            "wall_s": round(time.time() - t0, 2), "sim_s": round(sum(r["sim_s"] for r in recs), 1)}


def read_vocabulary(ev: Events, out: Path, scene: dict, old: RelationSet | None = None, *, check_goals: bool = True):
    """Relations from the event store, primitives, and the learning directory files.  Returns
    (all relations, vocabulary, primitive specs, report)."""
    from .select import choose_primitives, primitive_relations, vocabulary_of
    from ..explore.archive import Archive
    from ..explore.explore import NoGoalSeed, seed_goals
    from ..explore.primitive import DiscoveredPrimitive
    out.mkdir(parents=True, exist_ok=True)
    save_probes(ev, out / "probes.npz")
    R, report = relations_from_events(ev, old)
    R.save(out / "relations_all")
    (out / "learning.json").write_text(json.dumps({"scene": scene["name"], "pairs": scene["pairs"]}, indent=1))
    prims = choose_primitives(R, str(out / "probes.npz")) if len(R) else []
    vocabulary = vocabulary_of(R, prims)
    while prims and check_goals:                        # keep primitives whose goal states can be reproduced
        viable = []
        for p in prims:
            prim = DiscoveredPrimitive(primitive_relations(vocabulary, p["target"]), p["target"], kinds=p["kinds"],
                                       probes=out / "probes.npz")
            try:
                seed_goals(Archive(), prim, np.random.default_rng(0), limit=1)
            except NoGoalSeed:
                continue
            viable.append(p)
        if len(viable) == len(prims):
            break
        prims = viable
        vocabulary = vocabulary_of(R, prims)
    vocabulary.save(out / "relations")
    (out / "primitives.json").write_text(json.dumps(prims, indent=1))
    report["primitives"] = [p["target"] for p in prims]
    (out / "events_report.json").write_text(json.dumps(report, indent=1, default=float))
    return R, vocabulary, prims, report
