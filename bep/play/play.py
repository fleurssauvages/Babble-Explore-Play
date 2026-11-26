"""Play: chained executed calls of the schemas, in blocks the arbiter schedules, to learn what each one does.

Each episode starts from a random layout of the scene; calls then follow each other, each from where the
previous one ended (so stacks, held bodies, ... arise by themselves).  A candidate call must lie where its
schema has validated experience (bep.explore.field.covered), must not try to make a relation that already
holds, and, once the schema has an effect model, must be predicted to change the relation graph.  Calls in a
relational context tried fewer than ``min_per_context`` times come first; otherwise calls are chosen where the
bootstrap ensemble of success models disagrees most; a share is chosen at random.  Calls run with the
learning's executor on the floating hand (bep.motor.execute.execute_call).

Each call is recorded (schema, roles, relation graph and poses before and after, outcome, time, labelled with
bep.play.effect.call_succeeded) together with its context at the start: the contact flags of its roles
(bep.babble.events.flags), Phi and the distance to the schema's validated graph (``flags0``, ``phi0``,
``hop0``: inputs of the preconditions, bep.play.schemas).  A failed call's starting pair view is handed back
to its schema's exploration as a start state (bep.explore.explore.Explorer).

A block's competence is 1 - the prequential log-loss of its calls (each predicted from the calls before it)
over log 2: how well play already predicts what its calls do.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import PLAY, REACH, task_seed
from ..babble.events import flags
from ..explore.field import covered
from ..explore.library import bindings, load_library, resolve_target
from ..motor.execute import execute_call
from ..world.geometry import Spot
from ..world.graph import as_tuples, object_poses, relation_graph
from ..world.objects import VIEW_S, apply_layout
from ..world.scenes import random_layout
from .effect import EffectModel, call_succeeded, load, made_instance


# ------------------------------------------------------------------ helpers
def _record(env, prim, x, target, g0, p0, g1, p1, success, result, t):
    tgt = [target.x, target.y] if isinstance(target, Spot) else target
    return {"objects": env.cubes, "parts": sorted(env.fixtures), "shapes": dict(env.kinds),
            "primitive": prim.target, "x": x, "target": tgt,
            "before": sorted(map(list, as_tuples(g0))), "after": sorted(map(list, as_tuples(g1))),
            "pose_before": {k: np.round(v, 4).tolist() for k, v in p0.items()},
            "pose_after": {k: np.round(v, 4).tolist() for k, v in p1.items()},
            "source": "executed", "success": bool(success), "result": result, "time": t}


def _left_workspace(env, poses) -> bool:
    return any(poses[n][2] < 0 or np.hypot(poses[n][0], poses[n][1]) > REACH[1] for n in env.free)


def context(proto, rels, poses, call) -> tuple:
    """The relational context of a call: what the success model sees (relations around X and its target, the
    hand), e.g. "grasp X while a body rests on X" differs from "grasp a free X"."""
    t, x, y = call
    return (t, tuple(np.round(proto[t].phi_now(rels, poses, x, _tgt(y))[0], 3).tolist()))


def _models(R, lib, history, env) -> dict:
    """One effect model per primitive with at least min_fit executed calls."""
    out = {}
    for t in lib:
        recs = [h for h in history if h["primitive"] == t]
        if len(recs) >= PLAY["min_fit"]:
            out[t] = EffectModel.for_relations(R, t).fit(recs).set_shapes(
                {n: env.shape(n) for n in env.cubes}, sorted(env.fixtures))
    return out


def _tgt(target):
    return target if isinstance(target, str) or target is None else [target.x, target.y]


def _chainable(env, lib, models, rels, call) -> bool:
    """A call that can follow from here: the primitive's experience covers the current state (apart from what
    it is meant to change), and, once a model exists, it is predicted to change the relation graph."""
    t, x, target = call
    prim, field, _ = lib[t]
    ref = prim.reference(env, x, target)
    rel = prim.rel(env, env.pair_view(env.sensors(), x, ref), VIEW_S)
    if not covered(prim, field, rel["z"], int(rel["mode"]), pair=prim.role_pair(env, x, target)):
        return False
    if t in models and models[t].most_likely(rels, env.cubes, x, _tgt(target), True) == set(rels):
        return False
    return True


def prequential(R, history, recs) -> list[tuple]:
    """(primitive, log-loss, fitted) of each new executed call, its success predicted from the earlier calls
    (``history``: the latest PLAY['history_cap'], as the play models);
    fitted = the primitive had a model (at least min_fit calls) at that point."""
    out, training = [], {}
    for r in recs:
        t = r["primitive"]
        if t not in training:
            hist = [h for h in history if h["primitive"] == t]
            training[t] = None
            if len(hist) >= PLAY["min_fit"]:
                m = EffectModel.for_relations(R, t)
                executed = [h for h in hist if h.get("source", "executed") in ("executed", "robot")]
                training[t] = (m, [m.phi(h) for h in executed], [h["success"] for h in executed],
                               [h.get("weight", 1.0) for h in executed])
        if training[t] is None:
            out.append((t, float(np.log(2.0)), False))
            continue
        m, X, y, weights = training[t]
        m.fit_success(np.asarray(X), np.asarray(y, float), np.asarray(weights, float))
        features = m.phi(r)
        p = float(np.clip(m.success.predict(features[:m.nrel])[0], 1e-4, 1 - 1e-4))
        out.append((t, -np.log(p) if r["success"] else -np.log(1 - p), True))
        if r.get("source", "executed") in ("executed", "robot"):
            X.append(features)
            y.append(r["success"])
            weights.append(r.get("weight", 1.0))
    return out



# ------------------------------------------------------------------ episodes and blocks
def library_version(run) -> tuple:
    """What the workers' library is built from: every .json and .npz file of the learning directory and of its
    schemas' folders (time and size; the append-only play records are not among them).  Workers rebuild the
    library only when it changes."""
    run = Path(run)
    files = sorted(f for d in [run, *run.glob("prim_*")] for f in d.iterdir()
                   if f.suffix in (".json", ".npz") and f.is_file())
    out = []
    for f in files:
        try:
            st = f.stat()
        except FileNotFoundError:                 # replaced while listing
            continue
        out.append((str(f), st.st_mtime_ns, st.st_size))
    return tuple(out)


def _library(run: str, version, spec_name: str):
    from ..workers import CACHE
    key = ("lib", run, version)
    if CACHE.get("lib_key") != key:
        from ..world.scenes import SCENES, build_scene
        env = CACHE.get("play_env")
        if env is None:
            env = CACHE["play_env"] = build_scene(SCENES[spec_name], "loopplay")
        R, lib = load_library(run)
        if (Path(run) / "qmap.npz").exists():                   # the schemas' Phi from the map where trusted
            from ..explore.qmap import QMap, wrap_library
            wrap_library(lib, QMap(Path(run) / "qmap.npz"))
        proto = {t: EffectModel.for_relations(R, t).set_shapes({n: env.shape(n) for n in env.cubes}, sorted(env.fixtures))
                 for t in lib}
        from .schemas import load_preconditions
        CACHE["pre"] = load_preconditions(Path(run) / "preconditions.npz") if PLAY["pre_filter"] > 0 else {}
        CACHE["lib"], CACHE["lib_key"] = (env, R, lib, proto), key
    return CACHE["lib"]


def call_and_record(env, R, lib, t: str, x: str, target) -> tuple[dict, dict]:
    """Execute one call with the learning's executor and record it (bep.play.play's record, plus the context
    at its start); returns (record, start: the pair view and command it started from, for exploration)."""
    prim, field, _ = lib[t]
    g0, p0 = relation_graph(env, R), object_poses(env)
    view0 = env.pair_view(env.sensors(), x, prim.reference(env, x, target))
    ctrl0 = env.data.ctrl.copy()
    rel0 = prim.rel(env, view0, VIEW_S)
    z0, m0 = rel0["z"], int(rel0["mode"])
    phi0 = float(field.value(z0, m0))                               # how far the schema thinks the goal is
    hop0 = float(field.roadmap.near(z0, m0))                         # how far its validated graph is (typical hops)
    out = execute_call(env, R, prim, field, None, x, target, timeout=PLAY["timeout_s"], learning=True)
    g1, p1 = relation_graph(env, R), object_poses(env)
    ok = call_succeeded(out["result"], made_instance(prim.kind, t, x, target), as_tuples(g0), as_tuples(g1))
    rec = _record(env, prim, x, target, g0, p0, g1, p1, ok, out["result"], out["time"])
    rec.update(flags0=[int(v) for v in flags(view0)], phi0=round(phi0, 3),
               hop0=round(hop0, 3) if np.isfinite(hop0) else None)
    start = {"primitive": t, "ok": ok, "view": view0, "ctrl": ctrl0, "pair": prim.role_pair(env, x, target),
             "flags0": rec["flags0"], "before": rec["before"], "x": x, "target": rec["target"]}
    return rec, start


def _plausible(env, lib, proto, rels, poses, cands):
    """The candidate calls whose schema's precondition predicts at least PLAY['pre_filter'] success (all of them
    when none does, or before any precondition was learned)."""
    from ..workers import CACHE
    from .schemas import context_record, features
    pre = CACHE.get("pre") or {}
    if PLAY["pre_filter"] <= 0 or not pre:
        return cands
    keep = []
    for c in cands:
        t, x, target = c
        if t not in pre:
            keep.append(c)
            continue
        prim, field, _ = lib[t]
        rec = context_record(env, prim, field, x, target, rels, poses)
        p, _ = pre[t].predict(features(proto[t], rec)[None])
        if p[0] >= PLAY["pre_filter"]:
            keep.append(c)
    return keep or cands


def episode(task: dict) -> dict:
    """One play episode (bep.play.play._execute_task) that also records each call's starting pair view."""
    env, R, lib, proto = _library(task["run"], task["version"], task["scene"])
    history = task["history"]
    rng = np.random.default_rng(task["seed"])
    models = _models(R, lib, history, env)
    ens = {t: m.ensemble(PLAY["ensemble"], rng) for t, m in models.items()}
    count = Counter(h["primitive"] for h in history)
    ctx_count = Counter(tuple((h["context"][0], tuple(h["context"][1]))) for h in history if "context" in h)
    layout, joints = random_layout(env, rng)
    apply_layout(env, layout, joints=joints)
    recs, starts = [], []
    for _ in range(PLAY["chain_max"]):
        g0, p0 = relation_graph(env, R), object_poses(env)
        rels = as_tuples(g0)
        cands = [(t, x, resolve_target(env, y, rng)) for t in lib for x, y in bindings(env, lib[t][0])]
        cands = [c for c in cands if made_instance(lib[c[0]][0].kind, *c) not in rels]
        cands = [c for c in cands if _chainable(env, lib, models, rels, c)]
        cands = _plausible(env, lib, proto, rels, p0, cands)
        if not cands:
            break
        ctx = {id(c): context(proto, rels, p0, c) for c in cands}
        rare = [c for c in cands if c[0] not in models or ctx_count[ctx[id(c)]] < PLAY["min_per_context"]
                or count[c[0]] < PLAY["min_per_prim"]]
        if rare and rng.random() < PLAY["rare_p"]:
            cands = rare
        scored = [c for c in cands if c[0] in models]
        if scored and rng.random() >= PLAY["random_frac"]:
            var = np.array([np.var([e.predict(models[t].phi_now(rels, p0, x, _tgt(y)))[0] for e in ens[t]])
                            for t, x, y in scored])
            call = scored[int(np.argmax(var + 1e-4 * rng.random(len(var))))]
        else:
            call = cands[int(rng.integers(len(cands)))]
        t, x, target = call
        cx = ctx[id(call)]
        rec, start = call_and_record(env, R, lib, t, x, target)
        rec.update(context=[cx[0], list(cx[1])], episode=task["seed"], step=len(recs))
        recs.append(rec)
        starts.append(start)
        p1 = object_poses(env)
        count[t] += 1
        ctx_count[cx] += 1
        if _left_workspace(env, p1):
            break
    return {"recs": recs, "starts": starts}


class Player:
    """Play blocks on the pool; the records accumulate in the run's play.jsonl."""

    def __init__(self, run: Path, scene: str, seed: int = 0):
        self.run, self.scene, self.seed = Path(run), scene, seed
        self.path = self.run / "play.jsonl"
        self.history: list = []                  # the most recent PLAY['history_cap'] calls (what models are fit on)
        self.calls: Counter = Counter()           # calls per primitive over the whole run
        if self.path.exists():
            self.add(load(self.path))
        self.n = 0

    def add(self, recs: list) -> None:
        """New calls (already written to play.jsonl): counted, and kept while among the latest history_cap."""
        self.calls.update(r["primitive"] for r in recs)
        self.history += recs
        del self.history[:-PLAY["history_cap"]]

    def run_block(self, pool, n: int | None = None) -> tuple[dict, dict]:
        """One block of episodes; returns (statistics, failed call starts per primitive as (view, ctrl, pair))."""
        n = n or pool.n
        R, lib = load_library(str(self.run))
        version = library_version(self.run)
        tasks = [{"run": str(self.run), "version": version, "scene": self.scene, "history": self.history,
                  "seed": task_seed(self.seed + 7, self.n * 1000 + k)} for k in range(n)]
        self.n += 1
        outs = pool.map("bep.play.play:episode", tasks)
        recs = [r for o in outs for r in o["recs"]]
        starts = [s for o in outs for s in o["starts"]]
        preq = prequential(R, self.history, recs)
        fitted = [l for _, l, f in preq if f]
        for r in recs:
            r["epoch"] = getattr(self, "epoch", None)
        with open(self.path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        self.add(recs)
        fails: dict = {}
        for s in starts:
            if not s["ok"]:
                fails.setdefault(s["primitive"], []).append((s["view"], s["ctrl"], s["pair"], s))
        by = Counter((r["primitive"], r["success"]) for r in recs)
        stats = {"calls": len(recs), "success": {t: f"{by[(t, True)]}/{by[(t, True)] + by[(t, False)]}" for t in lib},
                 "logloss": round(float(np.mean(fitted)), 4) if fitted else None,
                 "competence": round(1.0 - float(np.mean(fitted)) / np.log(2.0), 4) if fitted else 0.0}
        return stats, fails
