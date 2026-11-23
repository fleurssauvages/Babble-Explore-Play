"""Play: task-free executed calls of every primitive, to learn what each one does.

    python -m bep.play.play --scene runs/pair_cubes --spec pair_cubes --out runs/pair_cubes/play.jsonl

Episodes start from a random layout (bodies apart on the table).  Calls then follow each other, each
from where the previous one ended (so stacks, held bodies, ... arise by themselves): a candidate call
must lie where its primitive has validated experience (bep.explore.field.covered), must not try to make
a relation that already holds, and, once the primitive has an effect model, must be predicted to change
the relation graph.  Calls in a relational context (what the success model sees: the relations around
X and its target) tried fewer than ``min_per_context`` times come first; otherwise calls are chosen where
the bootstrap ensemble of success models disagrees most (what the model does not know yet); a share is
chosen at random.

Calls run with the very executor of the tasks (bep.motor.execute.execute_call): a call that breaks a
relation it should keep (e.g. grasping a body that supports another) stops as "broken", so play learns
what the tasks will meet.  Each executed call is recorded (primitive, roles, relation graph and poses before and after, outcome,
time) and labelled with bep.play.effect.call_succeeded: it finished and made its relation.

Stop: every primitive has at least ``min_per_prim`` calls, every context met has ``min_per_context``, and the prequential log-loss of the success
predictions (each call predicted from the calls before it) of fitted primitives no longer improves
over the last ``stop_window`` calls; or ``max_calls``.  Calls of primitives without a model yet do not
count toward convergence (their loss is a constant log 2, which would look like a plateau).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import PLAY, REACH, progress, task_seed
from ..explore.field import covered
from ..explore.library import bindings, load_library, resolve_target
from ..motor.execute import execute_call
from ..world.geometry import Spot
from ..world.graph import as_tuples, object_poses, relation_graph
from ..world.objects import VIEW_S, apply_layout, use_body
from ..world.scenes import random_layout
from .effect import EffectModel, call_succeeded, load, made_instance

_P: dict = {}


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


def _init(scene, spec_name, threads, body="float"):
    from ..motor.mppi import MPPIConfig, SpikeMPPI
    from ..world.scenes import SCENES, build_scene
    use_body(body)
    env = build_scene(SCENES[spec_name], f"play{os.getpid()}")
    R, lib = load_library(scene)
    proto = {t: EffectModel.for_relations(R, t).set_shapes({n: env.shape(n) for n in env.cubes}, sorted(env.fixtures))
             for t in lib}                         # unfitted models: the relational context of a call
    ctl = None if env.floating else SpikeMPPI(env, MPPIConfig(threads=threads, seed=0))
    _P.update(env=env, R=R, lib=lib, proto=proto, ctl=ctl)


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


def _execute_task(task):
    """One episode of real chains from a random layout: calls follow each other from where the previous one
    ended, until chain_max calls, nothing chainable, or a body left the workspace."""
    history, seed = task
    env, R, lib, ctl, proto = (_P[k] for k in ("env", "R", "lib", "ctl", "proto"))
    rng = np.random.default_rng(seed)
    if ctl is not None:
        ctl.rng = np.random.default_rng(seed + 1)
    models = _models(R, lib, history, env)
    ens = {t: m.ensemble(PLAY["ensemble"], rng) for t, m in models.items()}
    count = Counter(h["primitive"] for h in history)
    ctx_count = Counter(tuple((h["context"][0], tuple(h["context"][1]))) for h in history if "context" in h)
    seen = set()
    layout, joints = random_layout(env, rng)
    apply_layout(env, layout, joints=joints)
    if ctl is not None:
        ctl.reset(q_cmd=env.arm_q(), g_cmd=env.gripper_width())
    recs = []
    for _ in range(PLAY["chain_max"]):
        g0, p0 = relation_graph(env, R), object_poses(env)
        rels = as_tuples(g0)
        cands = [(t, x, resolve_target(env, y, rng)) for t in lib for x, y in bindings(env, lib[t][0])]
        cands = [c for c in cands if made_instance(lib[c[0]][0].kind, *c) not in rels]   # not made already
        cands = [c for c in cands if _chainable(env, lib, models, rels, c)]
        if not cands:
            break
        ctx = {id(c): context(proto, rels, p0, c) for c in cands}
        seen |= set(ctx.values())
        rare = [c for c in cands if c[0] not in models or ctx_count[ctx[id(c)]] < PLAY["min_per_context"]
                or count[c[0]] < PLAY["min_per_prim"]]
        if rare and rng.random() < PLAY["rare_p"]:          # contexts (and primitives) short of calls first
            cands = rare
        scored = [c for c in cands if c[0] in models]
        if scored and rng.random() >= PLAY["random_frac"]:
            feats = [models[t] for t, _, _ in scored]
            var = np.array([np.var([e.predict(m.phi_now(rels, p0, x, _tgt(y)))[0] for e in ens[t]])
                            for m, (t, x, y) in zip(feats, scored)])
            call = scored[int(np.argmax(var + 1e-4 * rng.random(len(var))))]
        else:
            call = cands[int(rng.integers(len(cands)))]
        t, x, target = call
        cx = ctx[id(call)]
        prim, field, _ = lib[t]
        out = execute_call(env, R, prim, field, ctl, x, target, timeout=PLAY["timeout_s"],
                           learning=env.floating)
        g1, p1 = relation_graph(env, R), object_poses(env)
        ok = call_succeeded(out["result"], made_instance(prim.kind, t, x, target), as_tuples(g0), as_tuples(g1))
        recs.append(_record(env, prim, x, target, g0, p0, g1, p1, ok, out["result"], out["time"]))
        recs[-1]["context"] = [cx[0], list(cx[1])]
        count[t] += 1
        ctx_count[cx] += 1
        if _left_workspace(env, p1):
            break
    return recs, [[c[0], list(c[1])] for c in seen]


# ------------------------------------------------------------------ stopping rule
def prequential(R, history, recs) -> list[tuple]:
    """(primitive, log-loss, fitted) of each new executed call, its success predicted from all earlier calls;
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


def play_converged(preq: list[tuple], counts: Counter, prims, n: int, rare: int = 0) -> tuple[bool, str]:
    """Every primitive has min_per_prim calls, every context met has min_per_context calls (``rare``: how
    many have not), and the prequential log-loss of fitted primitives no longer improves."""
    w = PLAY["stop_window"]
    short = [t for t in prims if counts[t] < PLAY["min_per_prim"]] + ([f"{rare} rare contexts"] if rare else [])
    fitted = [l for _, l, f in preq if f]
    if len(fitted) < 2 * w:
        return False, f"{len(fitted)} fitted calls" + (f", short of calls: {short}" if short else "")
    last, before = float(np.mean(fitted[-w:])), float(np.mean(fitted[-2 * w:-w]))
    msg = f"log-loss {before:.3f} -> {last:.3f}" + (f", short of calls: {short}" if short else "")
    return n >= PLAY["min_calls"] and not short and before - last < PLAY["stop_tol"], msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="runs/pair_cubes", help="directory of the scene's learning")
    ap.add_argument("--spec", required=True, help="scene name (bep.world.scenes.SCENES)")
    ap.add_argument("--actions", type=int, default=PLAY["max_calls"], help="maximum number of executed calls")
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--clips", type=int, default=0, help="rendered executed play episodes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or Path(args.scene) / "play.jsonl")
    use_body("float")
    history = load(out) if out.exists() else []
    t0 = time.time()
    R, lib = load_library(args.scene)
    counts = Counter(h["primitive"] for h in history)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.scene, args.spec, args.threads)) \
            as pool, open(out, "a") as f:
        done = queue.Queue()
        n_sub = [0]

        def submit():
            pool.apply_async(_execute_task, ((list(history), task_seed(args.seed, n_sub[0])),), callback=done.put,
                             error_callback=lambda e: done.put({"error": repr(e)}))
            n_sub[0] += 1
        for _ in range(args.workers):
            submit()
        preq: list[tuple] = []
        n_ex = len(history)
        met: set = set()                                   # relational contexts met as candidates
        ctx_n = Counter((h["context"][0], tuple(h["context"][1])) for h in history if "context" in h)
        while n_ex < args.actions:
            got = done.get()
            if isinstance(got, dict):
                raise RuntimeError(got["error"])
            recs, seen = got
            met.update((c[0], tuple(c[1])) for c in seen)
            preq += prequential(R, history, recs)
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            history += recs
            n_ex += len(recs)
            counts.update(r["primitive"] for r in recs)
            ctx_n.update((r["context"][0], tuple(r["context"][1])) for r in recs)
            rare = sum(ctx_n[c] < PLAY["min_per_context"] for c in met)
            stop, msg = play_converged(preq, counts, list(lib), n_ex, rare)
            progress(n_ex, args.actions, f"executed calls ({dict(counts)}) {msg}")
            print(f"{n_ex} executed calls ({time.time() - t0:.0f}s): "
                  f"{', '.join(r['primitive'] + ('+' if r['success'] else '-') for r in recs)} {msg}", flush=True)
            if stop and not args.no_stop:
                break
            submit()
        pool.terminate()
    if args.clips:
        record_clips(args, out.parent / "clips", args.clips)


def record_clips(args, path: Path, n: int = 2):
    """A few executed play episodes: chained calls, each covered by its primitive's experience from where the
    previous one ended (as in play)."""
    from ..world.clips import Recorder
    _init(args.scene, args.spec, os.cpu_count())
    env, R, lib, ctl = (_P[k] for k in ("env", "R", "lib", "ctl"))
    rng = np.random.default_rng(args.seed + 999)
    for i in range(n):
        layout, joints = random_layout(env, rng)
        apply_layout(env, layout, joints=joints)
        if ctl is not None:
            ctl.reset(q_cmd=env.arm_q(), g_cmd=env.gripper_width())
        rec = Recorder(env, path / f"play_executed_{i}.mp4", max_s=30.0)
        for _ in range(6):
            g0 = relation_graph(env, R)
            rels = as_tuples(g0)
            cands = [(t, x, resolve_target(env, y, rng)) for t in lib for x, y in bindings(env, lib[t][0])]
            cands = [c for c in cands if made_instance(lib[c[0]][0].kind, *c) not in rels]
            cands = [c for c in cands if _chainable(env, lib, {}, rels, c)]
            if not cands:
                break
            t, x, target = cands[int(rng.integers(len(cands)))]
            execute_call(env, R, lib[t][0], lib[t][1], ctl, x, target, timeout=6.0, on_step=lambda k: rec.frame(),
                         learning=env.floating)
        rec.save()


if __name__ == "__main__":
    main()
