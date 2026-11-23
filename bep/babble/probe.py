"""Babble: discover relations (affordances) with a Hand-of-God physics probe.

    python -m bep.babble.probe --probes 30000 --out runs/pair_cubes/relations

Each probe:
  1. draws a random pair of bodies (X, Y) of the scene and a random relational configuration in the
     bodies' own units (X around/above Y, the hand around/above X in any orientation, random finger
     opening and gripper command), realises it at a random world pose of Y (the arm teleported there)
     and lets physics settle it                                                       -> state S1
  2. runs a grasp probe (close the fingers, lift) or a motion probe (short random hand motions); every
     lift or motion is an *event*: the configuration and whether the body followed the hand,
  3. releases, retracts and lets the scene settle                                    -> state S2.

Probes run in batches.  After each batch the relations are rediscovered (bep.babble.discover); probing
stops once the set of primitive-worthy relations (bep.babble.select.worthy) is unchanged for
``PROBE['stable_checks']`` batches.  From the second batch on, ``focus_frac`` of the probes restart from
a perturbed copy of an earlier probe in which something happened (the body followed the hand or came
to rest against the other body): most random probes touch nothing, these concentrate on the boundary
of what the hand can do.  Every probe has its own seed (``config.task_seed``) and records are kept in
probe order, so a run is reproducible whatever the number of workers.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import mujoco
import numpy as np

from ..config import GRIP_MAX, PRIMS, PROBE, progress, task_seed
from ..world.geometry import hand_from_axis, random_rotation, rotvec_mat
from ..world.hog import GRIP_CLOSE, GRIP_HOLD, GRIP_OPEN, HAND_MIN_Z, TRAIN_REGION, GConfig, realize
from ..world.objects import FIXTURES, TYPES, VIEW_S, is_fixture, kind_shape, use_body
from ..world.reach import reach_map
from ..world.relations import to_metric_shapes
from .discover import created_relations, discover, relation_keys, same_relations
from .select import worthy

_E: dict = {}


# ------------------------------------------------------------------ configurations
def sample_config(rng, shapes, fixtures, pairs=None, *, grasp_seed: bool = False) -> GConfig:
    """A random relational configuration of two bodies of the scene and the hand, in the bodies' own units.
    X is a free body or (when the scene has fixtures) a moving part; Y any body.  ``pairs``: the (X, Y)
    kind pairs that exist in the scene (default: every combination of its kinds)."""
    P = PROBE
    if pairs:
        pair = tuple(pairs[int(rng.integers(len(pairs)))])
    else:
        subject = fixtures if fixtures and (not shapes or rng.random() < P["part_frac"]) else shapes
        pair = (str(rng.choice(subject)), str(rng.choice([*shapes, *fixtures])))
    sx, sy = kind_shape(pair[0]), kind_shape(pair[1])
    sweep = P["segments"] * P["segment"] * sx.length
    z0 = (sx.half[2] - sy.half[2]) / (sx.half[2] + sy.half[2])           # both resting on the table
    if rng.random() < P["near_frac"]:                                       # X near / above Y
        pn = np.array([rng.uniform(-P["near_xy"], P["near_xy"]), rng.uniform(-P["near_xy"], P["near_xy"]),
                       rng.uniform(z0, P["near_z"])])
    else:                                   # X apart on the table, at most one probe sweep beyond contact
        unit = float(sy.ext[0] + max(sx.ext[0], sx.ext[1]))             # contact distance (normalised = 1)
        a, r = rng.uniform(-np.pi, np.pi), rng.uniform(P["near_xy"], max(P["near_xy"], 1.0 + sweep / unit))
        pn = np.array([r * np.cos(a), r * np.sin(a), z0])
    if rng.random() < P["hand_near_frac"] and sx.kind == "compound":       # hand near one sub-part of the body
        h, o = sx.parts[int(rng.integers(len(sx.parts)))]
        h, o = np.array(h), np.array(o)
        pm = o + np.array([rng.normal(0, h[0]), rng.normal(0, h[1]), h[2] * rng.uniform(*P["hand_near_z"])])
        pen = pm / sx.ext
    elif rng.random() < P["hand_near_frac"]:                                # hand near X
        pen = np.array([rng.normal(0, P["hand_near_sd"]), rng.normal(0, P["hand_near_sd"]),
                        rng.uniform(*P["hand_near_z"])])
    else:                                   # hand farther from X, but within one probe sweep of it
        bx, by, bz = 1.0 + sweep / sx.ext[0], 1.0 + sweep / sx.ext[1], 1.0 + sweep / sx.ext[2]
        lo = P["hand_near_z"][1]
        pen = np.array([rng.uniform(-bx, bx), rng.uniform(-by, by), rng.uniform(lo, max(lo, bz))])
    pXY, pEX = to_metric_shapes(pn, pen, sx, sy)
    u = rng.random()
    grip = GRIP_CLOSE if u < 0.5 else (GRIP_OPEN if u < 0.75 else GRIP_HOLD)
    cfg = GConfig(pXY, sx.sample_yaw(rng), pEX, random_rotation(rng),
                  float(rng.uniform(0.0, GRIP_MAX)), grip, pair, _joint_sample(rng), _joint_sample(rng))
    if grasp_seed:
        seed_from_envelope(cfg, sx, rng)
    return cfg


def seed_from_envelope(cfg, shape, rng) -> None:
    """Sample a gripper pose around an opposing face pair that fits its aperture."""
    half, center = (shape.ext, np.zeros(3))
    if shape.parts:
        half, center = (np.asarray(v, float) for v in shape.parts[int(rng.integers(len(shape.parts)))])
    axes = np.flatnonzero(2 * half < GRIP_MAX)
    if not len(axes):
        return
    axis = int(rng.choice(axes))
    closing = np.eye(3)[axis]
    if shape.kind == "cylinder" and axis != 2:
        yaw = rng.uniform(-np.pi, np.pi)
        closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    cfg.pEX = center + half * rng.uniform(-0.5, 0.5, 3)
    cfg.pEX[2] = center[2] + half[2] * rng.uniform(0.0, 1.0)
    cfg.rEX = rotvec_mat(closing * rng.uniform(-np.pi / 4, np.pi / 4)) @ hand_from_axis(closing)
    cfg.width = float(rng.uniform(2 * half[axis], GRIP_MAX))
    cfg.grip = GRIP_HOLD


def perturb_config(cfg: GConfig, rng) -> GConfig:
    """A copy of an earlier probe's configuration, moved by PROBE['focus_sd'] (object units / radians)."""
    sd = PROBE["focus_sd"]
    sx, sy = kind_shape(cfg.pair[0]), kind_shape(cfg.pair[1])
    unit_xy, unit_ex = to_metric_shapes(np.ones(3), np.ones(3), sx, sy)
    c = cfg.copy()
    c.pXY = np.asarray(c.pXY, float) + rng.normal(0, sd, 3) * unit_xy
    c.pEX = np.asarray(c.pEX, float) + rng.normal(0, sd, 3) * unit_ex
    c.yawXY = float(c.yawXY + rng.normal(0, sd))
    c.rEX = rotvec_mat(rng.normal(0, sd, 3)) @ np.asarray(c.rEX, float)
    c.width = float(np.clip(c.width + rng.normal(0, sd) * 0.25 * GRIP_MAX, 0.0, GRIP_MAX))
    c.qx = float(np.clip(c.qx + rng.normal(0, sd), 0, 1))
    c.qy = float(np.clip(c.qy + rng.normal(0, sd), 0, 1))
    if rng.random() < 0.25:                                              # sometimes another gripper command
        c.grip = int(rng.choice([GRIP_OPEN, GRIP_CLOSE, GRIP_HOLD]))
    return c


def _joint_sample(rng) -> float:
    """Normalised joint value: at one end of the range (mechanical stop) or anywhere in between."""
    if rng.random() < PROBE["joint_limit_frac"]:
        return float(rng.integers(2))
    return float(rng.random())


def sample_ctx(rng, pair):
    """Where the anchor of the pair stands: a fixture's own region, else the table workspace."""
    for k in pair:
        if is_fixture(k):
            reg = FIXTURES[k].region
            return np.array([rng.uniform(*reg[0]), rng.uniform(*reg[1]), rng.uniform(*reg[2])])
    return TRAIN_REGION.sample(rng)


# ------------------------------------------------------------------ one probe
def _init(shapes, fixtures, pairs=None) -> None:
    use_body("float")
    _E.update(shapes=shapes, fixtures=fixtures, pairs=pairs)


def _move_hand(env, target, RE, seconds: float) -> bool:
    """Ramp the hand toward a pinch target, keeping its orientation RE (gripper command unchanged); False if
    the target is out of the robot's reach.  The floating hand's target moves in a straight line; the arm
    (robot body) is ramped in joint space toward its IK solution."""
    d = env.data
    target = np.array(target, float)
    target[2] = max(target[2], HAND_MIN_Z + 0.005)
    if env.floating:
        if not bool(reach_map().reachable(target, RE)):
            return False
        p0, _ = env.hand_pose()
        n = max(1, int(seconds / env.dt))
        for k in range(n):
            env.move_hand(p0 + (target - p0) * (k + 1) / n, RE)
            _sim(env, 1)
        return True
    q0 = env.arm_q()
    q1, ok = env.solve_ik(target, RE, q0)
    if not ok:
        return False
    n = max(1, int(seconds / env.dt))
    for k in range(n):
        d.ctrl[:7] = q0 + (q1 - q0) * (k + 1) / n
        _sim(env, 1)
    return True


def _sim(env, n: int):
    """Step the physics n times (rendering frames when a clip is being recorded)."""
    rec = _E.get("rec")
    if rec is None:
        mujoco.mj_step(env.model, env.data, nstep=max(1, n))
        return
    for _ in range(max(1, n)):
        mujoco.mj_step(env.model, env.data)
        rec.frame()


def _follows(dA, dE, segment: float) -> bool:
    """The body moved a good part of the hand's way, in the hand's direction."""
    nE, nA = np.linalg.norm(dE), np.linalg.norm(dA)
    return bool(nE > 0.5 * segment and nA > PROBE["follow_frac"] * nE
                and float(dA @ dE) > PROBE["follow_cos"] * nA * nE)


def _probe(rng, base: GConfig | None = None):
    """One probe: realise a random configuration (or a perturbed ``base``), then either a grasp probe (close
    the fingers, lift) or a motion probe (a few short hand motions in random directions, gripper command
    held).  Every lift or motion is an event: the configuration at its end and whether the body followed
    the hand.  Finally the hand releases and retracts and the scene settles (rest state for object-object
    and state relations)."""
    lift_probe = rng.random() < PROBE["grasp_frac"]
    for _try in range(50):
        cfg = perturb_config(base, rng) if base is not None and _try < 25 else \
            sample_config(rng, _E["shapes"], _E["fixtures"], _E.get("pairs"),
                          grasp_seed=lift_probe and rng.random() < PROBE["grasp_envelope_frac"])
        r = realize(cfg, sample_ctx(rng, cfg.pair), settle_s=PROBE["settle_s"], check_speed=False)
        if not isinstance(r, str):
            break
    else:
        return None
    env, _, _, s1 = r
    d = env.data
    size = kind_shape(cfg.pair[0]).length
    lift = np.array([0.0, 0.0, PROBE["lift"] * size])
    segment = PROBE["segment"] * size
    if _E.get("clip_path"):
        from ..world.clips import Recorder
        p = env.pair_view(env.sensors(), "A", env.ref)[VIEW_S.A_pos]
        _E["rec"] = Recorder(env, _E["clip_path"], lookat=(p[0], p[1], 0.05), distance=0.55, every=8)
        _E["rec"].hold(0.4)
    view = lambda: env.pair_view(env.sensors(), "A", env.ref)  # noqa: E731
    pA0 = view()[VIEW_S.A_pos].copy()
    RE = d.site_xmat[env.pinch_sid].reshape(3, 3).copy()      # the hand keeps its orientation while probing
    events = []
    if lift_probe:
        d.ctrl[env.gi] = 0.0                                         # close the fingers
        _sim(env, int(0.3 / env.dt))
        sc = view()
        pE, pA = sc[VIEW_S.ee_pos].copy(), sc[VIEW_S.A_pos].copy()
        if _move_hand(env, pE + lift, RE, 0.4):
            _sim(env, int(0.2 / env.dt))
            sl = view()
            followed = _follows(sl[VIEW_S.A_pos] - pA, sl[VIEW_S.ee_pos] - pE, segment)
            events.append((sc, followed, 0))
            if followed:
                # Store the demonstrated in-flight grip as well as the acquisition configuration.
                retained = sl[VIEW_S.lf_A] > 0 and sl[VIEW_S.rf_A] > 0
                events.append((sl, bool(retained), 2))
    else:
        for _ in range(PROBE["segments"]):
            sa = view()
            v = rng.normal(size=3)
            v[2] = max(v[2], -0.3 * np.linalg.norm(v[:2]))
            pE, pA = sa[VIEW_S.ee_pos].copy(), sa[VIEW_S.A_pos].copy()
            if not _move_hand(env, pE + segment * v / np.linalg.norm(v), RE, PROBE["segment_s"]):
                continue
            sb = view()
            events.append((sb, _follows(sb[VIEW_S.A_pos] - pA, sb[VIEW_S.ee_pos] - pE, segment), 1))
    # release, retract, settle
    d.ctrl[env.gi] = 255.0
    _sim(env, int(0.2 / env.dt))
    pE = view()[VIEW_S.ee_pos].copy()
    moved = _move_hand(env, pE + lift, RE, 0.4)
    _sim(env, int(0.3 / env.dt))
    s2 = view()
    # an interaction: the body followed the hand or was displaced (used to pick the clips)
    interacted = any(e[1] for e in events) or bool(np.linalg.norm(s2[VIEW_S.A_pos] - pA0) > 0.5 * size)
    if _E.get("rec") is not None:
        _E.pop("rec").save()
    return {"interaction": (events[0][2] if events else 1) if interacted else None, "sim_s": float(d.time),
            "s1": s1, "s2": s2, "moved": moved, "attached": any(e[1] for e in events), "events": events,
            "pair": [TYPES.index(cfg.pair[0]), TYPES.index(cfg.pair[1])], "cfg": cfg,
            "focus": bool(base is not None),
            # something happened: worth probing again nearby
            "eventful": bool(interacted or (moved and s2[VIEW_S.A_B] > 0))}


def _chunk(task):
    """Probes ``ids`` (each with its own seed); ``focus``: earlier configurations to restart from."""
    seed, ids, focus = task
    out = []
    for i in ids:
        rng = np.random.default_rng(task_seed(seed, i))
        base = focus[int(rng.integers(len(focus)))] if focus and rng.random() < PROBE["focus_frac"] else None
        r = _probe(rng, base)
        if r is not None:
            r["id"] = int(i)
            out.append(r)
    return out


# ------------------------------------------------------------------ main loop
def worthy_keys(rels) -> list[tuple]:
    return relation_keys([r for r in rels if worthy(r)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", type=int, default=30000, help="maximum number of probes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-members", type=int, default=PROBE["min_members"])
    ap.add_argument("--shapes", default="cube", help="comma-separated free body kinds in the probe scene")
    ap.add_argument("--fixtures", default=None, help="fixtures in the scene (default config.FIXTURE_KINDS)")
    ap.add_argument("--pairs", default=None, help="X/Y kind pairs of the scene, e.g. cube/cube")
    ap.add_argument("--no-focus", action="store_true", help="uniform probing only")
    ap.add_argument("--out", default="runs/pair_cubes/relations")
    ap.add_argument("--rediscover", action="store_true", help="re-run discovery on the saved probes.npz")
    ap.add_argument("--clips", type=int, default=0, help="rendered example probes (runs/<scene>/clips)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.rediscover:
        from .discover import load_records
        records = load_records(np.load(out.parent / "probes.npz"))
        rels, report = discover(records, VIEW_S, min_members=args.min_members)
        rels.save(out)
        (out.parent / "probe_report.json").write_text(json.dumps(report, indent=1))
        return
    shapes = [] if args.shapes == "-" else args.shapes.split(",")
    fixtures = args.fixtures.split(",") if args.fixtures else []
    pairs = [tuple(p.split("/")) for p in args.pairs.split(",")] if args.pairs else None
    graspable = [k for k in shapes if kind_shape(k).grasp_width < GRIP_MAX]
    rng = np.random.default_rng(args.seed)
    B, per = PROBE["batch"], 16
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(shapes, fixtures, pairs)) as pool:
        records, prev, calm, done = [], None, 0, 0
        while done < args.probes:
            n = min(B, args.probes - done)
            pool_cfgs = [r["cfg"] for r in records if r["eventful"]][-PROBE["focus_pool"]:] \
                if records and not args.no_focus else []
            tasks = []
            for a in range(done, done + n, per):
                f = [pool_cfgs[int(k)] for k in rng.integers(len(pool_cfgs), size=64)] if pool_cfgs else None
                tasks.append((args.seed, list(range(a, min(a + per, done + n))), f))
            for chunk in pool.map(_chunk, tasks):
                records += chunk
            done += n
            if done >= PROBE["min_probes"]:
                discovered, _ = discover(records, VIEW_S, min_members=args.min_members)
                keys = worthy_keys(discovered)
                lifts = Counter()
                for r in discovered.hc:
                    if r.carries:
                        lifts.update(r.stats["lift_subjects"])
                enough_lifts = all(lifts[k] >= PRIMS["min_members"] for k in graspable)
                calm = calm + 1 if enough_lifts and prev is not None and same_relations(keys, prev) else 0
                prev = keys
                progress(done, args.probes, f"{len(keys)} primitive-worthy relations, unchanged for {calm} batch(es)")
                if calm >= PROBE["stable_checks"]:
                    break                                      # the discovered relations have converged
            else:
                progress(done, args.probes, f"{sum(r['eventful'] for r in records)} eventful probes")
    ev = [(i, e) for i, r in enumerate(records) for e in r["events"]]
    np.savez_compressed(out.parent / "probes.npz", s1=np.array([r["s1"] for r in records]),
                        s2=np.array([r["s2"] for r in records]),
                        attached=np.array([r["attached"] for r in records]),
                        moved=np.array([r["moved"] for r in records]),
                        pair=np.array([r["pair"] for r in records]), types=np.array(TYPES),
                        focus=np.array([r["focus"] for r in records]),
                        ev_s=np.array([e[0] for _, e in ev]), ev_follow=np.array([e[1] for _, e in ev], bool),
                        ev_kind=np.array([e[2] for _, e in ev]), ev_probe=np.array([i for i, _ in ev]))
    rels, report = discover(records, VIEW_S, min_members=args.min_members)
    rels.save(out)
    report["wall_s"] = round(time.time() - t0, 1)
    report["sim_s"] = round(float(sum(r.get("sim_s", 0.0) for r in records)), 1)   # simulated seconds, all probes
    report["focused"] = int(sum(r["focus"] for r in records))
    report["eventful"] = int(sum(r["eventful"] for r in records))
    report["worthy"] = [r.name for r in rels if worthy(r)]
    if args.clips:
        record_clips(out.parent / "clips", args.clips, shapes, fixtures, args.seed, pairs)
    (out.parent / "probe_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "relations"}))
    for r in report["relations"]:
        print(json.dumps(r))


# ------------------------------------------------------------------ clips (rendered after learning)
def record_clips(path: Path, n: int, shapes, fixtures, seed: int = 0, pairs=None) -> None:
    """n rendered probes, only ones with an interaction (body carried or displaced), grasp and motion
    probes balanced: probes are screened without rendering, then the chosen one is replayed (same seed,
    deterministic physics) with the renderer."""
    _init(shapes, fixtures, pairs)
    kept, per_kind = 0, {0: 0, 1: 0}
    for i in range(200 * n):
        if kept >= n:
            break
        r = _probe(np.random.default_rng(task_seed(seed + 999, i)))
        k = None if r is None else r["interaction"]
        if k is None or per_kind[k] >= (n + 1) // 2:
            continue
        _E["clip_path"] = Path(path) / f"probe_{kept}.mp4"
        _probe(np.random.default_rng(task_seed(seed + 999, i)))
        _E.pop("clip_path")
        per_kind[k] += 1
        kept += 1


def record_relation_clips(R, path: Path, shapes, fixtures, pairs=None, seed: int = 0, tries: int = 4000) -> dict:
    """One rendered probe per discovered relation, a probe that brought that relation about.  Returns
    relation -> clip."""
    _init(shapes, fixtures, pairs)
    want, done = {r.name for r in R}, {}
    for i in range(tries):
        if not want - set(done):
            break
        rec = _probe(np.random.default_rng(task_seed(seed + 4242, i)))
        if rec is None:
            continue
        new = sorted(created_relations(R, rec) & (want - set(done)))
        if not new:
            continue
        _E["clip_path"] = Path(path) / f"{new[0]}.mp4"
        _probe(np.random.default_rng(task_seed(seed + 4242, i)))
        _E.pop("clip_path")
        done[new[0]] = Path(path) / f"{new[0]}.mp4"
    return done


if __name__ == "__main__":
    main()
