"""Hand probes: the hand acts on two bodies; every motion is recorded with its start, end and rest states.

    python -m bep.babble.hand_probe --grasp 2000 --interact 1000 --out runs/pair_cubes_hand

Only the hand acts (moved, or its fingers opened / closed); the bodies move through physics.  The hand is
pointed down with a random yaw, or (HPROBE['side_frac']) sideways: horizontal approach in a random direction,
the fingers closing horizontally or vertically.  A probe starts from a random relational configuration of the
two bodies (bep.babble.probe.sample_config), settled, with the hand outside a random point of X, its approach
axis (the fingers' direction) pointing at it.

* grasp probe: approach until the touching body pairs change, close until the opening stops changing, lift
  PROBE['lift'] body lengths; from the lifted state (saved), ``place_branches`` placements: move X to a random
  spot within interaction distance of Y, descend until the touching pairs change, open, retract, settle.
  Without X in hand: open, retract, settle.
* interaction probe: approach until the touching pairs change; then ``depth`` steps, each from a state of the
  probe's archive (keyed by its touching body pairs, rare keys first): ``branches`` motions, each a random
  translation with a random turn about the vertical, or a random finger command, until HPROBE['after_len']
  body lengths (at the hand's speed) after the touching pairs change, or its time is up (the state at that change is recorded: the interaction
  starts there); after each, a settle branch gives the rest state.

Contacts are read per body (each body of the hand's subtree is its own role "E.<body>"; A, B the free bodies;
S anything static).  Nothing here names a relation: bep.babble.motion_clusters clusters the motions.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from ..config import CONTROL_DT, GRIP_MAX, HPROBE, PROBE, SUBSTEPS, progress, task_seed
from ..world.env import ee_down_rot
from ..world.geometry import rotvec_mat
from ..world.hog import GRIP_HOLD, GRIP_OPEN, realize
from ..world.objects import kind_shape, use_body
from ..world.reach import reach_map
from .probe import sample_config, sample_ctx

_W: dict = {}


# ------------------------------------------------------------------ bodies and contacts
def roles(env) -> dict:
    """Body id -> role: "E.<name>" for every body of the hand's subtree, A / B the free bodies, S the rest."""
    if "_hp_roles" not in env.__dict__:
        m = env.model
        bid = lambda n: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n))   # noqa: E731
        hand = bid("hand")
        out = {}
        for b in range(m.nbody):
            p = b
            while p > 0 and p != hand:
                p = int(m.body_parentid[p])
            out[b] = f"E.{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}" if p == hand else "S"
        for n in env.free:
            out[bid(f"cube{n}")] = n
        env._hp_roles, env._hp_bid = out, {n: bid(f"cube{n}") for n in env.free}
    return env._hp_roles


def touching(env) -> str:
    """The touching body pairs, as a sorted key "r1/r2|..." (roles sorted within each pair)."""
    m, d, r = env.model, env.data, roles(env)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = r[int(m.geom_bodyid[c.geom1])], r[int(m.geom_bodyid[c.geom2])]
        if a != b:
            pairs.add("/".join(sorted((a, b))))
    return "|".join(sorted(pairs))


def save(env) -> tuple:
    return env.get_state(), env.data.ctrl.copy()


def restore(env, snap) -> None:
    env.set_state(snap[0], snap[1])
    mujoco.mj_forward(env.model, env.data)


# ------------------------------------------------------------------ hand orientations and motions
def hand_rotation(rng) -> np.ndarray:
    """Pointed down with a random yaw, or (side_frac) sideways: approach (column 2) horizontal in a random
    direction, closing axis (column 1) horizontal or vertical."""
    yaw = rng.uniform(-np.pi, np.pi)
    if rng.random() >= HPROBE["side_frac"]:
        return ee_down_rot(yaw)
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.array([-np.sin(yaw), np.cos(yaw), 0.0]) if rng.random() < 0.5 else np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(y, z), y, z], axis=1)


def move(env, v, yaw_rate: float, seconds: float, stop_on_contact: bool = True, after_s: float = 0.0) -> str:
    """Translate the hand at v (m/s) while turning it about the vertical; stops ``after_s`` after the touching
    pairs change (debounced), when the time is up, or when the hand would leave the robot's reach."""
    p, R = env.hand_pose()
    key0, changed = touching(env), 0
    n = max(1, int(round(seconds / CONTROL_DT)))
    k = 0
    while k < n:
        k += 1
        p = p + np.asarray(v, float) * CONTROL_DT
        R = rotvec_mat(np.array([0.0, 0.0, yaw_rate * CONTROL_DT])) @ R
        if not bool(reach_map().reachable(p, R)):
            return "unreachable"
        env.move_hand(p, R)
        _step(env, SUBSTEPS)
        if stop_on_contact and key0 is not None:
            changed = changed + 1 if touching(env) != key0 else 0
            if changed >= HPROBE["debounce"]:
                if after_s <= 0:
                    return "contact"
                env._hp_contact = save(env)                         # where the interaction starts
                key0, n = None, k + int(round(after_s / CONTROL_DT))   # go on after_s past the change
    return "contact" if key0 is None else "timeout"


def actuate(env, ctrl: float, seconds: float, *, until_still: bool = False, stop_on_contact: bool = False,
            after_s: float = 0.0) -> str:
    """Command the fingers; stop once the opening stops changing (until_still), or ``after_s`` after the
    touching pairs change."""
    env.data.ctrl[env.gi] = ctrl
    key0, changed, w0, calm = touching(env), 0, env.gripper_width(), 0
    n = max(1, int(round(seconds / CONTROL_DT)))
    k = 0
    while k < n:
        k += 1
        _step(env, SUBSTEPS)
        w = env.gripper_width()
        calm, w0 = (calm + 1 if abs(w - w0) < 1e-4 else 0), w
        if until_still and calm * CONTROL_DT >= HPROBE["still_s"]:
            return "still"
        if stop_on_contact and key0 is not None:
            changed = changed + 1 if touching(env) != key0 else 0
            if changed >= HPROBE["debounce"]:
                if after_s <= 0:
                    return "contact"
                env._hp_contact = save(env)
                key0, n = None, k + int(round(after_s / CONTROL_DT))
    return "contact" if key0 is None else "timeout"


def _step(env, n: int) -> None:
    """Step the physics n times (rendering frames when a clip is being recorded, see record_clips)."""
    rec = _W.get("rec")
    if rec is None:
        mujoco.mj_step(env.model, env.data, nstep=max(1, n))
        return
    for _ in range(max(1, n)):
        mujoco.mj_step(env.model, env.data)
        rec.frame()


def settle(env) -> None:
    """The settle branch (not filmed: the probe goes back to where the motion ended)."""
    mujoco.mj_step(env.model, env.data, nstep=max(1, int(HPROBE["settle_s"] / env.dt)))


class Log:
    """The probe's motions: kind, why it stopped, start / end states, touching pairs, and a rest state."""

    def __init__(self, env, probe: str, index: int):
        self.env, self.probe, self.index, self.motions = env, probe, index, []

    def begin(self):
        self._s, self._k = save(self.env), touching(self.env)
        self.env._hp_contact = None

    def end(self, kind: str, reason: str, rest: bool = True) -> None:
        env = self.env
        sc = env._hp_contact or self._s                     # the state at the first contact change (else the start)
        m = dict(probe=self.probe, index=self.index, kind=kind, reason=reason, s0=self._s[0], c0=self._s[1],
                 key0=self._k, sc=sc[0], cc=sc[1], s1=env.get_state(), c1=env.data.ctrl.copy(), key1=touching(env))
        if rest:                                     # settle branch: the rest state after the motion
            here = save(env)
            settle(env)
            m.update(s_rest=env.get_state(), key_rest=touching(env))
            restore(env, here)
        else:
            m.update(s_rest=m["s1"], key_rest=m["key1"])
        self.motions.append(m)


# ------------------------------------------------------------------ probes
def _init(shapes, pairs) -> None:
    use_body("float")
    _W.update(shapes=shapes, pairs=pairs)


def _start(rng, opening, grip):
    """A settled configuration with the hand outside a random point of X, pointing at it."""
    for _ in range(100):
        cfg = sample_config(rng, _W["shapes"], [], _W["pairs"])
        sx = kind_shape(cfg.pair[0])
        R = hand_rotation(rng)                                      # in X's yaw frame (yaw is random anyway)
        target = rng.uniform(-0.5, 0.5, 3) * np.asarray(sx.half, float)
        reach = float(np.linalg.norm(sx.half)) + HPROBE["standoff"] * sx.length
        cfg.pEX, cfg.rEX, cfg.width, cfg.grip = target - R[:, 2] * reach, R, opening(rng), grip
        r = realize(cfg, sample_ctx(rng, cfg.pair), settle_s=PROBE["settle_s"], check_speed=False)
        if not isinstance(r, str):
            env = r[0]
            roles(env)
            return env, sx, reach
    return None


def _approach(env, log, sx, reach) -> str:
    log.begin()
    axis = env.hand_pose()[1][:, 2]
    why = move(env, HPROBE["speed"] * axis, 0.0, (reach + HPROBE["overshoot"] * sx.length) / HPROBE["speed"])
    log.end("approach", why)
    return why


def _release(env, log, lift) -> None:
    log.begin()
    actuate(env, 255.0, HPROBE["open_s"])
    log.end("open", "timeout")
    log.begin()
    move(env, lift / HPROBE["open_s"], 0.0, HPROBE["open_s"], stop_on_contact=False)
    log.end("retract", "timeout")


def grasp_probe(rng, index: int) -> list[dict]:
    got = _start(rng, lambda r: GRIP_MAX, GRIP_OPEN)
    if got is None:
        return []
    env, sx, reach = got
    log, L = Log(env, "grasp", index), sx.length
    lift = np.array([0.0, 0.0, PROBE["lift"] * L])
    _approach(env, log, sx, reach)
    log.begin()
    actuate(env, 0.0, HPROBE["close_s"], until_still=True)
    log.end("close", "still")
    log.begin()
    why = move(env, lift / 0.4, 0.0, 0.4, stop_on_contact=False)
    _step(env, max(1, int(HPROBE["hold_s"] / env.dt)))
    log.end("lift", why)
    d, bA, bB = env.data, env._hp_bid["A"], env._hp_bid["B"]
    held = ("A/S" not in touching(env).split("|") and any(p.startswith("A/E.") for p in touching(env).split("|")))
    lifted = save(env)
    _release(env, log, lift)
    if not held:
        return log.motions
    sy = env.shape("B")
    unit = np.array([sy.half[0], sy.half[1]]) + max(sx.half[0], sx.half[1])
    for _ in range(HPROBE["place_branches"]):
        restore(env, lifted)
        spot = d.xpos[bB][:2] + rng.uniform(-PROBE["near_xy"], PROBE["near_xy"], 2) * unit
        gap = np.r_[spot - d.xpos[bA][:2], 0.0]
        log.begin()
        why = move(env, HPROBE["speed"] * gap / max(np.linalg.norm(gap), 1e-9), 0.0,
                   np.linalg.norm(gap) / HPROBE["speed"], stop_on_contact=False)
        log.end("transfer", why)
        if why == "unreachable":
            continue
        log.begin()
        why = move(env, np.array([0.0, 0.0, -HPROBE["speed"]]), 0.0, HPROBE["place_drop"] * L / HPROBE["speed"])
        log.end("descend", why)
        _release(env, log, lift)
    return log.motions


def interaction_probe(rng, index: int) -> list[dict]:
    got = _start(rng, lambda r: float(r.uniform(0.0, GRIP_MAX)), GRIP_HOLD)
    if got is None:
        return []
    env, sx, reach = got
    log = Log(env, "interact", index)
    _approach(env, log, sx, reach)
    after = HPROBE["after_len"] * sx.length / HPROBE["speed"]
    archive, seen = [(touching(env), save(env))], Counter()
    for _ in range(HPROBE["depth"]):
        w = np.array([1.0 / np.sqrt(1.0 + seen[k]) for k, _ in archive])
        _, node = archive[int(rng.choice(len(archive), p=w / w.sum()))]
        for _b in range(HPROBE["branches"]):
            restore(env, node)
            log.begin()
            if rng.random() < 0.5:
                u = rng.normal(size=3)
                why = move(env, HPROBE["speed"] * u / np.linalg.norm(u), rng.normal(0.0, HPROBE["yaw_rate"]),
                           HPROBE["move_s"], after_s=after)
                log.end("move", why)
            else:
                why = actuate(env, float(rng.choice([0.0, 255.0])), HPROBE["grip_s"], stop_on_contact=True,
                              after_s=after)
                log.end("grip", why)
            key = touching(env)
            seen[key] += 1
            archive.append((key, save(env)))
    return log.motions


def _chunk(task):
    seed, which, ids = task
    fn = grasp_probe if which == "grasp" else interaction_probe
    out = []
    for i in ids:
        out += fn(np.random.default_rng(task_seed(seed + (0 if which == "grasp" else 7919), i)), i)
    return out


def record_clips(path: Path, n: int, seed: int = 0) -> list[Path]:
    """n rendered probes of each kind, each one in which the hand had an effect of its probe's kind (a grasp
    probe that carried the body, an interaction probe in which the body followed the hand on its support, by
    bep.babble.motion_clusters.effect_key): probes are screened without rendering, then the chosen one is
    replayed (same seed, deterministic physics) with the renderer."""
    from ..world.clips import Recorder
    from .motion_clusters import effect_key, features
    _init(["cube"], [("cube", "cube")])
    wanted = {"grasp": "follows, no support", "interact": "follows, support kept"}
    out = []
    for which, fn in (("grasp", grasp_probe), ("interact", interaction_probe)):
        kept = 0
        for i in range(400):
            if kept >= n:
                break
            seed_i = task_seed(seed + 31337 + (0 if which == "grasp" else 7919), i)
            motions = fn(np.random.default_rng(seed_i), i)
            if not motions:
                continue
            from ..world.objects import pair_world
            env = pair_world("cube", "cube")
            roles(env)
            if not any(effect_key(r) == wanted[which] for m in motions for r in features(env, m)):
                continue
            restore(env, (motions[0]["s0"], motions[0]["c0"]))
            p = env.data.xpos[env._hp_bid["A"]]
            clip = Path(path) / f"probe_{which}_{kept}.mp4"
            _W["rec"] = Recorder(env, clip, lookat=(float(p[0]), float(p[1]), 0.05), distance=0.6, every=4, max_s=20.0)
            try:
                fn(np.random.default_rng(seed_i), i)
                _W["rec"].save()
            finally:
                _W.pop("rec", None)
            out.append(clip)
            kept += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grasp", type=int, default=2000, help="grasp probes")
    ap.add_argument("--interact", type=int, default=1000, help="interaction probes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pair_cubes_hand")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, per = time.time(), 8
    tasks = [(args.seed, w, list(range(a, min(a + per, n))))
             for w, n in (("grasp", args.grasp), ("interact", args.interact)) for a in range(0, n, per)]
    motions: list[dict] = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(["cube"], [("cube", "cube")])) as pool:
        for k, chunk in enumerate(pool.imap(_chunk, tasks)):
            motions += chunk
            progress(k + 1, len(tasks), f"{len(motions)} motions")
    np.savez_compressed(out / "motions.npz", **{k: np.array([m[k] for m in motions]) for k in motions[0]})
    rep = {"wall_s": round(time.time() - t0, 1), "motions": len(motions),
           "by_probe_kind_reason": {f"{p}:{k}": dict(Counter(m["reason"] for m in motions if m["probe"] == p and m["kind"] == k))
                                    for p, k in sorted({(m["probe"], m["kind"]) for m in motions})}}
    (out / "hand_probe.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
