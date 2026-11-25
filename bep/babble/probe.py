"""Babble: unscripted motor babbling with the floating hand (no grasp or placement script).


A probe starts from a random relational configuration of two bodies and the hand (``sample_config``: X near
or apart from Y, the hand near X, in the bodies' own units; the hand in the robot's working posture, pointing down, most of the time: a body schema), or from an archived state where a rare event happened (a novelty archive: rarely seen
event keys are revisited, as Go-Explore returns to rare cells).  The hand then issues random motor
commands, each either

* a straight move of random direction and length (the hand keeps its orientation, with a small random turn
  about the vertical), or
* a finger command (close, or open),

and after each command the scene is left to settle with the command held, in a branch that is not followed
(``settled``): an event is the change between two settled states (bep.babble.events).  At the end the fingers
open, the hand withdraws and the scene rests (``rest``): what holds after letting go.

Every command records the sensors at its start, at its end and after settling, and its end state (so the
archive can restart from it).  Nothing here names a relation.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..config import CONTROL_DT, GRIP_MAX, PROBE, SUBSTEPS, BABBLE, task_seed
from ..world.env import ee_down_rot
from ..world.geometry import random_rotation, rotvec_mat
from ..world.hog import GRIP_CLOSE, GRIP_HOLD, GRIP_OPEN, HAND_MIN_Z, TRAIN_REGION, GConfig, realize
from ..world.objects import FIXTURES, is_fixture, kind_shape, pair_world
from ..world.reach import reach_map
from ..world.relations import to_metric_shapes


# ------------------------------------------------------------------ configurations
def sample_config(rng, shapes, fixtures, pairs=None) -> GConfig:
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
    return cfg


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



# ------------------------------------------------------------------ motor babbling
OPEN, CLOSE = 255.0, 0.0


def _steps(seconds: float) -> int:
    return max(1, int(round(seconds / CONTROL_DT)))


def _settled(env) -> np.ndarray:
    """Sensors after letting the scene settle with the current commands held (a branch: the state is
    restored afterwards)."""
    d = env.data
    state, ctrl = env.get_state(), d.ctrl.copy()
    mocap = (d.mocap_pos.copy(), d.mocap_quat.copy())
    mujoco.mj_step(env.model, d, nstep=_steps(BABBLE["settle_s"]) * SUBSTEPS)
    s = env.sensors().copy()
    env.set_state(state, ctrl)
    d.mocap_pos[:], d.mocap_quat[:] = mocap
    mujoco.mj_forward(env.model, d)
    return s


def _move(env, delta, turn: float, seconds: float) -> bool:
    """Ramp the hand's target along ``delta`` (m) while turning it by ``turn`` (rad) about the vertical; False
    when the end pose is out of the robot's reach (nothing happens then)."""
    p0, R0 = env.hand_pose()
    p1 = p0 + np.asarray(delta, float)
    p1[2] = max(p1[2], HAND_MIN_Z + 0.005)
    R1 = rotvec_mat(np.array([0.0, 0.0, turn])) @ R0
    if not bool(reach_map().reachable(p1, R1)):
        return False
    n = _steps(seconds)
    for k in range(n):
        a = (k + 1) / n
        env.move_hand(p0 + a * (p1 - p0), rotvec_mat(np.array([0.0, 0.0, a * turn])) @ R0)
        mujoco.mj_step(env.model, env.data, nstep=SUBSTEPS)
    return True


def _grip(env, ctrl: float, seconds: float) -> None:
    env.data.ctrl[env.gi] = ctrl
    mujoco.mj_step(env.model, env.data, nstep=_steps(seconds) * SUBSTEPS)


def _random_command(rng, env, L: float):
    """(kind, function) of a random motor command."""
    if rng.random() < BABBLE["grip_p"]:
        ctrl = CLOSE if env.data.ctrl[env.gi] > 127 else OPEN        # the other finger command
        return ("close" if ctrl == CLOSE else "open"), lambda: (_grip(env, ctrl, BABBLE["grip_s"]), True)[1]
    v = rng.normal(size=3)
    v[2] = max(v[2], -BABBLE["move_up_bias"] * np.linalg.norm(v[:2]))
    v *= rng.uniform(*BABBLE["move_len"]) * L / max(np.linalg.norm(v), 1e-9)
    turn = float(rng.normal(0.0, 0.3))
    return "move", lambda: _move(env, v, turn, BABBLE["move_s"])


def _start(rng, scene: dict, base):
    """A settled start: a random configuration (or a perturbed copy of ``base``'s configuration), or an
    archived physical state restored as it was (``base['state']``)."""
    if base is not None and base.get("state") is not None:
        env = pair_world(*base["pair"])
        env.set_state(base["state"], base["ctrl"])
        env.ref = "B"
        mujoco.mj_forward(env.model, env.data)
        return env, tuple(base["pair"])
    for _ in range(50):
        cfg = sample_config(rng, scene["shapes"], scene["fixtures"], scene["pairs"])
        if base is not None and base.get("cfg") is not None:
            cfg = perturb_config(base["cfg"], rng)
        elif rng.random() < BABBLE["down_p"]:            # the robot's working posture, any yaw
            cfg.rEX = ee_down_rot(float(rng.uniform(-np.pi, np.pi)))
        r = realize(cfg, sample_ctx(rng, cfg.pair), settle_s=PROBE["settle_s"], check_speed=False)
        if not isinstance(r, str):
            return r[0], tuple(cfg.pair)
    return None, None


def probe(task) -> list[dict]:
    """Probes ``task['ids']`` (each with its own seed); ``task['bases']``: archived states to restart from."""
    from ..workers import CACHE
    scene = CACHE["scene"]
    out = []
    for i, base in zip(task["ids"], task["bases"]):
        rng = np.random.default_rng(task_seed(task["seed"], i))
        env, pair = _start(rng, scene, base)
        if env is None:
            continue
        L = kind_shape(pair[0]).length
        t0 = float(env.data.time)
        rec = {"id": int(i), "pair": pair, "restart": base is not None, "s_init": _settled(env), "commands": []}
        plan = [None] * BABBLE["commands"] + ["open", "retract"]
        for which in plan:
            s_start = env.sensors().copy()
            if which is None:
                kind, fn = _random_command(rng, env, L)
            elif which == "open":
                kind, fn = "open", lambda: (_grip(env, OPEN, BABBLE["release_s"]), True)[1]
            else:
                kind, fn = "retract", lambda: _move(env, np.array([0.0, 0.0, BABBLE["retract"] * L]), 0.0,
                                                   BABBLE["move_s"])
            if not fn():
                continue
            rec["commands"].append({"kind": kind, "reset": which is not None, "s_start": s_start,
                                    "s_end": env.sensors().copy(), "s_set": _settled(env),
                                    "state": env.get_state().copy(), "ctrl": env.data.ctrl.copy()})
        mujoco.mj_step(env.model, env.data, nstep=_steps(BABBLE["rest_s"]) * SUBSTEPS)
        rec["s_rest"] = env.sensors().copy()
        rec["sim_s"] = float(env.data.time) - t0
        out.append(rec)
    return out
