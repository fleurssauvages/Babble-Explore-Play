"""A primitive toward one discovered relation (input to reverse Go-Explore).

Two target kinds, one machinery:

* object-object relation r(X, Y): roles hand E, moved body X (= A), reference body Y (= B);
  X is a free body, Y a free body or a moving part.
* object-state relation s(X): X (= A) is a moving part; its reference is a fixed virtual spot at
  the part's base (so "X relative to the reference" encodes how far the part has moved), and a
  free body of the scene is parked out of reach.

State z = the 14-D relational vector in the bodies' own units (p_X^Y, yaw_XY, pinch point p_E^X,
closing axis, opening, joint X, joint Y); mode = bitmask of the discovered relations between the roles.
The hand's rotation about its closing axis is not in z (it does not change where the fingers touch):
configurations carry the full 6-DoF hand pose, the state only what fixes the contacts.
Goal: the target relation holds and no other relation does (in particular the hand has let go).

Backward operators, each perturbing one relational quantity and letting the Hand of God settle
decide the resulting mode:

    XY     move X relative to Y   (the hand keeps its pose relative to X; free X only)
    EX     move the hand relative to X (and turn it a little)
    rot    turn the hand about its pinch point (any orientation)
    grip   new finger opening and gripper command (open / close / hold)
    rel    switch a discovered relation on, using a probe state where it held
    joint  move a moving part's joint (X's, or Y's when Y is a part)
    swap   same relational configuration, another pair of free shapes

The roles' kinds (which shapes / fixtures) are those the target relation was observed with in
the probe.  A context ``ctx`` = (x, y, yaw of the anchor, kind of X, kind of Y) fixes where and
with which bodies a configuration is realised.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from ..config import CONTROL_DT, EXEC, EXPLORE, GRIP_MAX, PROBE, SHAPES, L, z_scale
from ..world.env import HOME_Q
from ..world.env import ee_down_rot, yaw_rot
from ..world.geometry import hand_from_axis, mat_rotvec, rotvec_mat
from ..world.hog import GRIP_CLOSE, GRIP_HOLD, GRIP_OPEN, TRAIN_REGION, GConfig, precheck, realize, teleport_ik
from ..world.objects import FIXTURES, TYPES, VIEW_S, Spot, is_fixture, kind_shape, pair_world, swap_view
from ..world.relations import (CC, CLOSING, HC, SC, Z_AXIS, Z_OPEN, Z_PEX, Z_QX, Z_QY, RelationSet, axis_code,
                               cc_features, hc_features, joint_norm, relational_z, sym_code, to_metric_shapes)

W = 1.0 / z_scale()                              # fixed metric weights of z (object units, angle codes rescaled)


def to_metric(pn, pen, pair):
    """Object-unit relative positions -> metres for a pair of kinds (inverse of the features)."""
    return to_metric_shapes(pn, pen, kind_shape(pair[0]), kind_shape(pair[1]))


def decode_angle(code, period):
    if period <= 0:
        return 0.0
    return float(np.arctan2(code[1], code[0]) * period / (2 * np.pi))


def ctx_pair(ctx) -> tuple[str, str]:
    return TYPES[int(ctx[3])], TYPES[int(ctx[4])]


def base_spot(env, n) -> Spot:
    """Virtual reference at the base of fixture n (its pose in the world)."""
    bid = env.fix_bid[n]
    p, q = env.model.body_pos[bid], env.model.body_quat[bid]
    return Spot(float(p[0]), float(p[1]), float(2 * np.arctan2(q[3], q[0])))


class DiscoveredPrimitive:
    def __init__(self, relations: RelationSet, target: str, *, kinds: tuple | None = None,
                 probes: str | None = "runs/pair_cubes/probes.npz", steps: dict | None = None) -> None:
        self.R = relations
        self.target = target
        r = relations.get(target)
        if r.kind not in (CC, SC, HC):
            raise ValueError(f"unsupported relation kind {r.kind}")
        self.kind = r.kind
        self.rel_target = r
        self.goal_mode = relations.bit(target)
        self.name = f"prim_{target}"
        self.eps = _ConstDict(EXPLORE["eps"])
        self.steps = dict(steps or EXPLORE["step0"])
        self.probes = dict(np.load(probes)) if probes and Path(probes).exists() else None
        self.kinds = tuple(kinds) if kinds else self._dominant_kinds()
        # coordinates of z the target relation constrains, and their typical value
        self.goal_slice = {SC: np.array([Z_QX]), HC: np.array([5, 6, 7, 8, 9, 10, Z_OPEN]),
                           CC: np.array([0, 1, 2])}[self.kind]
        self.goal_center = np.asarray(r.center, float)
        self.goal_members = np.asarray(r.members, float).reshape(len(r.members), -1)
        self.chain: list = []                     # (kind, kinds, goal states) of primitives explored before
        self.metric: dict = {}                    # mode -> learned distance weights (default: W)
        self.pools = self._member_pools() if self.probes is not None else {}

    # ------------------------------------------------------------ data from the probe
    def oriented(self, key: str = "s2"):
        """Probe rest states in both role orders: (views, pairs of kinds) with the swapped copies appended."""
        s, pairs = self.probes[key], [(TYPES[int(a)], TYPES[int(b)]) for a, b in self.probes["pair"]]
        if key != "s2":
            return s, pairs
        return np.concatenate([s, swap_view(s)]), pairs + [(b, a) for a, b in pairs]

    def _dominant_kinds(self) -> tuple:
        """The pair of body kinds the target relation was observed with most often in the probe (either
        role order); object-object primitives need a free X."""
        if self.probes is None:
            return (SHAPES[0], SHAPES[0])
        if self.kind == HC:                       # hand-object: the body kinds of the events where it held
            ev = self.probes["ev_s"]
            holds = self.R.holds_all(ev, VIEW_S)[self.target]
            kinds = [TYPES[int(self.probes["pair"][int(i)][0])] for i in self.probes["ev_probe"][holds]]
            if not kinds:
                raise ValueError(f"relation {self.target} never held in the probe")
            return (Counter(kinds).most_common(1)[0][0], SHAPES[0])
        s2, pairs = self.oriented()
        holds = self.R.holds_all(s2, VIEW_S)[self.target]
        if self.kind == SC:
            holds = holds & (s2[:, VIEW_S.A_art] > 0)
        c = Counter(p for p, h in zip(pairs, holds) if h and (self.kind == SC or not is_fixture(p[0])))
        if not c:
            raise ValueError(f"relation {self.target} never held with a free subject in the probe")
        tx, ty = c.most_common(1)[0][0]
        return (tx, SHAPES[0]) if self.kind == SC else (tx, ty)

    def observed_roles(self, share: float = 0.05) -> list:
        """Every (X kind, Y kind) the target relation held with in the probe, for at least ``share`` of its
        occurrences (hand-object and state relations: Y is irrelevant, recorded as "")."""
        if self.probes is None:
            return [tuple(self.kinds)]
        if self.kind == HC:
            ev = self.probes["ev_s"]
            holds = self.R.holds_all(ev, VIEW_S)[self.target]
            c = Counter((TYPES[int(self.probes["pair"][int(i)][0])], "") for i in self.probes["ev_probe"][holds])
        else:
            s2, pairs = self.oriented()
            holds = self.R.holds_all(s2, VIEW_S)[self.target]
            if self.kind == SC:
                holds = holds & (s2[:, VIEW_S.A_art] > 0)
            c = Counter((p[0], "") if self.kind == SC else p for p, h in zip(pairs, holds) if h)
        n = sum(c.values())
        return sorted(k for k, v in c.items() if v >= share * n) if n else [tuple(self.kinds)]

    def _member_pools(self) -> dict:
        """Probe states in which each relation held, in object units."""
        pools = {}
        for r in self.R:
            if r.kind == CC:
                s = self.oriented()[0]
                sel = self.R.holds_all(s, VIEW_S)[r.name]
                c = cc_features(s[sel], VIEW_S, "AB")
                pools[r.name] = (r.kind, c["p"], c["yaw"])
            elif r.kind == HC:
                s = self.probes["ev_s"]
                sel = self.R.holds_all(s, VIEW_S)[r.name]
                h = hc_features(s[sel], VIEW_S)
                pools[r.name] = (r.kind, h["p"], h["R_m"], h["width"])
        return {k: v for k, v in pools.items() if len(v[1])}

    # ------------------------------------------------------------ roles and state
    @property
    def x_is_part(self) -> bool:
        return is_fixture(self.kinds[0])

    @property
    def y_is_part(self) -> bool:
        return is_fixture(self.kinds[1]) and self.kind == CC

    @property
    def virtual_y(self) -> bool:
        """State and hand-object primitives have no second body: Y is a virtual reference."""
        return self.kind != CC

    def reference(self, env, x="A", target="B"):
        """Y's role: the target body; for a state primitive the virtual spot at X's base; for a hand-object
        primitive a virtual spot on the table under X (only X's height above it is used)."""
        if self.kind == CC:
            if isinstance(target, str) and target not in getattr(env, "cubes", [target]):
                return getattr(env, "ref", target)      # single-body world: the virtual reference
            return target
        if x in getattr(env, "fixtures", {}):
            return base_spot(env, x)
        return Spot(0.0, 0.0, 0.0)

    def rel(self, env, sens, S=None, x="A", target="B"):
        """Relational quantities from raw sensors (pair view of x and its reference), or from a view (S)."""
        if S is None:
            sens, S = env.pair_view(sens, x, self.reference(env, x, target)), VIEW_S
        c = cc_features(sens, S, "AB")
        z = relational_z(sens, S)
        if self.kind == HC and not self.x_is_part:
            z[..., 0:2] = 0.0                    # only X's height above the table matters
            z[..., 3:5] = 0.0
        out = {"z": z, "mode": self.R.mode(sens, S), "tiltB": c["tilt_y"],
               "B_speed": np.linalg.norm(sens[..., S.B_vel], axis=-1),
               "A_speed": np.linalg.norm(sens[..., S.A_vel], axis=-1),
               "width_m": sens[..., S.finger1] + sens[..., S.finger2]}
        out["pXY"] = out["z"][..., 0:3]
        return out

    def typical_scale(self):
        """(W, sigma): exploration's metric on the goal slice, and the members' median residual from the
        relation's centre in it."""
        W = np.asarray(self.metric_of(self.goal_mode), float)[self.goal_slice]
        key = W.tobytes()
        if getattr(self, "_sigma", (None,))[0] != key:
            s = float(np.median((np.abs(self.goal_members - self.goal_center) * W).sum(-1)))
            self._sigma = (key, W, max(s, 1e-9))
        return self._sigma[1], self._sigma[2]

    def typicality(self, rel, center=None):
        """How typical a goal instance is: exp(-residual / sigma), the residual from the relation's centre (or a
        chosen one) in exploration's metric of the goal mode (typical_scale)."""
        W, sigma = self.typical_scale()
        c = self.goal_center if center is None else np.asarray(center, float)
        return np.exp(-(np.abs(rel["z"][..., self.goal_slice] - c) * W).sum(axis=-1) / sigma)

    def is_goal(self, rel):
        return rel["mode"] == self.goal_mode

    def goal_residual(self, rel):
        """Distance (object units) from the target relation's typical value."""
        return np.abs(rel["z"][..., self.goal_slice] - self.goal_center).sum(axis=-1)

    def safety(self, rel, dt: float = CONTROL_DT):
        """Leave Y alone: its tilt (relative to the learned resting tilt) and its speed, priced like time
        spent away from the goal."""
        w = EXEC["side_effect_weight"] * EXEC["phi_weight"] * dt
        return w * (rel["tiltB"] / self.R.rest_tilt + rel["B_speed"] / L)

    def failed(self, rel):
        return "knocked_Y" if np.any(rel["tiltB"] > EXPLORE["knock_factor"] * self.R.rest_tilt) else None

    def transient(self, rel) -> bool:
        """X moving freely: no discovered relation holds and X is not at rest (falling, tipping)."""
        return int(rel["mode"]) == 0 and float(rel["A_speed"]) > PROBE["rest_speed"] and not self.x_is_part

    def metric_of(self, mode) -> np.ndarray:
        """Weights of the distance in a mode: the fixed weights W, or (once exploration has learned how fast
        each coordinate moves in that mode, bep.explore.field.learned_metric) W rescaled per coordinate."""
        return self.metric.get(int(mode), W)

    def set_metric(self, metric: dict) -> None:
        self.metric = {int(m): np.asarray(w, float) for m, w in (metric or {}).items()}

    def distance(self, z, Z, mode):
        return np.linalg.norm((np.asarray(z) - Z) * self.metric_of(mode), axis=-1)

    def z_of_cfg(self, cfg) -> np.ndarray:
        """The relational state a configuration should produce, as ``rel`` reads it (before settling)."""
        z = z_of(cfg, self.kind)
        if self.kind == HC and not self.x_is_part:     # only X's height above the table matters (see rel)
            z[0:2] = 0.0
            z[3:5] = 0.0
        return z

    def initial_grip(self, rel0):
        return float(rel0["grip_cmd"])

    # ------------------------------------------------------- hand of god
    def sample_ctx(self, rng, pair=None):
        pair = tuple(pair or self.kinds)
        fx = next((k for k in pair if is_fixture(k)), None)
        if fx:
            reg = FIXTURES[fx].region
            anchor = np.array([rng.uniform(*reg[0]), rng.uniform(*reg[1]), rng.uniform(*reg[2])])
        else:
            anchor = TRAIN_REGION.sample(rng)
        return np.r_[anchor, TYPES.index(pair[0]), TYPES.index(pair[1])]

    def precheck(self, cfg: GConfig, ctx) -> bool:
        return precheck(cfg, ctx) is None

    def realize(self, env, cfg: GConfig, ctx, q_hint=None):
        """Realise cfg in the pair world of ctx's kinds (``env`` is ignored, kept for the baseline API)."""
        cfg = cfg.copy()
        cfg.pair = ctx_pair(ctx)
        cfg.park = self.virtual_y
        cfg.virtual_y = self.virtual_y
        r = realize(cfg, ctx, q_hint=q_hint, settle_s=EXPLORE["settle_s"])
        if isinstance(r, str):
            return r
        env, state, ctrl, _ = r
        rel = self.rel(env, env.sensors())
        if rel["tiltB"] > self.R.rest_tilt:
            return "unsettled"
        rel["grip_cmd"] = float(ctrl[-1]) / 255.0 * GRIP_MAX          # the gripper: the last actuator
        return state, ctrl, rel

    def cfg_from_view(self, view, ctrl, pair) -> GConfig:
        c = cc_features(view, VIEW_S, "AB")
        h = hc_features(view, VIEW_S)
        g = float(ctrl[-1])
        grip = GRIP_CLOSE if g < 1.0 else (GRIP_OPEN if g > 254.0 else GRIP_HOLD)
        pXY, yawXY = np.array(c["p_m"]), float(c["yaw"])
        if self.kind == HC and not self.x_is_part:          # only X's height above the table matters
            pXY[:2], yawXY = 0.0, 0.0
        return GConfig(pXY, yawXY, np.array(h["p_m"]), np.array(h["R_m"]),
                       float(h["width_m"]), grip, tuple(pair), float(joint_norm(view, VIEW_S, "A")),
                       float(joint_norm(view, VIEW_S, "B")))

    def cfg_from_z(self, z, pair, grip=GRIP_HOLD, rng=None) -> GConfig:
        """A metric configuration for ``pair`` with the relational state z (object units); the hand's rotation
        about its closing axis, which z does not fix, is the most downward approach (random with ``rng``)."""
        sx = kind_shape(pair[0])
        sy = kind_shape(pair[1]) if self.kind == CC else _spot_shape()
        pXY, pEX = to_metric_shapes(z[0:3], z[Z_PEX], sx, sy)
        if self.kind == HC:
            pXY[:2] = 0.0
        unit = GRIP_MAX if sx.kind == "compound" else sx.grasp_width
        return GConfig(pXY, decode_angle(z[3:5], sx.sym), pEX, hand_from_axis(z[Z_AXIS], rng),
                       float(np.clip(z[Z_OPEN] * unit, 0.0, GRIP_MAX)), grip, tuple(pair), float(z[Z_QX]),
                       float(z[Z_QY]))

    def node_cfg(self, nd) -> GConfig:
        """An archived node's configuration, read in a scratch copy of the pair world's data (the world itself
        may be running a call: restoring into it would teleport its bodies)."""
        pair = ctx_pair(nd.B_pose)
        env = pair_world(*pair)
        scratch = getattr(env, "_scratch", None)
        if scratch is None:
            scratch = env._scratch = mujoco.MjData(env.model)
        env.set_state(nd.state, nd.ctrl, data=scratch)
        view = env.pair_view(env.sensors(scratch), "A", self.reference(env))
        return self.cfg_from_view(view, nd.ctrl, pair)

    # --------------------------------------------------------- operators
    def propose(self, cfg: GConfig, rng, op: str | None = None) -> tuple[str, GConfig]:
        op = op or str(rng.choice(EXPLORE["ops"], p=EXPLORE["op_p"]))
        c = cfg.copy()
        if op == "swap" and (len(SHAPES) == 1 or self.x_is_part or self.y_is_part):
            op = "joint" if (self.x_is_part or self.y_is_part) else "XY"
        if op == "joint" and not (self.x_is_part or self.y_is_part):
            op = "XY"
        if op == "XY" and self.x_is_part:
            op = "joint"
        if op == "swap":
            pair = (str(rng.choice(SHAPES)), str(rng.choice(SHAPES)))
            return "swap", self.cfg_from_z(z_of(c, self.kind), pair, cfg.grip, rng)
        if op == "joint":
            if self.x_is_part and (not self.y_is_part or rng.random() < 0.5):
                c.qx = float(np.clip(c.qx + rng.normal(0, EXPLORE["joint_step"]), 0, 1))
            else:
                c.qy = float(np.clip(c.qy + rng.normal(0, EXPLORE["joint_step"]), 0, 1))
            return "joint", c
        if op == "rel" and self.pools:
            name = str(rng.choice(sorted(self.pools)))
            pool = self.pools[name]
            j = int(rng.integers(len(pool[1])))
            if pool[0] == CC and not self.x_is_part:
                c.pXY = to_metric(pool[1][j], np.zeros(3), c.pair)[0]
                c.yawXY = float(pool[2][j])
            elif pool[0] == HC:
                sx = kind_shape(c.pair[0])
                c.pEX = to_metric(np.zeros(3), pool[1][j], c.pair)[1]
                c.rEX = np.array(pool[2][j])
                unit = GRIP_MAX if sx.kind == "compound" else sx.grasp_width
                c.width = float(np.clip(pool[3][j] * unit, 0, GRIP_MAX))
                c.grip = int(rng.choice([GRIP_CLOSE, GRIP_HOLD]))
            return f"rel:{name}", c
        if op == "rel":
            op = "EX"
        if op in ("XY", "EX"):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            key = "pXY" if op == "XY" else "pEX"
            setattr(c, key, getattr(c, key) + v * self.steps[op] * rng.uniform(0.3, 1.0))
            if op == "XY":
                c.yawXY = float(c.yawXY + rng.normal(0, 0.2))
            else:
                c.rEX = rotvec_mat(rng.normal(0, 0.2, 3)) @ c.rEX
        elif op == "rot":                                # turn the hand about the pinch point, any amount
            c.rEX = rotvec_mat(rng.normal(0, EXPLORE["rot_step"], 3)) @ c.rEX
        else:
            c.width = float(rng.uniform(0.0, GRIP_MAX))
            c.grip = int(rng.choice([GRIP_OPEN, GRIP_CLOSE, GRIP_HOLD]))
        return op, c

    def toward(self, cfg: GConfig, target: GConfig, rng) -> tuple[str, GConfig]:
        """Directed variant: step the quantity with the larger gap toward ``target`` (same pair)."""
        c = cfg.copy()
        gaps = {"EX": np.linalg.norm(target.pEX - cfg.pEX)}
        if not self.x_is_part:
            gaps["XY"] = np.linalg.norm(target.pXY - cfg.pXY)
        if self.x_is_part or self.y_is_part:
            gaps["joint"] = abs(target.qx - cfg.qx) + abs(target.qy - cfg.qy)
        ops = list(gaps)
        p = np.array([gaps[o] for o in ops]) + 1e-9
        op = ops[int(rng.choice(len(ops), p=p / p.sum()))]
        if op == "joint":
            for k in ("qx", "qy"):
                g = getattr(target, k) - getattr(c, k)
                setattr(c, k, float(getattr(c, k) + np.clip(g, -EXPLORE["joint_step"], EXPLORE["joint_step"])))
            return "joint*", c
        key = "pXY" if op == "XY" else "pEX"
        x, t = getattr(c, key), getattr(target, key)
        gap = float(np.linalg.norm(t - x))
        if gap > 1e-6:
            d = (t - x) / gap + 0.3 * rng.normal(size=3)
            d /= np.linalg.norm(d)
            setattr(c, key, x + d * min(gap, self.steps[op] * rng.uniform(0.3, 1.0)))
        if op == "EX":                                   # and turn the hand toward the target's orientation
            v = mat_rotvec(target.rEX @ c.rEX.T)
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                c.rEX = rotvec_mat(v * min(1.0, EXPLORE["rot_step"] / n)) @ c.rEX
        return op + "*", c

    def set_chain(self, ends) -> None:
        """Where the primitives explored before this one actually end: the physical end states of their
        executed calls (``[(states, ctrls)]``, bep.explore.explore saves them as ends.npz).  A chained
        primitive starts from them (each primitive only needs to connect to what its siblings produce)."""
        self.chain = [(np.asarray(S), np.asarray(C)) for S, C in ends if len(S)]

    def _start_from_end(self, rng):
        """A start state: an end state of an earlier primitive's executed call, as it is (the hand, its grip,
        the body it holds), with this primitive's own Y placed on the table relative to X as in a probe rest
        state (object-object primitives with a free Y)."""
        env = pair_world(*self.kinds)
        for _ in range(100):
            S, C = self.chain[int(rng.integers(len(self.chain)))]
            k = int(rng.integers(len(S)))
            if not self._set_end_state(env, S[k], C[k]):
                continue
            if self.kind == CC and not self.y_is_part:
                row = self._s2o[int(rng.choice(self._start_rows))]
                c = cc_features(row[None], VIEW_S, "AB")
                view = env.pair_view(env.sensors(), "A", "B")
                pA, qA = view[VIEW_S.A_pos], view[VIEW_S.A_quat]
                yawA = float(np.arctan2(2 * (qA[0] * qA[3] + qA[1] * qA[2]), 1 - 2 * (qA[2] ** 2 + qA[3] ** 2)))
                yawB = yawA - float(c["yaw"][0])
                pB = pA - yaw_rot(yawB) @ np.asarray(c["p_m"][0], float)
                if not (TRAIN_REGION.x[0] - 0.1 < pB[0] < TRAIN_REGION.x[1] + 0.1 and abs(pB[1]) < 0.3):
                    continue
                env.place("B", float(pB[0]), float(pB[1]), yawB)
                mujoco.mj_forward(env.model, env.data)
                if env.bad_penetration():
                    continue
            rel = self.rel(env, env.sensors())
            view = env.pair_view(env.sensors(), "A", self.reference(env))
            return self.cfg_from_view(view, env.data.ctrl, self.kinds), env, rel["z"], rel["mode"]
        raise RuntimeError("no start state from the earlier primitives' end states")

    def _set_end_state(self, env, state, ctrl) -> bool:
        """Put an end state (saved on the learning's body, the floating hand) into ``env``: as it is on the same
        body; on the robot the Hand-of-God way, same bodies, finger opening and grip, the arm teleported by IK
        to the same hand pose (False if the arm cannot take it)."""
        src = pair_world(*self.kinds, body="float")
        if env is src or (env.floating and len(state) == env.nstate):
            env.set_state(state, ctrl)
            return True
        scratch = getattr(src, "_scratch", None)
        if scratch is None:
            scratch = src._scratch = mujoco.MjData(src.model)
        src.set_state(state, ctrl, data=scratch)
        pE, RE = src.hand_pose(scratch)
        d = env.data
        env.reset_robot()
        for n in env.free:
            a, b = env.cube_qadr[n], src.cube_qadr[n]
            d.qpos[a:a + 7] = scratch.qpos[b:b + 7]
        q, ok = teleport_ik(env, pE, RE)
        if not ok:
            return False
        d.qpos[env.arm_qadr] = q
        d.ctrl[:7] = q
        d.qpos[env.finger_qadr] = scratch.qpos[src.finger_qadr]
        d.ctrl[env.gi] = float(ctrl[-1])
        mujoco.mj_forward(env.model, d)
        return True

    def _start_rows_of(self):
        S = VIEW_S
        s2, pairs = self.oriented()
        self._s2o = s2
        rest = (np.linalg.norm(s2[:, S.A_vel], axis=1) < PROBE["rest_speed"]) & \
               (np.linalg.norm(s2[:, S.B_vel], axis=1) < PROBE["rest_speed"])
        if self.kind != CC:
            same = np.array([p[0] == self.kinds[0] for p in pairs])
            ok = same & rest & ~self.R.holds_all(s2, S)[self.target]
        else:
            same = np.array([p == self.kinds for p in pairs])
            f = cc_features(s2, S, "AB")
            upright = np.maximum(f["tilt_x"], f["tilt_y"]) <= self.R.rest_tilt
            ok = same & rest & upright & (self.R.mode(s2, S) == 0)
        return np.nonzero(ok)[0]

    def sample_start(self, rng, chain: bool | None = None):
        """A start configuration: with probability chain_p (when earlier primitives exist), an end state of one of
        their executed calls (``_start_from_end``); otherwise the hand at home and the roles as in a probe rest
        state in which the target does not hold.
        Returns (cfg, pair world, z, mode)."""
        from ..world.hog import _place
        if self.probes is None:
            raise RuntimeError("probe data needed for start states")
        if not hasattr(self, "_start_rows"):
            self._start_rows = self._start_rows_of()
        if self.chain and (rng.random() < EXPLORE["chain_p"] if chain is None else chain):
            return self._start_from_end(rng)                # where an earlier primitive actually ended
        pair = self.kinds
        env = pair_world(*pair)
        ctx = self.sample_ctx(rng, pair)
        if self.kind == HC and not self.x_is_part:          # X resting on the table under the reference
            sx = kind_shape(pair[0])
            cfg = GConfig(np.array([0.0, 0.0, sx.half[2] + _spot_shape().half[2]]), float(rng.uniform(-np.pi, np.pi)),
                          np.zeros(3), ee_down_rot(0.0), GRIP_MAX, GRIP_OPEN, pair)
        else:
            cfg = self.cfg_from_view(self._s2o[int(rng.choice(self._start_rows))], np.r_[HOME_Q, 255.0], pair)
        cfg.park = cfg.virtual_y = self.virtual_y
        d = env.data
        env.reset_robot()                                   # the hand (or the arm) at home, fingers open
        _place(env, cfg, ctx)
        mujoco.mj_forward(env.model, d)
        rel = self.rel(env, env.sensors())
        view = env.pair_view(env.sensors(), "A", self.reference(env))
        return self.cfg_from_view(view, d.ctrl, pair), env, rel["z"], rel["mode"]


def restore(env, state, ctrl, ctx):
    """Set a pair world to an archived state; fixture bases (model-level poses) come from the context."""
    for role, k in zip(("A", "B"), ctx_pair(ctx)):
        if is_fixture(k):
            env.place_fixture(role, *ctx[:3])
    env.set_state(state, ctrl)
    mujoco.mj_forward(env.model, env.data)


def _spot_shape():
    from ..world.objects import SPOT
    return SPOT


def z_of(cfg: GConfig, kind=CC) -> np.ndarray:
    """Relational state (object units) of a metric configuration (inverse of cfg_from_z)."""
    sx = kind_shape(cfg.pair[0])
    sy = kind_shape(cfg.pair[1]) if kind == CC else _spot_shape()
    rx = max(sx.half[0], sx.half[1])
    pn = np.asarray(cfg.pXY) / np.array([sy.half[0] + rx, sy.half[1] + rx, sx.half[2] + sy.half[2]])
    pen = np.asarray(cfg.pEX) / sx.ext
    unit = GRIP_MAX if sx.kind == "compound" else sx.grasp_width
    return np.concatenate([pn, sym_code(cfg.yawXY, sx.sym), pen, axis_code(np.asarray(cfg.rEX)[:, CLOSING], sx.sym),
                           [cfg.width / unit, cfg.qx, cfg.qy]])



class _ConstDict(dict):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def __getitem__(self, k):
        return self.v

    def get(self, k, default=None):
        return self.v
