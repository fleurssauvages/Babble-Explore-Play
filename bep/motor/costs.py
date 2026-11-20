"""Spike-MPPI costs of the discovered primitives (all take rollout sensor data (N, H, ns)).

* ``LocalTargetCost``: drive the pair's relational state to one graph node (validates a backward
  proposal during exploration; the same cost tracks a node during execution by hops).
* ``PrimitiveCost``: execute a primitive on roles (X, target) in a world with N bodies: target = a
  body or a table ``Spot`` for object-object primitives, none for the others.  It is the primitive's
  field cost (or the cost of the graph node being tracked) plus two generic side-effect terms:
    keep        every relation that holds and does not involve X as subject; losing one costs as much
                as being as far from the goal as the field can express (its cap),
    leave alone every other free body: tilt (relative to the learned resting tilt) and speed.
    barrier     every undesired contact, per step: the hand or an arm link touching a body other than X, X touching a body
                other than its target (the table is not a body).
    goal        a discrete reward for the primitive's goal state (its relation holding alone, e.g. the hand
                having let go), EXEC['goal_reward'] x the share of the rollout's steps in it.
The barrier and the goal reward apply to either body (the native rollouts compute them too).
"""
from __future__ import annotations

import numpy as np

from ..config import CONTROL_DT, EXEC, HAND, L
from ..world.geometry import quat_to_mat
from ..world.objects import VIEW_S
from ..world.relations import SC, Z_OPEN, cc_features, sc_features, tilt_of


class LocalTargetCost:
    """sum_t d(z_t, z_target) + mode mismatch + safety (+ terminal), on the pair view of a two-body world."""

    def __init__(self, env, z_target: np.ndarray, mode_target: int, *, prim, mode_penalty: float = 0.05,
                 terminal_weight: float = 3.0) -> None:
        self.env, self.z, self.m, self.prim = env, np.asarray(z_target), int(mode_target), prim
        self.mode_penalty, self.terminal_weight = mode_penalty, terminal_weight

    def per_step(self, rel: dict) -> np.ndarray:
        if self.m == self.prim.goal_mode:
            d = self.prim.goal_residual(rel)
        else:
            d = self.prim.distance(rel["z"], self.z, self.m)
        return d + self.mode_penalty * (rel["mode"] != self.m) + self.prim.safety(rel)

    def __call__(self, sens: np.ndarray) -> np.ndarray:
        c = self.per_step(self.prim.rel(self.env, sens))
        return c.sum(axis=1) + self.terminal_weight * c[:, -1]


def _leave_alone(env, sens, names, rest_tilt, w):
    """Tilt and speed of every listed body, gathered in one pass."""
    if not names:
        return 0.0
    qi = np.stack([env.sidx[f"{n}_quat"] for n in names])
    vi = np.stack([env.sidx[f"{n}_vel"] for n in names])
    tilt = tilt_of(quat_to_mat(sens[..., qi]))
    speed = np.linalg.norm(sens[..., vi], axis=-1)
    return w * (tilt / rest_tilt + speed / L).sum(axis=-1)


class PrimitiveCost:
    def __init__(self, env, prim, field, x: str, target,
                 keep: list, *, dt: float = CONTROL_DT, center=None):
        """center: where in the goal relation to end (object units; default the relation's centre), e.g. a
        shifted placement chosen by the brain."""
        self.env, self.prim, self.field, self.x, self.dt = env, prim, field, x, dt
        self.center = np.asarray(prim.goal_center if center is None else center, float)
        self._shifted = center is not None
        self._target_arg = target
        self._tname = target if isinstance(target, str) else None
        self.target = prim.reference(env, x, target)
        self.keep = [k for k in keep if k.x != x and k.x != HAND]
        # bystanders: the bodies the call does not involve (not X, the target, or anything connected to them
        # through current relations); the controller keeps the arm links away from them in its null space
        tname0 = target if isinstance(target, str) else None
        involved = {x, tname0}                     # + every body connected to them through current relations
        grow = True                                # (the stack the target stands in does not repel)
        while grow:
            grow = False
            for k in keep:
                if k.x != HAND and k.y and (k.x in involved) != (k.y in involved):
                    involved |= {k.x, k.y}
                    grow = True
        self.bystanders = [n for n in env.cubes if n not in involved]
        radius = lambda n: float(np.max(env.shape(n).half))                       # noqa: E731
        self.rep_r = np.array([radius(n) for n in self.bystanders], float)     # (controller null space)
        self._keep_all = list(keep)
        self._keep_groups: dict = {}
        for k in self.keep:
            self._keep_groups.setdefault(k.rel, []).append(k)
        tname = target if isinstance(target, str) else None
        self.others = [n for n in env.free if n not in (x, tname)]
        self.w_step = EXEC["phi_weight"] * dt * EXEC["side_effect_weight"]
        self.w_break = self.w_step * field.cap
        # the barrier's contact sensors: the hand or an arm link on any body but X, X on any body but the target
        self.barrier = []
        for n in env.cubes:
            if n != x:
                self.barrier.append(int(env.sidx[f"body_{n}"][0]))      # the hand or any arm link
                if n != self._tname:
                    self.barrier.append(int(env.sidx[env.pair_contact(x, n)][0]))
        self.barrier = np.array(self.barrier, int)
        self.wp = None                     # waypoint (z, mode) when executing along the roadmap
        self._rel_cache = None

    def set_waypoint(self, z, mode: int) -> None:
        """Track one graph node, with the cost that validated the graph's edges (bep.costs.LocalTargetCost)
        plus the keep / leave-alone terms.  A goal node is tracked by its relation (at the chosen centre)
        and the hand's opening there (e.g. let go after a placement), not by where the hand rests."""
        z = np.array(z, float)
        if int(mode) == self.prim.goal_mode:
            z[self.prim.goal_slice] = self.center
        self.wp = (z, int(mode))
        if getattr(self, "native_spec", None) is not None:
            self.native_spec.update(cost_type=1, wp_resid=2 if self.wp[1] == self.prim.goal_mode else 0,
                                    z_target=self.wp[0], W_dist=np.asarray(self.prim.metric_of(self.wp[1]), float),
                                    m_target=self.wp[1], mode_penalty=EXEC["hop_mode_penalty"],
                                    terminal=EXEC["hop_terminal"])
            self.native_ver = getattr(self, "native_ver", 0) + 1

    def clear_waypoint(self) -> None:
        """Back to descending the field."""
        if self.wp is None:
            return
        self.wp = None
        if getattr(self, "native_spec", None) is not None:
            self.native_spec.update(cost_type=0, wp_resid=0, terminal=EXEC["terminal_weight"])
            self.native_ver = getattr(self, "native_ver", 0) + 1

    def __call__(self, sens: np.ndarray) -> np.ndarray:
        env, prim = self.env, self.prim
        rel = prim.rel(env, env.pair_view(sens, self.x, self.target), VIEW_S)
        if self.wp is None:
            phi = self.field.value(rel["z"], rel["mode"], rel)
            if self._shifted:                                # the goal mode's value: distance to the centre
                g = rel["mode"] == prim.goal_mode
                gs = prim.goal_slice
                phi = np.where(g, (np.abs(rel["z"][..., gs] - self.center) * self.field.W_goal[gs]).sum(-1), phi)
            # progress only on the final snapshot (its weight = the former per-step weight x H + terminal), so
            # that plans may pass through worse states (lift, open, let go) as long as they end well
            c = prim.safety(rel, self.dt)
            last, w_last = phi, EXEC["terminal_weight"] + self.dt * EXEC["phi_weight"] * sens.shape[1]
        else:
            zt, mt = self.wp
            if mt == prim.goal_mode:                         # the relation at the centre, and the hand's opening
                d = self._residual(rel) + prim.metric_of(mt)[Z_OPEN] * np.abs(rel["z"][..., Z_OPEN] - zt[Z_OPEN])
            else:
                d = prim.distance(rel["z"], zt, mt)
            last = d + EXEC["hop_mode_penalty"] * (rel["mode"] != mt)
            c = prim.safety(rel, self.dt)
            w_last = EXEC["hop_terminal"] + sens.shape[1]
        for rname, ks in self._keep_groups.items():          # one batched evaluation per kept relation
            r = getattr(prim, "Rall", prim.R).get(rname)
            if r.kind == SC:
                v = np.stack([env.pair_view(sens, k.x, next(n for n in env.cubes if n != k.x)) for k in ks])
                f = sc_features(v, VIEW_S, "A")
                c = c + (self.w_break * (~r.holds_f(f)) + self.w_step * np.abs(f["q"] - r.center[0])).sum(0)
            else:
                f = cc_features(np.stack([env.pair_view(sens, k.x, k.y) for k in ks]), VIEW_S, "AB")
                c = c + (self.w_break * (~r.holds_f(f)) + self.w_step * np.abs(f["p"] - r.center).sum(-1)).sum(0)
        c = c + _leave_alone(env, sens, self.others, prim.R.rest_tilt, self.w_step)
        if len(self.barrier):
            c = c + EXEC["barrier_weight"] * (sens[..., self.barrier] > 0).sum(axis=-1)
        out = c.sum(axis=1) + w_last * last[:, -1]
        # the goal, rewarded by how typical an instance it is
        typical = prim.typicality(rel, self.center)
        out = out - EXEC["goal_reward"] * ((rel["mode"] == prim.goal_mode) * typical).mean(axis=1)
        return out

    def _residual(self, rel):
        return np.abs(rel["z"][..., self.prim.goal_slice] - self.center).sum(axis=-1)

    def rel_now(self, sens: np.ndarray) -> dict:
        """Relational state of the real scene from its sensor row; evaluated once per sensor row (the done
        test, the navigator and the progress check of a control step share it)."""
        if self._rel_cache is None or self._rel_cache[0] is not sens:
            self._rel_cache = (sens, self.prim.rel(self.env, self.env.pair_view(sens, self.x, self.target), VIEW_S))
        return self._rel_cache[1]

    def done(self, sens: np.ndarray) -> bool:
        """In the goal mode: the relation holds and nothing else (e.g. the hand has let go), after which the
        call cannot improve anything; a chosen centre is aimed at, best effort."""
        rel = self.rel_now(sens)
        return bool(self.prim.is_goal(rel))

    def progress(self, sens: np.ndarray) -> float:
        """Field value (seconds to go) of the current state."""
        rel = self.rel_now(sens)
        return float(self.field.value(rel["z"], rel["mode"]))
