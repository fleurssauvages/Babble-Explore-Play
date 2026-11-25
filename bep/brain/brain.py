"""The loop's brain: deliberation over schemas with partial goals, imagination before acting, and habits.

* Transitions are the original world model's (bep.brain.model.WorldModel: the effect models learned in play,
  corrected by real outcomes); the search is a forward Dijkstra over relation graphs, each call costing its
  expected time with retries, but it stops at the first state that satisfies the goal's literals (required
  and forbidden relation instances, the hand let go), not at the example's whole snapshot.
* The first call's continuous parameter (where on the table, for a placement on a table spot) is chosen
  among BRAIN['spots'] clear spots by the schema's own Phi from the current state.
* The first call's success probability blends the effect model with the schema's precondition network
  (bep.play.schemas), whose inputs include the current contact flags, Phi and the distance to the schema's
  graph.
* Imagination: before acting, the first call is rehearsed in a copy of the scene on the floating hand with
  the learning's executor (IMAGINE['check']); a rehearsal that fails counts as a failed call in that state
  and the plan is made again (twice at most).
* Habits: a decision (abstract state, goal) -> call that succeeded BRAIN['habit_trust'] times in a row is
  taken without search; a failure sends it back to deliberation.  Habits persist across the episodes of a
  process (arbitration by certainty: Daw, Niv & Dayan 2005).
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from ..brain.model import WorldModel, name
from ..play.effect import call_succeeded
from ..config import BRAIN, IMAGINE

HABITS: dict = {}


class LoopWorldModel(WorldModel):
    """The original world model, with the precondition network's estimate (``p_pre``, for calls from the real
    current state) averaged with the effect model's before the real outcomes correct it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.p_pre: dict = {}

    def outcomes(self, s, poses, key):
        from ..config import BRAIN as B
        ck = (s, key)
        if ck in self._cache:
            return self._cache[ck]
        t, x, tt = key
        m = self.models[t]
        tgt = self.spot(poses, x) if tt == "spot" else tt
        p = m.p_ok(s, poses, x, tgt)
        if ck in self.p_pre:
            p = 0.5 * (p + self.p_pre[ck])
        n_ok, n = self.real.get(ck, (0, 0))
        p = (p * B["real_prior"] + n_ok) / (B["real_prior"] + n)
        out = []
        if p >= B["min_p"]:
            s_ok = self.abstract(m.most_likely(s, self.objects, x, tgt, True))
            if s_ok != s:
                s_fail = self.abstract(m.most_likely(s, self.objects, x, tgt, False))
                out = [(p, s_ok, m.time(s, poses, x, tgt, True), m.update_poses(s, s_ok, poses, x, tgt, True)),
                       (1.0 - p, s_fail, m.time(s, poses, x, tgt, False), m.update_poses(s, s_fail, poses, x, tgt, False))]
        self._cache[ck] = out
        return out


class LoopBrain:
    def __init__(self, models: dict, actions: list[tuple], objects, goal, *, spot_fn=None, seed: int = 0, env=None,
                 lib=None, R=None, scene=None, pre=None, imagine: bool = True, **_):
        self.model = LoopWorldModel(models, actions, objects, goal, spot_fn=spot_fn, seed=seed)
        self.goal, self.env, self.lib, self.R, self.scene = goal, env, lib, R, scene
        self.pre = pre or {}
        self.imagine = imagine and IMAGINE["check"] and env is not None
        self.plans, self.trace, self.imagined = 0, [], []
        self._copy = None

    def set_goal(self, goal) -> None:
        self.goal = goal
        self.model.set_goal(goal)

    def _goal_key(self):
        return frozenset(self.goal.require), frozenset(self.goal.forbid)

    def satisfied(self, s) -> bool:
        return self.goal.unmet(s) == 0

    # --------------------------------------------------------- search
    def search(self, s0, poses0, banned: set):
        M = self.model
        g, back, poses = {s0: 0.0}, {s0: None}, {s0: poses0}
        heap, n, settled = [(0.0, 0, s0)], 1, set()
        goal_state = None
        while heap and len(settled) < BRAIN["max_states"]:
            c, _, s = heapq.heappop(heap)
            if s in settled:
                continue
            settled.add(s)
            if self.satisfied(s):
                goal_state = s
                break
            for i, k in enumerate(M.keys):
                if s == s0 and i in banned:
                    continue
                o = M.outcomes(s, poses[s], k)
                if not o:
                    continue
                s2, c2 = o[0][1], c + M.expected_cost(o)
                if c2 < g.get(s2, np.inf):
                    g[s2], back[s2], poses[s2] = c2, (i, s), o[0][3]
                    heapq.heappush(heap, (c2, n, s2))
                    n += 1
        self.n_found = len(settled)
        if goal_state is None:
            return np.inf, []
        path, s = [], goal_state
        while back[s] is not None:
            i, s = back[s]
            path.append(i)
        return g[goal_state], path[::-1]

    # --------------------------------------------------------- the first call: parameters, preconditions, rehearsal
    def _spot(self, poses, x, t):
        """Among clear table spots, the one the schema's Phi says is quickest from the current state."""
        from ..world.geometry import Spot
        from ..world.objects import VIEW_S
        if self.env is None or t not in self.lib or not BRAIN["phi_spots"]:
            return self.model.spot(poses, x)
        prim, field, _ = self.lib[t]
        cands = [self.model.spot(poses, x) for _ in range(BRAIN["spots"])]
        sens = self.env.sensors()
        best, bv = cands[0], np.inf
        for c in cands:
            sp = Spot(float(c[0]), float(c[1]), 0.0)
            rel = prim.rel(self.env, self.env.pair_view(sens, x, prim.reference(self.env, x, sp)), VIEW_S)
            v = float(field.value(rel["z"], int(rel["mode"])))
            if v < bv:
                best, bv = c, v
        return best

    def _copy_world(self):
        """A floating-hand copy of the scene, in the current state (imagination's body)."""
        import mujoco
        from ..world.objects import use_body, body_kind
        from ..world.scenes import SCENES, build_scene
        if self._copy is None:
            kind = body_kind()
            use_body("float")
            try:
                self._copy = build_scene(SCENES[self.scene], "imagine", body="float")
            finally:
                use_body(kind)
        cp, env = self._copy, self.env
        cp.reset_robot()
        for n in env.free:
            a, b = cp.cube_qadr[n], env.cube_qadr[n]
            cp.data.qpos[a:a + 7] = env.data.qpos[b:b + 7]
        p, R = env.hand_pose()
        cp.set_hand(p, R)
        cp.data.qpos[cp.finger_qadr] = env.data.qpos[env.finger_qadr]
        cp.data.ctrl[cp.gi] = env.data.ctrl[env.gi]
        mujoco.mj_forward(cp.model, cp.data)
        return cp

    def _rehearse(self, key, spot) -> bool:
        """Imagine the call: run it in the floating-hand copy with the learning's executor."""
        from ..motor.hog_motor import run_primitive_hop
        from ..world.geometry import Spot
        from ..world.graph import as_tuples, relation_graph
        t, x, tt = key
        if ">" in t:                                    # a macro-schema: rehearse its first schema
            t, tt = t.split(">")[0], None
        prim, field, _ = self.lib[t]
        cp = self._copy_world()
        target = Spot(float(spot[0]), float(spot[1]), 0.0) if tt == "spot" else tt
        before = as_tuples(relation_graph(cp, self.R))
        out = run_primitive_hop(cp, self.R, prim, field, x, target, timeout=IMAGINE["check_timeout_s"])
        after = as_tuples(relation_graph(cp, self.R))
        from ..play.effect import made_instance
        made = made_instance(prim.kind, t, x, target if isinstance(target, str) else None)
        return call_succeeded(out["result"], made, before, after)

    def _precondition(self, s0, poses, key) -> None:
        """Blend the first call's success probability with the schema's precondition network."""
        t, x, tt = key
        if t not in self.pre or self.env is None:
            return
        from ..play.schemas import context_record, features
        from ..world.geometry import Spot
        prim, field, _ = self.lib[t]
        target = None if tt is None else (Spot(*self.model.spot(poses, x), 0.0) if tt == "spot" else tt)
        rec = context_record(self.env, prim, field, x, target, s0, poses)
        p_net, _ = self.pre[t].predict(features(self.model.models[t], rec)[None])
        self.model.p_pre[(s0, key)] = float(p_net[0])
        self.model._cache.pop((s0, key), None)

    # --------------------------------------------------------- decide / observe
    def decide(self, rels, poses: dict, allowed: set | None = None):
        t0 = time.time()
        M = self.model
        s0 = M.abstract(rels)
        hkey = (s0, self._goal_key())
        habit = HABITS.get(hkey)
        if habit is not None and habit[1] >= BRAIN["habit_trust"] and habit[0] in M.keys:
            key = habit[0]
            spot = self._spot(poses, key[1], key[0]) if key[2] == "spot" else None
            self.last_spot, self.last_habit = spot, True
            return key, spot, {"plan": [name(key)], "plan_keys": [key], "habit": True, "plan_s": round(time.time() - t0, 3)}
        self.last_habit = False
        for i, k in enumerate(M.keys):
            if M.outcomes(s0, poses, k):
                self._precondition(s0, poses, k)
        banned: set = set()
        if allowed is not None:
            banned = {i for i in range(len(M.keys)) if i not in allowed}
        rehearsals = []
        for _ in range(3):
            cost, path = self.search(s0, poses, banned)
            self.plans += 1
            if not path:
                break
            key = M.keys[path[0]]
            spot = self._spot(poses, key[1], key[0]) if key[2] == "spot" else None
            if not self.imagine or len(rehearsals) >= 2:
                break
            ok = self._rehearse(key, spot)
            rehearsals.append((name(key), ok))
            if ok:
                break
            n_ok, n = M.real.get((s0, key), (0, 0))
            M.real[(s0, key)] = (n_ok, n + 1)                  # an imagined failure counts as a failed call here
            M._cache.clear()
            banned.add(path[0])
        self.imagined.append(rehearsals)
        keys = [M.keys[i] for i in path]
        info = {"plan": [name(k) for k in keys], "plan_keys": keys, "plans": self.plans, "states": self.n_found,
                "V": round(cost, 1) if np.isfinite(cost) else None, "rehearsals": rehearsals,
                "plan_s": round(time.time() - t0, 3)}
        if not path:
            self.last_spot = None
            return None, None, info
        self.last_spot = spot
        return keys[0], spot, info

    def observe(self, rels, key: tuple, dur: float, rels2, poses=None, result: str = "done") -> dict:
        ok = self.model.observe(rels, key, dur, rels2, poses, result, getattr(self, "last_spot", None))
        hkey = (self.model.abstract(rels), self._goal_key())
        h = HABITS.get(hkey)
        HABITS[hkey] = (key, (h[1] + 1 if h is not None and h[0] == key else 1)) if ok else (key, 0)
        out = {"ok": ok, "habit": bool(getattr(self, "last_habit", False))}
        self.trace.append(out)
        return out
