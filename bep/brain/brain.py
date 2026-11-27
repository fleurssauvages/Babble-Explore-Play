"""The loop's brain: deliberation over schemas with partial goals, and habits.

* Transitions are the original world model's (bep.brain.model.WorldModel: the effect models learned in play,
  corrected by real outcomes); the search over relation graphs prioritizes expected time with retries plus
  an estimate of the time still needed: the learned cost of each unmet goal literal (bep.brain.heuristic) or, without
  it, a penalty per unmet literal. It stops at the first state that satisfies the goal's literals
  (required and forbidden relation instances, the hand let go), not at the example's whole snapshot.
* The first call's continuous parameter (where on the table, for a placement on a table spot) is chosen
  among BRAIN['spots'] clear spots by the schema's own Phi from the current state.
* The first call's success probability blends the effect model with the schema's precondition network
  (bep.play.schemas), whose inputs include the current contact flags, Phi and the distance to the schema's
  graph.
* Planning is symbolic: the search runs on the schemas' effect models only, nothing is simulated before
  acting; a call that fails in reality is learned from (``observe``) and the brain plans again.
* Habits: a decision (abstract state, goal) -> call that succeeded BRAIN['habit_trust'] times in a row is
  taken without search; a failure sends it back to deliberation.  Habits persist across the episodes of a
  process (arbitration by certainty: Daw, Niv & Dayan 2005).
"""
from __future__ import annotations

import heapq
import time
from functools import partial

import numpy as np

from ..brain.model import WorldModel, name
from ..config import BRAIN

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
                 lib=None, R=None, pre=None, heuristic=None, backward=None, **_):
        self.model = LoopWorldModel(models, actions, objects, goal, spot_fn=spot_fn, seed=seed)
        self.goal, self.env, self.lib, self.R = goal, env, lib, R
        self.pre = pre or {}
        self.heuristic = heuristic          # learned cost of the unmet literals (bep.brain.heuristic), else a count
        self.backward = backward            # the schemas as lifted operators (bep.brain.regression): bidirectional
        self.plans, self.trace = 0, []

    def set_goal(self, goal) -> None:
        self.goal = goal
        self.model.set_goal(goal)

    def _goal_key(self):
        return frozenset(self.goal.require), frozenset(self.goal.forbid)

    def satisfied(self, s) -> bool:
        return self.goal.unmet(s) == 0

    def h(self, s) -> float:
        """The search's estimate of the time still needed from s: the learned literal costs when given, else
        BRAIN['goal_weight_s'] per unmet literal."""
        if self.heuristic is not None:
            return BRAIN["h_weight"] * self.heuristic(self.goal, s)
        return BRAIN["goal_weight_s"] * self.goal.unmet(s)

    # --------------------------------------------------------- search
    def search(self, s0, poses0, banned: set, *, on_expand=None):
        """(expected cost, call indices) of a plan from s0 to the goal: forward, or bidirectional with a backward
        model."""
        if self.backward is not None:
            return self._search_both(s0, poses0, banned, on_expand=on_expand)
        return self._search_forward(s0, poses0, banned, on_expand=on_expand)

    def _search_both(self, s0, poses0, banned: set, *, on_expand=None):
        """Alternate a forward step (as ``_search_forward``) and a backward step (goal regression through the
        schemas' lifted operators, bep.brain.regression).  A forward state that satisfies a backward subgoal is a
        meeting point: the subgoal's calls are replayed from it with the effect models, and the first plan that
        reaches the goal that way (or a forward state that satisfies the goal) is returned, priced forward."""
        M, B = self.model, self.backward
        limit = BRAIN["max_states"]
        g, back, poses = {s0: 0.0}, {s0: None}, {s0: poses0}
        fheap, n = [(self.h(s0), 0, 0.0, s0)], 1
        G0 = B.goal_subgoal(self.goal)
        bheap, calls = [(B.mean_cost * B.unmet(G0, s0), 0, 0.0, G0)], {G0: ()}
        reached, by_lit = [], {}                      # forward states expanded, indexed by their instances
        subgoals = [G0]
        nf = nb = 0
        found = None

        def path_to(s):
            out = []
            while back[s] is not None:
                i, s = back[s]
                out.append(i)
            return out[::-1]

        def meet(s, G):
            if not B.satisfies(s, G):
                return None
            st, ps, c = s, poses[s], g[s]
            for j, i in enumerate(calls[G]):
                if j == 0 and st == s0 and i in banned:
                    return None
                o = M.outcomes(st, ps, M.keys[i])
                if not o:
                    return None
                c += M.expected_cost(o)
                st, ps = o[0][1], o[0][3]
            return (c, path_to(s) + list(calls[G])) if self.satisfied(st) else None

        if on_expand is not None:
            on_expand(0, limit)
        while (fheap or bheap) and nf + nb < limit and found is None:
            while fheap:                                  # ---- one forward step
                _, _, c, s = heapq.heappop(fheap)
                if c != g[s]:
                    continue
                nf += 1
                if self.satisfied(s):
                    found = (g[s], path_to(s))
                    break
                for G in subgoals:
                    r = meet(s, G)
                    if r is not None and (found is None or r[0] < found[0]):
                        found = r
                if found is not None:
                    break
                reached.append(s)
                for inst in s:
                    by_lit.setdefault(inst, []).append(s)
                for i, k in enumerate(M.keys):
                    if s == s0 and i in banned:
                        continue
                    o = M.outcomes(s, poses[s], k)
                    if not o:
                        continue
                    s2, c2 = o[0][1], c + M.expected_cost(o)
                    if c2 < g.get(s2, np.inf):
                        g[s2], back[s2], poses[s2] = c2, (i, s), o[0][3]
                        heapq.heappush(fheap, (c2 + self.h(s2), n, c2, s2))
                        n += 1
                break
            if found is not None or not bheap:
                continue
            _, _, c, G = heapq.heappop(bheap)             # ---- one backward step
            nb += 1
            for i, k in enumerate(M.keys):
                G2 = B.regress(G, k)
                if G2 is None or G2 in calls:
                    continue
                calls[G2] = (i,) + calls[G]
                c2 = c + B.cost(k)
                heapq.heappush(bheap, (c2 + B.mean_cost * B.unmet(G2, s0), n, c2, G2))
                n += 1
                subgoals.append(G2)
                pos = G2[0]
                cands = by_lit.get(next(iter(pos)), []) if pos else reached
                for st in cands:
                    r = meet(st, G2)
                    if r is not None and (found is None or r[0] < found[0]):
                        found = r
            if on_expand is not None and (nf + nb) % 16 == 0:
                on_expand(nf + nb, limit)
        self.n_found, self.n_forward, self.n_backward = nf + nb, nf, nb
        self.search_limited = found is None and nf + nb >= limit
        if on_expand is not None:
            on_expand(nf + nb, limit)
        return (np.inf, []) if found is None else found

    def _search_forward(self, s0, poses0, banned: set, *, on_expand=None):
        M = self.model
        g, back, poses = {s0: 0.0}, {s0: None}, {s0: poses0}
        limit = BRAIN["max_states"]
        heap, n, expanded = [(self.h(s0), 0, 0.0, s0)], 1, 0
        goal_state = None
        if on_expand is not None:
            on_expand(0, limit)
        while heap and expanded < limit:
            _, _, c, s = heapq.heappop(heap)
            if c != g[s]:
                continue
            expanded += 1
            if on_expand is not None and expanded % 16 == 0:
                on_expand(expanded, limit)
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
                    priority = c2 + self.h(s2)
                    heapq.heappush(heap, (priority, n, c2, s2))
                    n += 1
        self.n_found = expanded
        self.search_limited = goal_state is None and any(cost == g[state] for _, _, cost, state in heap)
        if on_expand is not None and expanded % 16 != 0:
            on_expand(expanded, limit)
        if goal_state is None:
            return np.inf, []
        path, s = [], goal_state
        while back[s] is not None:
            i, s = back[s]
            path.append(i)
        return g[goal_state], path[::-1]

    # --------------------------------------------------------- the first call: parameters, preconditions
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
    def decide(self, rels, poses: dict, allowed: set | None = None, *, on_progress=None):
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
        on_expand = partial(on_progress, "search") if on_progress is not None else None
        cost, path = self.search(s0, poses, banned, on_expand=on_expand)
        self.plans += 1
        key = M.keys[path[0]] if path else None
        spot = self._spot(poses, key[1], key[0]) if key is not None and key[2] == "spot" else None
        keys = [M.keys[i] for i in path]
        info = {"plan": [name(k) for k in keys], "plan_keys": keys, "plans": self.plans, "states": self.n_found,
                "state_limit_reached": self.search_limited,
                "V": round(cost, 1) if np.isfinite(cost) else None, "plan_s": round(time.time() - t0, 3)}
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
