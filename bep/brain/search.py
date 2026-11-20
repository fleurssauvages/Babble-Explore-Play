"""The brain: tree search over the relation graphs the effect models predict, pricing failures too.

Each call (primitive, X, target) from an abstract state s has two predicted outcomes (bep.brain.model):
success (probability p, next state s_ok, duration t_ok) and failure (s_fail, t_fail).  The value of a
call is its expected time to the goal,

    Q(s, a) = p (t_ok + V(s_ok)) + (1 - p) (t_fail + V(s_fail)),         V(s) = min_a Q(s, a),  V(goal) = 0,

so a failure costs what it undoes: a failed grasp under a stack drops the stack and V(s_fail) charges
the time to rebuild it, while a failure that leaves the state unchanged (s_fail = s) reduces to the
expected time with retries, (p t_ok + (1 - p) t_fail) / p.

Search (anytime, ``BRAIN['max_iters']`` iterations or ``BRAIN['max_plan_s']``): nodes are states, shared
between the paths that reach them.  Each iteration descends from the current state, choosing the call
with the lowest Q (minus an optional exploration bonus c sqrt(ln N / (1 + n)), ``explore_c``), samples
its outcome with the model's probability, stops at the goal, a state already on the path, or the depth
limit, and backs up full expectations (Bellman, not sample averages) along the path.  With the model
known this is real-time dynamic programming (Barto, Bradtke & Singh, 1995).

Unexpanded states are estimated additively, as the h_add heuristic of classical planning but from the
learned model: before each search, every literal gets a cost measured from the current state, for a
required literal the cheapest expected cost of making it with at most two predicted calls (e.g. grasp,
then place), for a forbidden literal or the held hand the cheapest call that removes it (a forbidden
literal that does not hold yet: the typical cost of its relation, since undoing takes the same calls as
making); a literal that holds now costs the typical cost of its relation; a state's
estimate is the sum over what is unmet there.  It is informed (realistic magnitudes, so the search
focuses) but not guaranteed optimistic.  The decision is the call with the
lowest Q in the current state; the search restarts at every decision, after the real outcome has refit
the models.

The search is written for discrete calls; continuous call parameters (where in a relation to place,
transformed primitives) and imagined physical outcomes will widen the same tree.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import BRAIN
from .model import WorldModel, name


class _Node:
    __slots__ = ("poses", "acts", "N", "n")

    def __init__(self, poses, acts):
        self.poses, self.acts = poses, acts      # acts: [(call index, outcomes)]
        self.N, self.n = 0, np.zeros(len(acts))


class TreeBrain:
    def __init__(self, models: dict, actions: list[tuple], objects, goal, *, spot_fn=None, seed: int = 0, **_):
        self.model = WorldModel(models, actions, objects, goal, spot_fn=spot_fn, seed=seed)
        self.goal = goal
        self.rng = np.random.default_rng(seed)
        self.plans, self.trace = 0, []

    def set_goal(self, goal) -> None:
        """A new task: the next decision searches toward it."""
        self.goal = goal
        self.model.set_goal(goal)

    # ------------------------------------------------------------------ values
    def unmet(self, s) -> int:
        u = self._unmet.get(s)
        if u is None:
            u = self._unmet[s] = self.goal.unmet(s)
        return u

    def _h(self, s) -> float:
        h = sum(self.c_lit.get(l, self.c_default) for l in self.goal.require if l not in s)
        h += sum(self.c_lit.get(l, self.c_default) for l in self.goal.forbid if l in s)
        if self.goal.hand_free and self.goal.holding(s):
            h += self.c_hand
        return h

    def _literal_costs(self, s0, root) -> None:
        """Cost of every literal from the current state: making a required one with at most two predicted
        calls (cheapest expected cost), removing a forbidden one or the hand's hold with one call."""
        M, G = self.model, self.goal
        make, remove = {}, {}
        first = [(M.expected_cost(o), o[0][1], o[0][3]) for _, o in root.acts]
        for c1, s1, ps1 in first:
            for l in s1 - s0:
                make[l] = min(make.get(l, np.inf), c1)
            for l in s0 - s1:
                remove[l] = min(remove.get(l, np.inf), c1)
        for c1, s1, ps1 in sorted(first, key=lambda f: f[0])[:BRAIN["h_first"]]:
            for k in M.keys:
                o = M.outcomes(s1, ps1, k)
                if o:
                    c2 = c1 + M.expected_cost(o)
                    for l in o[0][1] - s1 - s0:
                        make[l] = min(make.get(l, np.inf), c2)
        # a literal that holds now (or was not made in two calls) costs what its relation typically costs
        per_rel: dict = {}
        for l, c in make.items():
            per_rel.setdefault(l[0], []).append(c)
        typical = {q: float(np.median(v)) for q, v in per_rel.items()}
        self.c_lit = {l: make.get(l, typical.get(l[0], self.c_unit)) for l in G.require}
        # a forbidden literal not removable in one call now costs what making its relation costs (the same
        # calls take it apart: free the body, put it elsewhere)
        self.c_lit.update({l: remove.get(l, typical.get(l[0], self.c_unit)) for l in G.forbid})
        self.c_default = self.c_unit
        holds = [remove[l] for l in remove if l[1] == "hand"]
        self.c_hand = float(min(holds)) if holds else self.c_unit

    def V(self, s) -> float:
        if self.unmet(s) == 0:
            return 0.0
        v = self.Vt.get(s)
        return self._h(s) if v is None else v

    def Q(self, s, outs) -> float:
        (p, s_ok, t_ok, _), (_, s_f, t_f, _) = outs
        if s_f == s:                                         # failing changes nothing: retry until success
            return (p * (t_ok + self.V(s_ok)) + (1 - p) * t_f) / p
        return p * (t_ok + self.V(s_ok)) + (1 - p) * (t_f + self.V(s_f))

    def _node(self, s, poses) -> _Node:
        if s not in self.tree:
            acts = [(i, o) for i, k in enumerate(self.model.keys) if (o := self.model.outcomes(s, poses, k))]
            self.tree[s] = _Node(poses, acts)
        return self.tree[s]

    def _backup(self, s) -> None:
        node = self.tree[s]
        self.Vt[s] = min((self.Q(s, o) for _, o in node.acts), default=self._h(s) + BRAIN["dead_end_s"])

    # ------------------------------------------------------------------ search
    def _simulate(self, s0, poses0) -> None:
        path, s, ps, on_path = [], s0, poses0, {s0}
        for _ in range(self.depth):
            if self.unmet(s) == 0:
                break
            node = self._node(s, ps)
            if not node.acts:
                self.Vt[s] = self._h(s) + BRAIN["dead_end_s"]
                break
            q = np.array([self.Q(s, o) for _, o in node.acts])
            if BRAIN["explore_c"]:
                q = q - self.c_unit * BRAIN["explore_c"] * np.sqrt(np.log(node.N + 1) / (node.n + 1))
            k = int(np.argmin(q + 1e-9 * self.rng.random(len(q))))
            node.N += 1
            node.n[k] += 1
            outs = node.acts[k][1]
            o = outs[0] if self.rng.random() < outs[0][0] else outs[1]
            path.append(s)
            s, ps = o[1], o[3]
            if s in on_path:
                break
            on_path.add(s)
        for st in reversed(path):
            self._backup(st)

    def _plan(self, s0) -> list:
        """The greedy plan: lowest-Q calls along predicted successes (for display)."""
        out, s, seen = [], s0, {s0}
        while s in self.tree and self.tree[s].acts and self.unmet(s) and len(out) < self.depth:
            i, outs = min(self.tree[s].acts, key=lambda a: self.Q(s, a[1]))
            out.append(name(self.model.keys[i]))
            s = outs[0][1]
            if s in seen:
                break
            seen.add(s)
        return out

    # ------------------------------------------------------------------ interface (bep.brain.solve, bep.app)
    def decide(self, rels, poses: dict, allowed: set | None = None):
        """The next call.  ``allowed``: indices (in model.keys) of the calls whose primitive has validated
        experience from the current real state (bep.explore.field.covered); only those are tried at the root,
        deeper levels plan on the model (only the current state is real)."""
        t0 = time.time()
        s0 = self.model.abstract(rels)
        self.tree, self.Vt, self._unmet = {}, {}, {}
        root = self._node(s0, poses)
        if allowed is not None and any(i in allowed for i, _ in root.acts):
            root.acts = [(i, o) for i, o in root.acts if i in allowed]
            root.n = np.zeros(len(root.acts))
        costs = [self.model.expected_cost(o) for _, o in root.acts]
        self.c_unit = float(np.median(costs)) if costs else 1.0     # scale of the exploration bonus
        self._literal_costs(s0, root)
        self.depth = 2 * (len(self.goal.require) + len(self.goal.forbid & set(s0))) + 4
        it = 0
        while it < BRAIN["max_iters"] and time.time() - t0 < BRAIN["max_plan_s"]:
            self._simulate(s0, poses)
            it += 1
        self.plans += 1
        if root.acts:
            i, outs = min(root.acts, key=lambda a: self.Q(s0, a[1]))
            key = self.model.keys[i]
        else:                                                # nothing available: any call
            key = self.model.keys[int(self.rng.integers(len(self.model.keys)))]
        spot = self.model.spot(poses, key[1]) if key[2] == "spot" else None
        self.last_spot = spot
        return key, spot, {"plan": self._plan(s0), "plans": self.plans, "iterations": it, "states": len(self.tree),
                           "V": round(self.V(s0), 1), "plan_s": round(time.time() - t0, 3)}

    def observe(self, rels, key: tuple, dur: float, rels2, poses=None, result: str = "done") -> dict:
        ok = self.model.observe(rels, key, dur, rels2, poses, result, getattr(self, "last_spot", None))
        out = {"ok": ok}
        self.trace.append(out)
        return out
