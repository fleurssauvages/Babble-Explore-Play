"""The brains' transition model: what a primitive call does to the abstract state, from the effect models.

The abstract state is the relation graph restricted to the relations the brain can act on (its
primitives' relations) or the goal mentions, as a frozenset of instances.  For a call (primitive, X,
target) from state s, the effect model of the primitive (bep.play.effect, learned in play) gives

    p         the probability that the call succeeds (corrected by the real outcomes seen in this task),
    s_ok      the most likely graph after a success, and s_fail after a failure,
    t_ok/fail the expected durations,

so a call has two outcomes, (p, s_ok, t_ok) and (1 - p, s_fail, t_fail).  Calls less likely than
BRAIN['min_p'] or predicted to change nothing are not available.  Body positions are carried along
(the effect model's pose update) only for the duration estimates.

Learning from real calls (``observe``): the effect model of the primitive is refit with the call
(weighted BRAIN['real_weight']), and the call's real success count in that state is blended with the
model's prediction (BRAIN['real_prior'] calls of prior), so a call that keeps failing is avoided.
"""
from __future__ import annotations

import numpy as np

from ..config import BRAIN, table_spot
from ..play.effect import call_succeeded, made_instance


def name(key) -> str:
    t, x, tt = key
    return f"{t}({x})" if tt is None else f"{t}({x}, {tt})"


class WorldModel:
    def __init__(self, models: dict, actions: list[tuple], objects, goal, *, spot_fn=None, seed: int = 0):
        """models: primitive -> EffectModel (with the world's shapes set); actions: (primitive, X, target) with
        target a body, "spot" or None."""
        self.models, self.objects, self.goal = models, list(objects), goal
        self.keys = [a for a in actions if a[0] in models]
        self.spot_fn = spot_fn or table_spot
        self.rng = np.random.default_rng(seed)
        self.names = set(models) | {i[0] for i in goal.require | goal.forbid}
        self.real: dict = {}                      # (state, call) -> (successes, tries) in reality
        self._cache: dict = {}

    def set_goal(self, goal) -> None:
        self.goal = goal
        self.names = set(self.models) | {i[0] for i in goal.require | goal.forbid}
        self._cache.clear()

    def abstract(self, rels) -> frozenset:
        return frozenset(tuple(i) for i in rels if i[0] in self.names)

    def spot(self, poses: dict, x: str) -> list[float]:
        """The table spot, among a few candidates, farthest from the other bodies."""
        others = [np.asarray(p)[:2] for n, p in poses.items() if n != x]
        best, bd = None, -1.0
        for _ in range(12):
            sp = self.spot_fn(self.rng)
            d = min([np.hypot(sp[0] - o[0], sp[1] - o[1]) for o in others], default=1.0)
            if d > bd:
                best, bd = list(sp), d
        return best

    def made(self, key) -> tuple | None:
        t, x, tt = key
        return made_instance(self.models[t].prim_kind, t, x, tt if tt != "spot" else None)

    def outcomes(self, s: frozenset, poses: dict, key: tuple) -> list:
        """[(probability, next state, duration, next poses)] of a call from s: success first, then failure;
        [] when the call is unavailable."""
        ck = (s, key)
        if ck in self._cache:
            return self._cache[ck]
        t, x, tt = key
        m = self.models[t]
        tgt = self.spot(poses, x) if tt == "spot" else tt
        out = []
        p = m.p_ok(s, poses, x, tgt)
        n_ok, n = self.real.get(ck, (0, 0))
        p = (p * BRAIN["real_prior"] + n_ok) / (BRAIN["real_prior"] + n)
        if p >= BRAIN["min_p"]:
            s_ok = self.abstract(m.most_likely(s, self.objects, x, tgt, True))
            if s_ok != s:
                s_fail = self.abstract(m.most_likely(s, self.objects, x, tgt, False))
                out = [(p, s_ok, m.time(s, poses, x, tgt, True), m.update_poses(s, s_ok, poses, x, tgt, True)),
                       (1.0 - p, s_fail, m.time(s, poses, x, tgt, False),
                        m.update_poses(s, s_fail, poses, x, tgt, False))]
        self._cache[ck] = out
        return out

    def expected_cost(self, outs) -> float:
        """Expected time until the call succeeds, retrying after failures: (p t_ok + (1 - p) t_fail) / p."""
        (p, _, t_ok, _), (_, _, t_fail, _) = outs
        return (p * t_ok + (1 - p) * t_fail) / p

    def observe(self, rels, key, dur: float, rels2, poses, result: str, spot=None) -> bool:
        """A real call: success test, effect model refit, real success count.  Returns success."""
        t, x, tt = key
        ok = call_succeeded(result, self.made(key), rels, rels2)
        s = self.abstract(rels)
        n_ok, n = self.real.get((s, key), (0, 0))
        self.real[(s, key)] = (n_ok + ok, n + 1)
        tgt = None if tt is None else (list(spot) if tt == "spot" and spot is not None else tt)
        if poses is not None:
            self.models[t].learn(s, poses, x, tgt, ok, dur, weight=BRAIN["real_weight"])
        self._cache.clear()                      # the success model changed everywhere
        return ok
