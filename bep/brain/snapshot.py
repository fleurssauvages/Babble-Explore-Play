"""The snapshot brain: the cheapest sequence of calls from the current scene to the goal example.

The goal is the example's whole relation graph (its snapshot: which body rests on which, what the hand
holds), not a list of literals.  The brain values scene states by their expected time from the scene:

    states:  those the effect models (bep.brain.model) predict from the current scene, call after call;
    search:  forward Dijkstra over the predicted transitions s -a-> s', each costing the expected time of a
             with retries (success probability, feasibility and durations learned in play), expanding states
             as they are settled and stopping when the snapshot is: the first solution is the cheapest.

States are relation graphs only (contacts: which body rests on which, what the hand holds), not positions,
so the current scene is matched by its graph.  The order of a construction falls out of the values (a tower
is built bottom first), with no goal literal decomposition.

Decision: the first call of the cheapest plan (restricted, when any is available, to the calls with
validated experience from the real state, as the tree brain's root).  The search restarts at every decision,
after the real outcome has refit the models (bep.brain.model.WorldModel.observe).
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from ..config import BRAIN
from .model import WorldModel, name


class SnapshotBrain:
    def __init__(self, models: dict, actions: list[tuple], objects, goal, *, spot_fn=None, seed: int = 0, **_):
        self.model = WorldModel(models, actions, objects, goal, spot_fn=spot_fn, seed=seed)
        self.goal = goal
        self.plans, self.trace = 0, []

    def set_goal(self, goal) -> None:
        self.goal = goal
        self.model.set_goal(goal)

    def search_forward(self, s_start, poses_start, allowed: set | None = None):
        """Forward Dijkstra from the current scene over the predicted transitions (expected time of each call
        with retries), expanding states as they are settled and stopping as soon as the snapshot is: the first
        solution is the cheapest.  ``allowed`` restricts the first call (when any of them is available).
        Returns (cost, [call indices]) or (inf, [])."""
        M = self.model
        root = M.abstract(self.goal.snapshot)
        g, back, poses = {s_start: 0.0}, {s_start: None}, {s_start: poses_start}
        heap, n, settled = [(0.0, 0, s_start)], 1, set()
        first = None
        if allowed is not None:
            avail = {i for i, k in enumerate(M.keys) if M.outcomes(s_start, poses_start, k)}
            first = allowed & avail or None
        while heap and len(settled) < BRAIN["snapshot_max_states"]:
            c, _, s = heapq.heappop(heap)
            if s in settled:
                continue
            settled.add(s)
            if s == root:
                break
            for i, k in enumerate(M.keys):
                if s == s_start and first is not None and i not in first:
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
        if root not in settled:
            return np.inf, []
        path, s = [], root
        while back[s] is not None:
            i, s = back[s]
            path.append(i)
        return g[root], path[::-1]

    def decide(self, rels, poses: dict, allowed: set | None = None):
        t0 = time.time()
        M = self.model
        s0 = M.abstract(rels)
        cost, path = self.search_forward(s0, poses, allowed)
        self.plans += 1
        keys = [M.keys[i] for i in path]
        info = {"plan": [name(k) for k in keys], "plan_keys": keys, "plans": self.plans, "states": self.n_found,
                "start_reached": bool(path), "V": round(cost, 1), "plan_s": round(time.time() - t0, 3)}
        if not path:
            self.last_spot = None
            return None, None, info
        key = keys[0]
        spot = M.spot(poses, key[1]) if key[2] == "spot" else None
        self.last_spot = spot
        return key, spot, info

    def observe(self, rels, key: tuple, dur: float, rels2, poses=None, result: str = "done") -> dict:
        ok = self.model.observe(rels, key, dur, rels2, poses, result, getattr(self, "last_spot", None))
        out = {"ok": ok}
        self.trace.append(out)
        return out
