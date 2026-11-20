"""The snapshot brain: Reverse Go-Explore over scenes, rooted at the goal example.

The goal is the example's whole relation graph (its snapshot: which body rests on which, what the hand
holds), not a list of literals.  As exploration grows a primitive's graph backward from its goal
(bep.explore.explore), the brain values scene states by their time to the snapshot:

    states:  those the effect models (bep.brain.model) predict from the snapshot (taking it apart) and from
             the current scene, call after call;
    values:  backward Dijkstra from the snapshot over every predicted transition s -a-> s' among them,
             V(s) = min_a  expected time of a with retries + V(s'),  V(snapshot) = 0.

States are relation graphs only (contacts: which body rests on which, what the hand holds), not positions,
so the current scene is matched by its graph.  The order of a construction falls out of the values (a tower
is dismantled top first, hence built bottom first), with no goal literal decomposition.

Decision: among the calls available now (those with validated experience from the real state, as the tree
brain's root), the one minimising expected time + V(predicted success).  The search restarts at every
decision, after the real outcome has refit the models (bep.brain.model.WorldModel.observe).
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

    def search(self, s_start, poses_start):
        """States predicted from the snapshot (taken apart) and from the current scene, call after call (up to
        BRAIN['snapshot_max_states']); then cost-to-go by backward Dijkstra from the snapshot over every predicted
        transition among them.  Returns (V, next) over the states that reach the snapshot."""
        M = self.model
        root = M.abstract(self.goal.snapshot)
        found = {root: self.goal.snapshot_poses, s_start: poses_start}
        queue, k0, pred = [root, s_start], 0, {}
        while k0 < len(queue) and len(found) < BRAIN["snapshot_max_states"]:
            s = queue[k0]
            k0 += 1
            for i, k in enumerate(M.keys):
                o = M.outcomes(s, found[s], k)
                if not o:
                    continue
                s2 = o[0][1]
                pred.setdefault(s2, []).append((M.expected_cost(o), i, s))
                if s2 not in found:
                    found[s2] = o[0][3]
                    queue.append(s2)
        V, nxt, heap, settled, n = {root: 0.0}, {root: None}, [(0.0, 0, root)], set(), 1
        while heap:
            v, _, s = heapq.heappop(heap)
            if s in settled:
                continue
            settled.add(s)
            for c, i, u in pred.get(s, ()):
                if v + c < V.get(u, np.inf):
                    V[u], nxt[u] = v + c, (i, s)
                    heapq.heappush(heap, (V[u], n, u))
                    n += 1
        self.n_found = len(found)
        return V, nxt

    def decide(self, rels, poses: dict, allowed: set | None = None):
        t0 = time.time()
        M = self.model
        s0 = M.abstract(rels)
        V, nxt = self.search(s0, poses)
        self.plans += 1
        cands = []
        for i, k in enumerate(M.keys):
            o = M.outcomes(s0, poses, k)
            if o and o[0][1] in V:
                cands.append((M.expected_cost(o) + V[o[0][1]], i))
        if allowed is not None and any(i in allowed for _, i in cands):
            cands = [c for c in cands if c[1] in allowed]
        plan, keys, s, seen = [], [], s0, {s0}
        while nxt.get(s) is not None and len(plan) < 32:
            i, s = nxt[s]
            plan.append(name(M.keys[i]))
            keys.append(M.keys[i])
            if s in seen:
                break
            seen.add(s)
        info = {"plan": plan, "plan_keys": keys, "plans": self.plans, "states": self.n_found, "reach_goal": len(V), "start_reached": s0 in V,
                "V": round(V.get(s0, np.inf), 1), "plan_s": round(time.time() - t0, 3)}
        if not cands:
            self.last_spot = None
            return None, None, info
        key = M.keys[min(cands)[1]]
        spot = M.spot(poses, key[1]) if key[2] == "spot" else None
        self.last_spot = spot
        return key, spot, info

    def observe(self, rels, key: tuple, dur: float, rels2, poses=None, result: str = "done") -> dict:
        ok = self.model.observe(rels, key, dur, rels2, poses, result, getattr(self, "last_spot", None))
        out = {"ok": ok}
        self.trace.append(out)
        return out
