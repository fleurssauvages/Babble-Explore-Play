"""Goals over discovered relations, read from a goal example.

A goal is a set of literals over relation instances: instances that must hold (``require``), instances
that must not (``forbid``) and, optionally, no hand relation (the hand has let go).  It is read from an
*example*: the bodies arranged as they should end up (bep.brain.tasks); the relations that hold among
them there are required, the other object-object relations among them are forbidden.

``success_test`` is the shape-aware yardstick used to score episodes.
"""
from __future__ import annotations

import numpy as np

from ..config import GRIP_MAX
from ..world.graph import as_tuples, object_poses, relation_graph


class Goal:
    def __init__(self, require, hc_names=(), hand_free: bool = True, forbid=()):
        self.require, self.hc, self.hand_free, self.forbid = set(require), set(hc_names), hand_free, set(forbid)
        self.poses: dict = {}           # body -> position in the goal example (raw-distance diagnostic)
        self.lengths: dict = {}
        self.snapshot = frozenset()     # the example's whole relation graph (bep.brain.snapshot)
        self.snapshot_poses: dict = {}  # every body's position in the example

    def unmet(self, rels) -> int:
        n = sum(i not in rels for i in self.require) + sum(i in rels for i in self.forbid)
        if self.hand_free:
            n += self.holding(rels)
        return n

    def holding(self, rels) -> bool:
        return any(q in self.hc for (q, u, v) in rels)

    def distance(self, poses: dict) -> float:
        """Raw object distance to the example (object units, summed over the goal's bodies) after the best
        horizontal shift (a structure may stand anywhere)."""
        names = [n for n in self.poses if n in poses]
        if not names:
            return 0.0
        P = np.array([poses[n] for n in names], float)
        G = np.array([self.poses[n] for n in names], float)
        P[:, :2] += (G[:, :2] - P[:, :2]).mean(0)
        scale = [self.lengths.get(n, GRIP_MAX) for n in names]
        return float((np.linalg.norm(P - G, axis=1) / scale).sum())


def goal_from_example(env, R, bodies) -> Goal:
    """The goal = every object-object relation among ``bodies`` holding in the current (example) state;
    those that do not hold there are forbidden."""
    g = as_tuples(relation_graph(env, R))
    cc = {r.name for r in R.cc}
    req = {i for i in g if i[0] in cc and i[1] in bodies and i[2] in bodies}
    forbid = {(q, u, v) for q in cc for u in bodies for v in bodies if u != v and (q, u, v) not in g}
    goal = Goal(req, [r.name for r in R.hc], True, forbid)
    s = env.sensors()
    goal.poses = {n: s[env.sidx[f"{n}_pos"]].copy() for n in bodies}
    goal.lengths = {n: env.shape(n).length for n in bodies}
    goal.snapshot, goal.snapshot_poses = frozenset(g), object_poses(env)
    return goal


def success_test(env, support) -> bool:
    """Every body rests on its demonstrated support with the hand disengaged."""
    f = env.facts()
    return f["held"] is None and all(f["support"].get(n) == s for n, s in support.items())
