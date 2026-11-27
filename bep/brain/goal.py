"""Goals over discovered relations, read from a goal example.

A goal is a set of literals over relation instances: instances that must hold (``require``), instances
that must not (``forbid``) and, optionally, no hand relation (the hand has let go).  It is read from an
*example*: the bodies arranged as they should end up (bep.brain.tasks).  Only literals the example shows
robustly enter the goal (``goal_from_example``):

* the example is settled as given and in GOAL['samples'] copies jittered by the Hand of God (each body
  displaced and turned a little, in its own units; moving parts' joints a little), each settled;
* required: the object-object and object-state instances among the example's bodies that hold in at least
  GOAL['robust'] of these samples;
* forbidden: near misses only, instances that (almost) never hold in the samples but are within GOAL['near']
  of holding in the example (the relation's own residual) or that appear in some jittered sample;
* every other instance is left free (``dont_care``): what the example shows only fragilely, and what it
  is far from showing, constrains nothing.

``success_test`` is the shape-aware yardstick used to score episodes.
"""
from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np

from ..config import GOAL, GRIP_MAX
from ..world.graph import as_tuples, object_poses, relation_graph, relation_graphs


class Goal:
    def __init__(self, require, hc_names=(), hand_free: bool = True, forbid=()):
        self.require, self.hc, self.hand_free, self.forbid = set(require), set(hc_names), hand_free, set(forbid)
        self.poses: dict = {}           # body -> position in the goal example (raw-distance diagnostic)
        self.lengths: dict = {}
        self.snapshot = frozenset()     # the example's whole relation graph (bep.brain.snapshot)
        self.snapshot_poses: dict = {}  # every body's position in the example
        self.dont_care: set = set()     # instances the example shows only fragilely (neither required nor forbidden)
        self.robustness: dict = {}      # instance -> share of the example's samples in which it holds

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


def _candidates(env, R, bodies) -> list[tuple]:
    """Every object-object instance on an ordered pair of ``bodies`` and every object-state instance of a
    moving part among them (the relation applying to the bodies' kinds, as bep.world.graph reads them)."""
    kd = getattr(env, "kinds", {})
    out = [(r.name, u, v) for r in R.cc for u in bodies for v in bodies
           if u != v and r.applies(kd.get(u), kd.get(v))]
    out += [(r.name, u, "") for r in R.sc for u in bodies if u in env.fixtures and r.applies(kd.get(u))]
    return out


def residuals(env, R, sens, insts) -> dict:
    """Distance (object units) of each instance's feature to its relation's members in state ``sens``
    (0: it holds geometrically)."""
    from ..world.geometry import Spot
    from ..world.objects import VIEW_S
    from ..world.relations import cc_features, sc_features
    out, feats = {}, {}
    for q, u, v in insts:
        r = R.get(q)
        if (u, v) not in feats:
            view = env.pair_view(sens, u, v if v else Spot(0.0, 0.0, 0.0))
            feats[(u, v)] = cc_features(view, VIEW_S, "AB") if v else sc_features(view, VIEW_S, "A")
        out[(q, u, v)] = float(np.asarray(r.residual_f(feats[(u, v)])).reshape(-1)[0])
    return out


def _joint_norm(env, n) -> float:
    lo, hi = env.fixtures[n].range
    return float((env.data.qpos[env.joint_qadr[n]] - lo) / (hi - lo))


def _jittered(env, bodies, rng):
    """The example jittered by the Hand of God (see the module docstring) and settled: its sensors, or None
    when every try interpenetrates.  Leaves the scene in that state."""
    from ..world.objects import realize_world
    st, ctrl = env.get_state(), env.data.ctrl.copy()
    for _ in range(GOAL["tries"]):
        env.set_state(st, ctrl)
        mujoco.mj_forward(env.model, env.data)
        for n in bodies:
            if n not in env.free:
                continue
            a = env.cube_qadr[n]
            p, quat = env.data.qpos[a:a + 3].copy(), env.data.qpos[a + 3:a + 7].copy()
            p[:2] += rng.normal(0.0, GOAL["jitter"] * env.shape(n).length, 2)
            turn, q = rng.normal(0.0, GOAL["yaw_jitter"]), np.empty(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)]), quat)
            env.place_cube(n, p[0], p[1], 0.0, z=p[2], quat=q)
        joints = {n: float(np.clip(_joint_norm(env, n) + rng.normal(0.0, GOAL["joint_jitter"]), 0, 1))
                  for n in bodies if n in env.fixtures}
        sens = realize_world(env, {}, None, joints=joints, settle_s=GOAL["settle_s"])
        if sens is not None:
            return sens
    return None


def example_samples(env, bodies, rng) -> list[np.ndarray]:
    """Sensors of the example settled as given, then of GOAL['samples'] jittered copies; the example's state
    is restored afterwards."""
    from ..world.objects import realize_world
    st, ctrl = env.get_state(), env.data.ctrl.copy()
    try:
        first = realize_world(env, {}, None, settle_s=GOAL["settle_s"])
        out = [env.sensors() if first is None else first]
        for _ in range(GOAL["samples"]):
            env.set_state(st, ctrl)
            mujoco.mj_forward(env.model, env.data)
            s = _jittered(env, bodies, rng)
            if s is not None:
                out.append(s)
    finally:
        env.set_state(st, ctrl)
        mujoco.mj_forward(env.model, env.data)
    return out


def goal_from_example(env, R, bodies, rng=None) -> Goal:
    """The goal the example shows robustly (see the module docstring)."""
    rng = np.random.default_rng(GOAL["seed"]) if rng is None else rng
    bodies = list(bodies)
    g = as_tuples(relation_graph(env, R))
    cand = _candidates(env, R, bodies)
    samples = example_samples(env, bodies, rng)
    graphs = [as_tuples(h) for h in relation_graphs(env, R, samples)]
    cset = set(cand)
    freq = Counter(i for h in graphs for i in h if i in cset)
    n = len(graphs)
    share = {i: freq[i] / n for i in cand}
    req = {i for i in cand if share[i] >= GOAL["robust"]}
    rarely = [i for i in cand if share[i] <= 1.0 - GOAL["robust"]]
    res = residuals(env, R, env.sensors(), [i for i in rarely if i not in g])
    forbid = {i for i in rarely if i not in g and (share[i] > 0 or res[i] <= GOAL["near"])}
    goal = Goal(req, [r.name for r in R.hc], True, forbid)
    goal.dont_care = set(cand) - req - forbid
    goal.robustness = {i: round(share[i], 3) for i in cand if share[i] > 0 or i in forbid}
    s = env.sensors()
    goal.poses = {n: s[env.sidx[f"{n}_pos"]].copy() for n in bodies}
    goal.lengths = {n: env.shape(n).length for n in bodies}
    goal.snapshot, goal.snapshot_poses = frozenset(g), object_poses(env)
    return goal


def success_test(env, support) -> bool:
    """Every body rests on its demonstrated support with the hand disengaged."""
    f = env.facts()
    return f["held"] is None and all(f["support"].get(n) == s for n, s in support.items())
