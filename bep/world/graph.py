"""The relation graph of a world with N bodies: every discovered relation on every pair of bodies.

* object-object relations on every ordered pair of bodies,
* hand-object relations on the hand and every body (instance (g, "hand", X)),
* object-state relations on every moving part (instance (s, part, "")),

all read through pair views (``ObjWorld.pair_view``), so a relation learned on one pair of shapes is
read identically on any other pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HAND
from .geometry import Spot
from .objects import VIEW_DIM, VIEW_S
from .relations import RelationSet


@dataclass(frozen=True)
class Inst:
    rel: str
    x: str                 # subject (HAND in hand-object relations)
    y: str


def relation_graph(env, R: RelationSet, sens: np.ndarray | None = None) -> set[Inst]:
    """Every discovered relation on every ordered pair of bodies, in one batched evaluation."""
    return relation_graphs(env, R, [env.sensors() if sens is None else sens])[0]


def relation_graphs(env, R: RelationSet, sens_list) -> list[set[Inst]]:
    """relation_graph of several states (sensor rows) in a single batched evaluation."""
    objs = env.cubes
    pairs = [(x, y) for x in objs for y in objs if x != y] or [(x, Spot(0.0, 0.0, 0.0)) for x in objs]  # a lone body
    n = len(pairs)
    views = np.stack([env.pair_view(np.asarray(sens_list), x, y) for x, y in pairs], axis=1).reshape(-1, VIEW_DIM)
    h = R.holds_all(views, VIEW_S)
    first, kd = {}, getattr(env, "kinds", {})
    for k, (x, y) in enumerate(pairs):
        first.setdefault(x, k)                      # hand-object and state relations: one view per body
    outs = []
    for i in range(len(sens_list)):
        out = set()
        for k, (x, y) in enumerate(pairs):
            for r in R.cc:
                if isinstance(y, str) and h[r.name][i * n + k] and r.applies(kd.get(x), kd.get(y)):
                    out.add(Inst(r.name, x, y))
        for x, k in first.items():
            for r in R.hc:
                if h[r.name][i * n + k] and r.applies(kd.get(x)):
                    out.add(Inst(r.name, HAND, x))
            if x in getattr(env, "fixtures", {}):
                for r in R.sc:
                    if h[r.name][i * n + k] and r.applies(kd.get(x)):
                        out.add(Inst(r.name, x, ""))
        outs.append(out)
    return outs


def as_tuples(graph) -> frozenset:
    return frozenset((i.rel, i.x, i.y) for i in graph)


def object_poses(env, sens=None) -> dict[str, np.ndarray]:
    s = env.sensors() if sens is None else sens
    return {n: s[env.sidx[f"{n}_pos"]].copy() for n in env.cubes}
