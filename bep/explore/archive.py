"""Reachability archive of one primitive: the graph of validated transitions + labelled field samples.

Nodes are realised states (relational state z, mode, full physical state).  An edge i -> j says that
Spike-MPPI, started at node i and tracking j, reached a solved node; it carries the time it took and a
reliability count: edges are re-validated now and then during exploration (bep.explore.explore), and
Dijkstra prices an edge by its expected time until success, c / p, with p = (successes + 1) / (tries + 1)
(the first validation counts as one success).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..world.relations import ZDIM
from ..world.objects import LEGACY_TYPES, TYPES


@dataclass
class Node:
    z: np.ndarray
    mode: int
    state: np.ndarray          # full physical state it was validated from
    ctrl: np.ndarray
    B_pose: np.ndarray         # context: anchor pose and kinds of the roles (bep.explore.primitive.ctx_pair)
    op: str
    parent: int                # node the backward proposal was generated from
    goal: bool = False
    tries: int = 0
    D: float = np.inf


@dataclass
class Archive:
    nodes: list[Node] = field(default_factory=list)
    edges: list[tuple[int, int, float]] = field(default_factory=list)       # i -> j, seconds
    e_tries: list[int] = field(default_factory=list)                         # validations of each edge
    e_succ: list[int] = field(default_factory=list)                          # ... that succeeded
    # Field samples from successful trajectories: (z, mode, target node, remaining seconds)
    sZ: list[np.ndarray] = field(default_factory=list)
    sM: list[int] = field(default_factory=list)
    sT: list[int] = field(default_factory=list)
    sR: list[float] = field(default_factory=list)

    def add_node(self, node: Node) -> int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_transition(self, i: int, j: int, cost: float, traj_z, traj_m, dt: float, stride: int = 2) -> None:
        self.edges.append((i, j, float(cost)))
        self.e_tries.append(1)
        self.e_succ.append(1)
        for t in range(0, len(traj_m), stride):
            self.sZ.append(np.asarray(traj_z[t], float))
            self.sM.append(int(traj_m[t]))
            self.sT.append(j)
            self.sR.append(float(cost - t * dt))

    def record_validation(self, k: int, ok: bool, cost: float | None = None) -> None:
        """A re-validation of edge k: its count, and on success a running mean of its time."""
        i, j, c = self.edges[k]
        if ok and cost is not None:
            c = (c * self.e_succ[k] + float(cost)) / (self.e_succ[k] + 1)
            self.edges[k] = (i, j, c)
        self.e_tries[k] += 1
        self.e_succ[k] += int(bool(ok))

    def reliability(self, k: int) -> float:
        return (self.e_succ[k] + 1.0) / (self.e_tries[k] + 1.0)

    def edge_cost(self, k: int) -> float:
        """Expected time until the edge succeeds (retrying it): c / p."""
        return self.edges[k][2] / self.reliability(k)

    # ----------------------------------------------------------- graph
    def dijkstra(self) -> np.ndarray:
        n = len(self.nodes)
        D = np.full(n, np.inf)
        rev: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for k, (i, j, _) in enumerate(self.edges):
            rev[j].append((i, self.edge_cost(k)))
        heap = []
        for k, nd in enumerate(self.nodes):
            if nd.goal:
                D[k] = 0.0
                heap.append((0.0, k))
        heapq.heapify(heap)
        while heap:
            d, j = heapq.heappop(heap)
            if d > D[j]:
                continue
            for i, c in rev[j]:
                if d + c < D[i]:
                    D[i] = d + c
                    heapq.heappush(heap, (D[i], i))
        for k, nd in enumerate(self.nodes):
            nd.D = float(D[k])
        return D

    def solved_arrays(self):
        """(z, mode, cost-to-go, node index) of every solved node (z as float32: shipped with every task)."""
        D = np.array([nd.D for nd in self.nodes])
        ok = np.isfinite(D)
        Z = np.array([nd.z for nd in self.nodes]).reshape(len(self.nodes), -1)
        M = np.array([nd.mode for nd in self.nodes], dtype=int)
        ids = np.nonzero(ok)[0]
        return Z[ok].astype(np.float32), M[ok], D[ok], ids

    def samples(self):
        """All labelled (z, mode, cost-to-go) pairs: nodes + trajectory states."""
        D = np.array([nd.D for nd in self.nodes])
        Z = [nd.z for nd in self.nodes] + self.sZ
        M = [nd.mode for nd in self.nodes] + self.sM
        lab = list(D) + [r + D[t] for r, t in zip(self.sR, self.sT)]
        return np.array(Z).reshape(len(Z), -1), np.array(M, int), np.array(lab)

    # ------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        nd = self.nodes
        np.savez_compressed(
            path,
            z=np.array([n.z for n in nd]), mode=np.array([n.mode for n in nd]),
            state=np.array([n.state for n in nd]), ctrl=np.array([n.ctrl for n in nd]),
            B_pose=np.array([n.B_pose for n in nd]), op=np.array([n.op for n in nd]),
            types=np.asarray(TYPES),
            parent=np.array([n.parent for n in nd]), goal=np.array([n.goal for n in nd]),
            tries=np.array([n.tries for n in nd]), D=np.array([n.D for n in nd]),
            edges=np.array(self.edges, dtype=float).reshape(-1, 3),
            e_tries=np.array(self.e_tries, int), e_succ=np.array(self.e_succ, int),
            sZ=np.array(self.sZ).reshape(len(self.sZ), -1 if self.sZ else (len(nd[0].z) if nd else ZDIM)),
            sM=np.array(self.sM, int), sT=np.array(self.sT, int), sR=np.array(self.sR, float),
        )

    @staticmethod
    def load(path: str | Path) -> "Archive":
        a = Archive()
        with np.load(path, allow_pickle=False) as f:
            # NPZ members are decompressed on every lookup. Read each column once.
            columns = [f[key] for key in ("z", "mode", "state", "ctrl", "B_pose", "op", "parent", "goal", "tries", "D")]
            types = tuple(map(str, f["types"])) if "types" in f else LEGACY_TYPES
            if types != tuple(TYPES):
                poses = columns[4]
                for index in (3, 4):
                    poses[:, index] = [TYPES.index(types[int(i)]) for i in poses[:, index]]
            a.nodes = [Node(z, int(mode), state, ctrl, pose, str(op), int(parent), bool(goal), int(tries), float(d))
                       for z, mode, state, ctrl, pose, op, parent, goal, tries, d in zip(*columns)]
            a.edges = [(int(i), int(j), float(c)) for i, j, c in f["edges"]]
            a.e_tries = [int(x) for x in f["e_tries"]] if "e_tries" in f else [1] * len(a.edges)
            a.e_succ = [int(x) for x in f["e_succ"]] if "e_succ" in f else [1] * len(a.edges)
            a.sZ, a.sM = list(f["sZ"]), [int(x) for x in f["sM"]]
            a.sT, a.sR = [int(x) for x in f["sT"]], [float(x) for x in f["sR"]]
        a.dijkstra()
        return a
