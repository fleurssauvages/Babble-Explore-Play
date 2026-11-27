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
    # Field samples from successful trajectories: (z, mode, target node, remaining seconds, source node: -1 unknown)
    sZ: list[np.ndarray] = field(default_factory=list)
    sM: list[int] = field(default_factory=list)
    sT: list[int] = field(default_factory=list)
    sR: list[float] = field(default_factory=list)
    sS: list[int] = field(default_factory=list)

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
            self.sS.append(i)

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

    # ----------------------------------------------------------- size budget
    def prune_to(self, budget: int, distance, goal_share: float = 0.33) -> int:
        """Keep at most ``budget`` nodes (0: no budget): the graph keeps a constant size, renewing itself where it is
        dense.  Only nodes no shortest path runs through may go (no node's successor toward the goal, not a goal
        seed).  Goal nodes beyond the share exploration keeps them at (goal_share of the others, so at most
        budget x goal_share / (1 + goal_share)) go first, then the others: first those that cannot reach the goal,
        then the most redundant, i.e. nearest (``distance(z, Z, mode)``, the schema's metric) to another node of
        the same mode with no higher cost-to-go.  Their edges and trajectory samples go with them.  Returns the
        number of nodes removed."""
        removed = 0
        goal_cap = int(budget * goal_share / (1.0 + goal_share))
        while budget and len(self.nodes) > budget:
            D = self.dijkstra()
            succ = np.full(len(self.nodes), -1)
            best = np.full(len(self.nodes), np.inf)
            for k, (i, j, _) in enumerate(self.edges):
                c = self.edge_cost(k) + D[j]
                if c < best[i]:
                    best[i], succ[i] = c, j
            goal = np.array([nd.goal for nd in self.nodes])
            protected = np.zeros(len(self.nodes), bool)
            protected[succ[(succ >= 0) & np.isfinite(D)]] = True
            protected |= goal & np.array([nd.parent < 0 for nd in self.nodes])
            Z = np.array([nd.z for nd in self.nodes]).reshape(len(self.nodes), -1)
            M = np.array([nd.mode for nd in self.nodes])

            def ranked(cand):
                """Candidates, most removable first (unreachable, then most redundant); never a node alone in its
                mode at its cost-to-go or below."""
                red = np.full(len(cand), -np.inf)
                for n, k in enumerate(cand):
                    if not np.isfinite(D[k]):
                        continue
                    other = np.flatnonzero((M == M[k]) & (D <= D[k]) & (goal == goal[k]))
                    other = other[other != k]
                    red[n] = float(distance(Z[k], Z[other], int(M[k])).min()) if len(other) else np.inf
                o = np.argsort(red, kind="stable")
                return cand[o][red[o] < np.inf]

            excess = len(self.nodes) - budget
            n_goal_drop = min(excess, max(0, int(goal.sum()) - goal_cap))
            drop = list(ranked(np.flatnonzero(~protected & goal))[:n_goal_drop])
            drop += list(ranked(np.flatnonzero(~protected & ~goal))[:excess - len(drop)])
            if not drop:
                break
            self._remove(set(int(k) for k in drop))
            removed += len(drop)
        if removed:
            self.dijkstra()
        return removed

    def _remove(self, gone: set) -> None:
        keep = np.array([k not in gone for k in range(len(self.nodes))])
        new = np.cumsum(keep) - 1                              # old index -> new index (where kept)
        remap = lambda k: int(new[k]) if k >= 0 and keep[k] else -1          # noqa: E731
        self.nodes = [nd for k, nd in enumerate(self.nodes) if keep[k]]
        for nd in self.nodes:
            nd.parent = remap(nd.parent)                      # a removed parent: the node aims at its successor
        ek = [k for k, (i, j, _) in enumerate(self.edges) if keep[i] and keep[j]]
        self.edges = [(remap(self.edges[k][0]), remap(self.edges[k][1]), self.edges[k][2]) for k in ek]
        self.e_tries = [self.e_tries[k] for k in ek]
        self.e_succ = [self.e_succ[k] for k in ek]
        src = self.sS if len(self.sS) == len(self.sT) else [-1] * len(self.sT)
        sk = [k for k, (t, a) in enumerate(zip(self.sT, src)) if keep[t] and (a < 0 or keep[a])]
        self.sZ = [self.sZ[k] for k in sk]
        self.sM = [self.sM[k] for k in sk]
        self.sR = [self.sR[k] for k in sk]
        self.sT = [remap(self.sT[k]) for k in sk]
        self.sS = [remap(src[k]) if src[k] >= 0 else -1 for k in sk]

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
            sS=np.array(self.sS if len(self.sS) == len(self.sT) else [-1] * len(self.sT), int),
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
            a.sS = [int(x) for x in f["sS"]] if "sS" in f else [-1] * len(a.sT)
        a.dijkstra()
        return a
