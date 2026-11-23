"""Cost-to-go field Phi over discovered contact modes, compacted.

    Phi_m(z) = softmin_tau_i [ L_i + kappa_m * || z - z_i ||_{W_m, 1} ]     (non-goal modes)
    Phi_goal = || pXY - centre(target relation) ||_{W_goal, 1}             (goal mode)

* labels L_i: seconds to the goal of every validated node and trajectory sample,
* metric W_m: the exploration metric of mode m (prim.metric_of: fixed object-unit weights rescaled by the
  demonstrated rates' relative factors, clipped, see learned_metric),
* kappa_m: low quantile of the demonstrated descent slopes in mode m,
* samples marked transient (mode -1) by exploration are ignored,
* symmetric bodies: every sample also stands for its images under the bodies' yaw symmetries (X's position
  in Y's frame turned by Y's period, the pinch point in X's frame by X's), so that Phi does not prefer one
  of a body's equivalent sides because exploration happened to visit it (symmetry_images),
* compaction: a sample j is dropped when another kept sample i satisfies
  L_i + kappa_m ||z_j - z_i|| <= L_j + slack.  By the triangle inequality the
  hard-min envelope moves by at most ``slack`` anywhere.  The default slack is
  the soft-min's own resolution tau * ln(k), so compaction removes only what the
  field cannot resolve anyway.

Near the validated graph the executor tracks its nodes hop by hop (``Roadmap``, bep.motor.execute);
far from it Spike-MPPI descends Phi.  ``covered`` tells whether a state lies within reach of the graph
(used by play and the brain to keep calls where the primitive has experience); ``learned_metric``
turns the demonstrated rates into the per-mode distance used by exploration and the roadmap.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..config import EXEC, EXPLORE, FIELD, FIELD_LEARN
from ..world.relations import ZDIM


def _consecutive(archive):
    Z, M, T, R = (np.array(archive.sZ), np.array(archive.sM), np.array(archive.sT), np.array(archive.sR))
    if len(Z) < 2:
        return None
    dR = R[:-1] - R[1:]
    i = np.nonzero((T[1:] == T[:-1]) & (M[1:] == M[:-1]) & (M[:-1] >= 0) & (dR > 0) & (dR < 0.1))[0]
    return Z, M, dR, i


def mode_metrics(archive) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-mode metric weights 1 / max(rate, floor) from consecutive trajectory samples."""
    c = _consecutive(archive)
    if c is None or not len(c[3]):
        return {}, np.ones(ZDIM)
    Z, M, dR, i = c
    v = np.abs(Z[i + 1] - Z[i]) / dR[i, None]
    pooled = np.percentile(v, FIELD["rate_quantile"], axis=0)
    rates = {}
    for m in np.unique(M[i]):
        s = M[i] == m
        rates[int(m)] = np.percentile(v[s], FIELD["rate_quantile"], axis=0) if s.sum() >= FIELD["min_pairs"] else pooled
    # floor: a coordinate that does not move in one mode can still be changed by passing through
    # the mode that changes it fastest, so its rate is floored at that mode's rate.
    floor = np.max(np.array(list(rates.values())), axis=0)
    floor = np.where(floor > 0, floor, 1.0)
    W = {m: 1.0 / np.maximum(r, floor) for m, r in rates.items()}
    return W, 1.0 / np.maximum(pooled, floor)


def descent_slopes(archive, W: dict, W_pooled) -> dict[int, np.ndarray]:
    c = _consecutive(archive)
    if c is None:
        return {}
    Z, M, dR, i = c
    out = {}
    for m in np.unique(M[i]):
        s = i[M[i] == m]
        d = np.abs((Z[s + 1] - Z[s]) * W.get(int(m), W_pooled)).sum(axis=1)
        out[int(m)] = dR[s] / np.maximum(d, 1e-6)
    return out


def _turns(periods) -> int:
    """Number of yaw rotations that are symmetries of every listed shape period (0: any angle, taken by
    quarter turns; 2 pi: none)."""
    p = max([np.pi / 2 if q <= 0 else float(q) for q in periods], default=2 * np.pi)
    return max(1, int(round(2 * np.pi / p)))


def symmetry_images(Z: np.ndarray, prim):
    """The samples' images under the yaw symmetries of the bodies (index of the source sample per image):
    Y turned by a multiple of its period moves X's position in Y's frame (z[0:2]) and the code of their
    relative yaw (z[3:5], period of X); X turned by its period moves the pinch point in X's frame (z[5:7]).
    Periods are those of every pair of kinds the primitive learned on."""
    from ..world.objects import kind_shape
    from ..world.relations import CC
    pairs = getattr(prim, "training_pairs", (prim.kinds,))
    nx = _turns([kind_shape(a).sym for a, _ in pairs])
    ny = _turns([kind_shape(b).sym for _, b in pairs]) if prim.kind == CC else 1
    px = kind_shape(pairs[0][0]).sym
    kx = 2 * np.pi / px if px > 0 else 0.0                       # sym_code frequency of the relative yaw
    out, src = [], []
    for i in range(ny):
        dy = 2 * np.pi * i / ny
        for j in range(nx):
            dx = 2 * np.pi * j / nx
            z = Z.copy()
            c, s_ = np.cos(dy), np.sin(dy)
            z[:, 0], z[:, 1] = c * Z[:, 0] + s_ * Z[:, 1], -s_ * Z[:, 0] + c * Z[:, 1]
            c3, s3 = np.cos(-kx * dy), np.sin(-kx * dy)          # relative yaw - dy (+ dx: a whole period)
            z[:, 3], z[:, 4] = c3 * Z[:, 3] - s3 * Z[:, 4], s3 * Z[:, 3] + c3 * Z[:, 4]
            c, s_ = np.cos(dx), np.sin(dx)
            z[:, 5], z[:, 6] = c * Z[:, 5] + s_ * Z[:, 6], -s_ * Z[:, 5] + c * Z[:, 6]
            out.append(z)
            src.append(np.arange(len(Z)))
    return np.concatenate(out), np.concatenate(src)


def prune(Zw: np.ndarray, L: np.ndarray, kappa: float, slack: float) -> np.ndarray:
    """Greedy envelope-dominance pruning (increasing label order); returns kept indices."""
    kept = np.empty(len(L), int)
    kz = np.empty_like(Zw)
    kl = np.empty(len(L))
    n = 0
    for j in np.argsort(L, kind="stable"):
        if n and (kl[:n] + kappa * np.abs(kz[:n] - Zw[j]).sum(1)).min() <= L[j] + slack:
            continue
        kept[n], kz[n], kl[n] = j, Zw[j], L[j]
        n += 1
    return kept[:n]


class Roadmap:
    """The validated graph itself, for execution by hops ("return" in Go-Explore): nodes with a finite
    cost-to-go and, for each, the successor on its shortest path to the goal.  Every edge was validated by
    Spike-MPPI tracking the successor node from the node's state until it reached any node of lower
    cost-to-go, so executing the same hops reproduces validated behaviour."""

    def __init__(self, archive, prim) -> None:
        from .primitive import ctx_pair

        D = np.array([nd.D for nd in archive.nodes])
        keep = np.nonzero(np.isfinite(D))[0]
        idx = {int(k): i for i, k in enumerate(keep)}
        self.prim, self._archive, self._keep, self._cfg = prim, archive, keep, {}
        self.Z = np.array([archive.nodes[k].z for k in keep]).reshape(len(keep), -1)
        self.M = np.array([int(archive.nodes[k].mode) for k in keep], int)
        self.D = D[keep]
        self.goal = np.array([bool(archive.nodes[k].goal) for k in keep])
        by_pair: dict[tuple, list] = {}
        for i, k in enumerate(keep):
            pair = ctx_pair(archive.nodes[k].B_pose)
            key = pair if prim.kind == "object-object" else pair[:1]
            by_pair.setdefault(key, []).append(i)
        self._all = np.arange(len(keep))
        self._by_pair = {p: np.asarray(indices, int) for p, indices in by_pair.items()}
        self.succ = np.full(len(keep), -1)
        best = np.full(len(keep), np.inf)
        for k, (i, j, _) in enumerate(archive.edges):
            i, j, c = int(i), int(j), archive.edge_cost(k)        # expected time (reliability included)
            if i in idx and j in idx and c + D[j] < best[idx[i]]:
                best[idx[i]], self.succ[idx[i]] = c + D[j], idx[j]
        lens = [float(prim.distance(self.Z[i], self.Z[self.succ[i]][None], self.M[self.succ[i]])[0])
                for i in range(len(keep)) if self.succ[i] >= 0]
        self.hop = float(np.median(lens)) if lens else 1.0          # typical validated hop (metric units)
        self.edge_s = float(np.median([c for _, _, c in archive.edges])) if archive.edges else 1.0

    def _cfg_of(self, a: int):
        """Configuration (hand pose relative to X, X relative to Y, grip) of archive node a, read from its
        archived state in the learning's body (the Hand-of-God motor's targets)."""
        if a not in self._cfg:
            self._cfg[a] = self.prim.node_cfg(self._archive.nodes[a])
        return self._cfg[a]

    def cfg(self, i: int):
        """Configuration of roadmap node i."""
        return self._cfg_of(int(self._keep[i]))

    def aim(self, i: int):
        """(z, mode) to aim at to leave node i along its validated edge (the target its edge was validated
        with, see aim_cfg)."""
        p = int(self._archive.nodes[int(self._keep[i])].parent)
        if p >= 0:
            nd = self._archive.nodes[p]
            return np.asarray(nd.z, float), (self.prim.goal_mode if nd.goal else int(nd.mode))
        j = i if self.goal[i] or self.succ[i] < 0 else int(self.succ[i])
        return self.Z[j], (self.prim.goal_mode if self.goal[j] else int(self.M[j]))

    def aim_cfg(self, i: int):
        """What to aim at to leave node i along its validated edge: the node it was proposed from, which was
        the target when its edge was validated (the edge ends at whichever solved node the motion reached);
        for a node without one (goal seeds), its successor."""
        p = int(self._archive.nodes[int(self._keep[i])].parent)
        if p >= 0:
            return self._cfg_of(p)
        j = i if self.goal[i] or self.succ[i] < 0 else int(self.succ[i])
        return self.cfg(j)

    def compatible(self, pair=None) -> np.ndarray:
        if pair is None:
            return self._all
        key = pair if self.prim.kind == "object-object" else pair[:1]
        return self._by_pair.get(key, self._all)

    def _same_mode(self, modes, mode, carrying):
        if not carrying:
            return modes == mode
        mask = self.prim.carry_mask
        return ((modes & mask) != 0) & ((modes & ~mask) == (mode & ~mask))

    def entry(self, z, mode: int, exclude=(), *, pair=None, carrying: bool = False) -> int:
        """The node to start from: low cost-to-go and near the current state (same mode when possible)."""
        # enter at a node with a validated path ahead (never straight at a goal node from outside the goal
        # mode: the straight line to it is what the graph's edges avoid, e.g. through the body)
        indices = self.compatible(pair)
        entry_ok = ~self.goal[indices] | (mode == self.prim.goal_mode)
        sel = indices[self._same_mode(self.M[indices], mode, carrying) & entry_ok]
        if not len(sel) and carrying:
            sel = indices[entry_ok & ((self.M[indices] & self.prim.carry_mask) != 0)]
        if not len(sel):
            sel = indices[entry_ok] if entry_ok.any() else indices
        if exclude:
            rest = np.array([i for i in sel if i not in exclude], int)
            sel = rest if len(rest) else sel           # everything failed: retry the best anyway
        d = self.prim.distance(z, self.Z[sel], mode, carrying=carrying)
        return int(sel[np.argmin(self.D[sel] + self.edge_s * d / self.hop)])

    def near(self, z, mode: int) -> float:
        """Distance (in typical hops) from the state to the nearest node of its mode."""
        sel = self.M == mode
        if not sel.any():
            return np.inf
        return float(self.prim.distance(z, self.Z[sel], mode).min()) / self.hop

    def reached(self, z, mode: int, D_max: float, *, pair=None, carrying: bool = False):
        """A node of cost-to-go <= D_max in the current mode within the exploration's eps of the state."""
        indices = self.compatible(pair)
        sel = indices[self._same_mode(self.M[indices], mode, carrying) & (self.D[indices] <= D_max + 1e-9)]
        if not len(sel):
            return None
        d = self.prim.distance(z, self.Z[sel], mode, carrying=carrying)
        ok = d < self.prim.eps[mode]
        return int(sel[ok][np.argmin(self.D[sel][ok])]) if ok.any() else None


class ModeField:
    def __init__(self, archive, prim, *, slack: float | None = None, learning: bool = False) -> None:
        """learning: the field exploration builds for itself (its stop rule and executor tests), with the
        settings of FIELD_LEARN; otherwise the field the tasks' executor descends (FIELD)."""
        F = {**FIELD, **FIELD_LEARN} if learning else FIELD
        self.prim = prim
        self.roadmap = Roadmap(archive, prim)
        self.goal_mode = int(prim.goal_mode)
        self.tau, self.k = float(F["tau"]), int(F["k"])
        self.slack = self.tau * np.log(self.k) if slack is None else float(slack)
        Z, M, L = archive.samples()
        ok = np.isfinite(L) & (M >= 0)
        Z, M, L = Z[ok], M[ok], L[ok]
        self.cap = float(L.max() + F["cap_margin"]) if len(L) else 30.0
        # the demonstrated rates give the exploration metric (learned_metric); Phi measures distances with that
        # metric (the fixed object-unit weights x the rates' relative factors, clipped), so that position,
        # orientation and opening count in the same units in every mode
        self.W_rate, W_rate_pooled = mode_metrics(archive)
        if F["metric"] == "rate":                     # 1 / demonstrated rate of each coordinate
            self.W, self.W_pooled = dict(self.W_rate), W_rate_pooled
        else:                                         # the exploration metric
            from .primitive import W as W_FIXED
            modes = set(np.unique(M).tolist()) | set(self.W_rate) | {self.goal_mode}
            self.W = {int(m): np.asarray(prim.metric_of(m), float) for m in modes}
            self.W_pooled = np.asarray(W_FIXED, float)
        slopes = descent_slopes(archive, self.W, self.W_pooled)
        Z, src = symmetry_images(Z, prim)                 # after the slopes (measured along the trajectories)
        M, L = M[src], L[src]
        self.kappas: dict[int, float] = {}
        self.trees: dict[int, tuple[cKDTree, float]] = {}
        self.n_full, self.n_kept = 0, 0
        self._kept: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for m in np.unique(M):
            m = int(m)
            sel = M == m
            if m == self.goal_mode or sel.sum() < F["min_samples"]:
                continue
            Wm = self.W.setdefault(m, self.W_pooled)
            s = slopes.get(m, np.array([]))
            self.kappas[m] = max(float(np.quantile(s, F["kappa_quantile"])), 1e-3) if len(s) >= 20 \
                else F["kappa_default"]
            Zw, Lm = Z[sel] * Wm, L[sel]
            keep = prune(Zw, Lm, self.kappas[m], self.slack) if F["compact"] else np.arange(len(Lm))
            self.n_full += int(sel.sum())
            self.n_kept += len(keep)
            Lmin = float(Lm.min())
            self.trees[m] = (cKDTree(np.hstack([Zw[keep], ((Lm[keep] - Lmin) / self.kappas[m])[:, None]])), Lmin)
            self._kept[m] = (Z[sel][keep], Lm[keep])
        self.W_goal = self.W.get(self.goal_mode, self.W_pooled)
        # modes with too few samples of their own use the envelope pooled over every non-goal sample
        pool = M != self.goal_mode
        self.pooled = None
        if pool.sum() >= F["min_samples"]:
            s_all = np.concatenate([v for k, v in slopes.items() if k != self.goal_mode]) if slopes else np.array([])
            kp = max(float(np.quantile(s_all, F["kappa_quantile"])), 1e-3) if len(s_all) >= 20 else F["kappa_default"]
            Zw, Lp = Z[pool] * self.W_pooled, L[pool]
            keep = prune(Zw, Lp, kp, self.slack) if F["compact"] else np.arange(len(Lp))
            Lmin = float(Lp.min())
            self.pooled = (cKDTree(np.hstack([Zw[keep], ((Lp[keep] - Lmin) / kp)[:, None]])), Lmin, kp)

    def explored(self, mode: np.ndarray) -> np.ndarray:
        """Modes with samples of their own (or the goal mode)."""
        return np.isin(np.asarray(mode), list(self.trees) + [self.goal_mode])

    def samples(self, mode: int):
        """The kept (z, label) samples of a mode (for visualisation)."""
        return self._kept.get(mode, (np.empty((0, ZDIM)), np.empty(0)))

    def value(self, z: np.ndarray, mode: np.ndarray, rel: dict | None = None) -> np.ndarray:
        shape = z.shape[:-1]
        zf = z.reshape(-1, z.shape[-1])
        mf = np.asarray(mode).reshape(-1)
        out = np.full(zf.shape[0], self.cap)
        for m, (tree, Lmin) in self.trees.items():
            sel = np.nonzero(mf == m)[0]
            if not len(sel):
                continue
            k = min(self.k, tree.n)
            q = np.hstack([zf[sel] * self.W[m], np.zeros((len(sel), 1))])
            dist, _ = tree.query(q, k=k, p=1)
            v = Lmin + self.kappas[m] * dist.reshape(len(sel), k)
            vmin = v.min(axis=1)
            out[sel] = vmin - self.tau * np.log(np.exp(-(v - vmin[:, None]) / self.tau).sum(axis=1))
        if self.pooled is not None:
            sel = np.nonzero(~np.isin(mf, list(self.trees) + [self.goal_mode]))[0]
            if len(sel):
                tree, Lmin, kp = self.pooled
                k = min(self.k, tree.n)
                q = np.hstack([zf[sel] * self.W_pooled, np.zeros((len(sel), 1))])
                dist, _ = tree.query(q, k=k, p=1)
                v = Lmin + kp * dist.reshape(len(sel), k)
                vmin = v.min(axis=1)
                out[sel] = vmin - self.tau * np.log(np.exp(-(v - vmin[:, None]) / self.tau).sum(axis=1)) \
                    + FIELD["unexplored_s"]
        sel = np.nonzero(mf == self.goal_mode)[0]
        if len(sel):
            if self.prim.goal_requires_lift:
                out[sel] = 0.0
            else:
                gs = self.prim.goal_slice
                res = np.abs(zf[sel][:, gs] - self.prim.goal_center) * self.W_goal[gs]
                out[sel] = res.sum(axis=1)
        return np.minimum(out, self.cap).reshape(shape)


def covered(prim, field, z, mode, *, pair=None) -> bool:
    """The primitive has validated experience from such a state: it lies within hop_join typical hops of its
    graph in the same mode, apart from what the primitive is meant to change (its goal dimensions, and for
    an object-object relation X's yaw relative to Y) where it controls X (e.g. X held for a placement)."""
    nav = field.roadmap
    indices = nav.compatible(pair)
    sel = indices[nav.M[indices] == mode]
    if not len(sel):
        return False
    keep = np.ones(len(z), bool)
    # the goal quantities can only be changed where the primitive controls X: always for a hand-object
    # primitive (the hand moves freely), otherwise only in modes where a hand relation on X holds (X held)
    names = prim.R.bit_names()
    hand_bits = [b for b, n in enumerate(names) if n in {r.name for r in prim.R.hc}]
    if prim.kind == "hand-object" or any(mode >> b & 1 for b in hand_bits):
        keep[prim.goal_slice] = False
        if prim.kind == "object-object":
            keep[3:5] = False
    return float(prim.distance(z * keep, nav.Z[sel] * keep, mode).min()) / nav.hop <= EXEC["hop_join"]


def learned_metric(field: ModeField, prim) -> dict:
    """Per-mode distance weights: the fixed weights rescaled, coordinate by coordinate, by how slowly the
    coordinate moves in that mode (the field's 1 / demonstrated rate) relative to its geometric mean over
    the coordinates; the factor is clipped to EXPLORE['metric_clip'] so that eps keeps its meaning."""
    from .primitive import W
    lo, hi = EXPLORE["metric_clip"]
    out = {}
    for m, w in field.W_rate.items():
        w = np.asarray(w, float)
        f = w / np.exp(np.mean(np.log(np.maximum(w, 1e-9))))
        out[int(m)] = W * np.clip(f, lo, hi)
    return out
