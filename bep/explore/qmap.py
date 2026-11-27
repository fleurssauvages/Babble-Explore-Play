"""Sleep's cognitive map: one learned temporal distance d(s, g) for every schema, and Phi from it.

    d(s, g) = || a(s) - a(g) ||_2 + max_j relu(b_j(g) - b_j(s))           (a quasimetric: the MRN head)

where (a, b) = f(feat(z)) is a small MLP (2 x QMAP['hidden']), one per member of an ensemble of
QMAP['ensemble'] trained together.  The symmetric part measures how far two states are, the asymmetric part
what must be gained to go from s to g (lifting is not dropping); both satisfy the triangle inequality, so d
does too.

Inputs are the relational state z only, no mode bitmask: feat(z) = z in the fixed object-unit metric plus two
signed distances read from z itself (the pinch point to X's box, X's box to Y's), so that approaching a body
and touching it are continuous in the input.

Training (at sleep, warm-started from the previous epoch's weights), on every schema's archive:
* labels: the schema's Dijkstra time-to-goal of every node and trajectory sample, L_i ~ Phi_k(z_i) =
  softmin over the schema's goal states g of d(z_i, g)  (Huber);
* trajectory upper bounds: d(z_t, z_t') <= t' - t along every validated motion;
* triangle lower bounds: d(z_i, z_j) >= L_i - L_j for random pairs of the same schema (Phi_k(z_i) <=
  d(z_i, z_j) + Phi_k(z_j));
* the bodies' yaw symmetries: every sample also stands for its symmetric images (bep.explore.field).

Trust (arbitration by certainty): at every sleep the map and each schema's own field are scored on held-out
samples (rank correlation with the time-to-goal labels); the map's Phi is used only for the schemas where it
is the better predictor (``QMap.trusted``), the others keep their field.

``NetField`` replaces a trusted schema's field: same interface as bep.explore.field.ModeField (the validated graph
is kept for hops and coverage), with Phi_k from the map outside the goal mode (the goal mode keeps the
relation residual).  Native rollouts get the network through bep.motor.native (``net`` in the spec).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .field import ModeField, symmetry_images, field_for
from .primitive import W as W_FIXED
from ..world.relations import ZDIM
from ..config import QMAP

FEAT = ZDIM + 2


def feat(z: np.ndarray) -> np.ndarray:
    """(..., FEAT [+ mode bits]) map inputs: z in the fixed metric, + signed distance of the pinch point to X's box
    and of X's box to Y's (object units: max coordinate - 1); columns of z beyond ZDIM (optional extra inputs)
    are passed as they are."""
    z = np.asarray(z, float)
    s_hand = np.abs(z[..., 5:8]).max(-1, keepdims=True) - 1.0
    s_xy = np.abs(z[..., 0:3]).max(-1, keepdims=True) - 1.0
    return np.concatenate([z[..., :ZDIM] * W_FIXED, s_hand, s_xy, z[..., ZDIM:]], axis=-1)


def with_bits(z: np.ndarray, mode, nb: int) -> np.ndarray:
    """z with the mode's first ``nb`` bits appended (nothing when nb = 0)."""
    z = np.asarray(z, float)
    if not nb:
        return z
    m = np.broadcast_to(np.asarray(mode, np.int64), z.shape[:-1])
    bits = ((m[..., None] >> np.arange(nb)) & 1).astype(float)
    return np.concatenate([z, np.where(m[..., None] < 0, 0.0, bits)], axis=-1)


# ------------------------------------------------------------------ the network (torch)
def _torch():
    import torch
    return torch


class Net:
    """E encoders FEAT -> h -> h -> 2k, as grouped weights (one batched matmul per layer)."""

    def __init__(self, E: int, h: int, k: int, seed: int = 0, nb: int = 0):
        torch = _torch()
        g = torch.Generator().manual_seed(seed)

        def layer(i, o):
            w = torch.randn(E, i, o, generator=g) * (2.0 / i) ** 0.5
            return w.requires_grad_(), torch.zeros(E, 1, o, requires_grad=True)
        self.E, self.h, self.k, self.nb = E, h, k, nb
        self.W1, self.b1 = layer(FEAT + nb, h)
        self.W2, self.b2 = layer(h, h)
        self.W3, self.b3 = layer(h, 2 * k)
        self.mu = torch.zeros(FEAT + nb)
        self.sd = torch.ones(FEAT + nb)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def embed(self, x):
        """x: (N, FEAT) or (E, N, FEAT) -> (E, N, 2k)."""
        torch = _torch()
        x = (x - self.mu) / self.sd
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.E, -1, -1)
        h = torch.relu(torch.bmm(x, self.W1) + self.b1)
        h = torch.relu(torch.bmm(h, self.W2) + self.b2)
        return torch.bmm(h, self.W3) + self.b3

    def dist(self, es, eg):
        """Quasimetric between embeddings es (E, N, 2k) and eg (E, N, 2k) -> (E, N)."""
        torch = _torch()
        k = self.k
        sym = torch.sqrt(((es[..., :k] - eg[..., :k]) ** 2).sum(-1) + 1e-8)
        asym = torch.relu(eg[..., k:] - es[..., k:]).max(-1).values
        return sym + asym

    def state(self) -> dict:
        return {k: getattr(self, k).detach().numpy() for k in ("W1", "b1", "W2", "b2", "W3", "b3", "mu", "sd")}

    def load(self, st: dict) -> None:
        torch = _torch()
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, k, torch.tensor(st[k], dtype=torch.float32).requires_grad_())
        self.mu, self.sd = torch.tensor(st["mu"], dtype=torch.float32), torch.tensor(st["sd"], dtype=torch.float32)


class FrozenNet:
    """A trained network's weights as numpy arrays only (Net's ``state`` without torch): what evaluating the map
    needs (np_embed, NetField, the native code).  Loading the map this way keeps torch (~180 MB per process) out
    of the pool's workers."""

    def __init__(self, st: dict, E: int, h: int, k: int, nb: int = 0):
        self.st = {key: np.asarray(v, np.float32) for key, v in st.items()}
        self.E, self.h, self.k, self.nb = E, h, k, nb

    def state(self) -> dict:
        return self.st


# numpy forward (fields and checks; mirrors the native code)
def np_embed(st: dict, x: np.ndarray) -> np.ndarray:
    x = (x - st["mu"]) / st["sd"]
    h = np.maximum(np.einsum("ni,eio->eno", x, st["W1"]) + st["b1"], 0.0)
    h = np.maximum(np.einsum("eni,eio->eno", h, st["W2"]) + st["b2"], 0.0)
    return np.einsum("eni,eio->eno", h, st["W3"]) + st["b3"]


def np_phi(st: dict, goals: np.ndarray, z: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std over the ensemble) of softmin_g d(z, g); goals: (E, G, 2k) embeddings."""
    k = st["W3"].shape[-1] // 2
    e = np_embed(st, feat(z))                                         # (E, N, 2k)
    sym = np.sqrt(((e[:, :, None, :k] - goals[:, None, :, :k]) ** 2).sum(-1) + 1e-8)
    asym = np.maximum(goals[:, None, :, k:] - e[:, :, None, k:], 0.0).max(-1)
    d = sym + asym                                                    # (E, N, G)
    dmin = d.min(-1, keepdims=True)
    v = dmin[..., 0] - tau * np.log(np.exp(-(d - dmin) / tau).sum(-1))
    return v.mean(0), v.std(0)


# ------------------------------------------------------------------ data from the archives
def _farthest(Z: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point subsample (indices) of goal states."""
    if len(Z) <= n:
        return np.arange(len(Z))
    X = feat(Z)
    pick = [0]
    d = np.abs(X - X[0]).sum(1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        pick.append(j)
        d = np.minimum(d, np.abs(X - X[j]).sum(1))
    return np.array(pick)


def schema_data(archive, prim, nb: int = 0) -> dict | None:
    """Labelled samples, goal states and trajectory pairs of one schema's archive (with symmetric images; with
    ``nb`` > 0, every state also carries its mode's bits)."""
    Z, M, L = archive.samples()
    ok = np.isfinite(L) & (M >= 0)
    if ok.sum() < 20:
        return None
    Zi, src = symmetry_images(Z[ok], prim)
    Zi = with_bits(Zi, M[ok][src], nb)
    Li = L[ok][src]
    if len(Li) > QMAP["max_samples"]:                     # long runs: a random subset per fit
        keep = np.random.default_rng(len(Li)).choice(len(Li), QMAP["max_samples"], replace=False)
        Zi, Li = Zi[keep], Li[keep]
    goal_nodes = [nd for nd in archive.nodes if nd.goal]
    if not goal_nodes:
        return None
    Gi, gsrc = symmetry_images(np.array([nd.z for nd in goal_nodes]).reshape(-1, ZDIM), prim)
    Gi = with_bits(Gi, np.array([nd.mode for nd in goal_nodes])[gsrc], nb)
    G = Gi[_farthest(Gi, QMAP["goals"])]
    # consecutive trajectory samples of the same validated motion: d(z_t, z_t+1) <= dt
    sZ, sT, sR = np.array(archive.sZ), np.array(archive.sT), np.array(archive.sR)
    sZ = with_bits(sZ.reshape(len(sZ), -1), np.array(archive.sM), nb) if len(sZ) else sZ
    pairs = np.zeros((0, 2, ZDIM + nb)), np.zeros(0)
    if len(sZ) > 1:
        dR = sR[:-1] - sR[1:]
        i = np.flatnonzero((sT[1:] == sT[:-1]) & (dR > 0) & (dR < 0.5))
        pairs = np.stack([sZ[i], sZ[i + 1]], axis=1), dR[i]
    return {"Z": Zi, "L": Li, "G": G, "P": pairs[0], "dP": pairs[1], "cap": float(Li.max() + 3.0)}


# ------------------------------------------------------------------ training
class QMap:
    def __init__(self, path: Path | None = None, nb: int = 0, *, inference: bool = False):
        """``inference``: load the trained map for evaluation only (FrozenNet: numpy, no torch; it cannot be fit)."""
        self.nb = nb                                    # extra input columns (0: z only)
        self.inference = inference
        self.net = None if inference else Net(QMAP["ensemble"], QMAP["hidden"], QMAP["embed"], nb=nb)
        self.goals: dict[str, np.ndarray] = {}         # schema -> goal z (G, ZDIM)
        self.caps: dict[str, float] = {}
        self.trusted: set = set()                       # schemas whose Phi comes from the map (it beat the field)
        self.calibration: dict = {}
        self.fitted = False
        if path is not None and Path(path).exists():
            self.load(path)

    def fit(self, data: dict[str, dict], steps: int | None = None) -> dict:
        torch = _torch()
        threads = torch.get_num_threads()
        torch.set_num_threads(QMAP["threads"])             # sleep: the pool's workers are idle
        try:
            return self._fit(data, steps)
        finally:
            torch.set_num_threads(threads)

    def _fit(self, data: dict[str, dict], steps: int | None = None) -> dict:
        torch = _torch()
        t0 = time.time()
        names = [n for n, d in data.items() if d is not None]
        if not names:
            return {"fitted": False}
        rng0 = np.random.default_rng(1)
        held = {}
        for n in names:                                   # held-out samples to judge the map against the field
            d = data[n]
            mask = rng0.random(len(d["L"])) < QMAP["holdout"]
            held[n] = (d["Z"][mask], d["L"][mask])
            data[n] = dict(d, Z=d["Z"][~mask], L=d["L"][~mask])
        allZ = np.concatenate([data[n]["Z"] for n in names])
        X = feat(allZ)
        if not self.fitted:
            self.net.mu = torch.tensor(X.mean(0), dtype=torch.float32)
            self.net.sd = torch.tensor(X.std(0) + 1e-3, dtype=torch.float32)
        new_schema = any(n not in self.goals for n in names)     # a schema the map has not learned yet: a full fit
        steps = steps or (QMAP["warm_steps"] if self.fitted and not new_schema else QMAP["steps"])
        opt = torch.optim.Adam(self.net.params(), lr=QMAP["lr"])
        T = {n: {"X": torch.tensor(feat(data[n]["Z"]), dtype=torch.float32),
                 "L": torch.tensor(data[n]["L"], dtype=torch.float32),
                 "G": torch.tensor(feat(data[n]["G"]), dtype=torch.float32),
                 "P0": torch.tensor(feat(data[n]["P"][:, 0]) if len(data[n]["P"]) else np.zeros((0, FEAT + self.nb)),
                                    dtype=torch.float32),
                 "P1": torch.tensor(feat(data[n]["P"][:, 1]) if len(data[n]["P"]) else np.zeros((0, FEAT + self.nb)),
                                    dtype=torch.float32),
                 "dP": torch.tensor(data[n]["dP"], dtype=torch.float32)} for n in names}
        rng = np.random.default_rng(0)
        B = max(64, QMAP["batch"] // len(names))
        batch = {n: B for n in names}
        tau = QMAP["tau"]
        E = self.net.E
        hist = []
        for step in range(steps):
            loss = 0.0
            for n in names:
                t = T[n]
                B = batch[n]
                idx = torch.tensor(rng.integers(len(t["L"]), size=B))
                es = self.net.embed(t["X"][idx])                                  # (E, B, 2k)
                eg = self.net.embed(t["G"])                                       # (E, G, 2k)
                Gn = eg.shape[1]
                d = self.net.dist(es.unsqueeze(2).expand(-1, -1, Gn, -1).reshape(E, B * Gn, -1),
                                  eg.unsqueeze(1).expand(-1, B, -1, -1).reshape(E, B * Gn, -1)).reshape(E, B, Gn)
                phi = -tau * torch.logsumexp(-d / tau, dim=-1)                    # (E, B)
                loss = loss + torch.nn.functional.huber_loss(phi, t["L"][idx].expand(E, -1), delta=1.0)
                # triangle lower bound on random pairs of the same schema
                j = torch.tensor(rng.integers(len(t["L"]), size=B))
                ej = self.net.embed(t["X"][j])
                dij = self.net.dist(es, ej)
                loss = loss + QMAP["spread_w"] * torch.relu(t["L"][idx] - t["L"][j] - dij).pow(2).mean()
                if len(t["dP"]):
                    p = torch.tensor(rng.integers(len(t["dP"]), size=B))
                    dp = self.net.dist(self.net.embed(t["P0"][p]), self.net.embed(t["P1"][p]))
                    loss = loss + QMAP["traj_w"] * torch.relu(dp - t["dP"][p]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == steps - 1:
                hist.append(round(float(loss.detach()) / len(names), 4))
        self.fitted = True
        for n in names:
            self.goals[n] = data[n]["G"]
            self.caps[n] = data[n]["cap"]
        # calibration on the held-out samples (per schema): rank correlation with the labels and mean error
        from scipy.stats import spearmanr
        cal = {}
        st = self.net.state()
        for n in names:
            Zh, Lh = held[n]
            if len(Lh) < 20:
                continue
            sel = rng.choice(len(Lh), size=min(2000, len(Lh)), replace=False)
            v, _ = np_phi(st, np_embed(st, feat(data[n]["G"])), Zh[sel], tau)
            cal[n] = {"spearman": round(float(spearmanr(v, Lh[sel])[0]), 3),
                      "mae_s": round(float(np.mean(np.abs(v - Lh[sel]))), 3)}
        self.calibration = cal
        return {"fitted": True, "steps": steps, "loss": hist, "held_out": cal, "fit_s": round(time.time() - t0, 1)}

    def goal_embed(self, name: str) -> np.ndarray:
        st = self.net.state()
        return np_embed(st, feat(self.goals[name]))

    def save(self, path: Path) -> None:
        st = self.net.state()
        np.savez_compressed(path, **st, **{f"goals__{n}": g for n, g in self.goals.items()},
                            caps=json.dumps(self.caps), trusted=json.dumps(sorted(self.trusted)), nb=self.nb)

    def load(self, path: Path) -> None:
        with np.load(path) as f:
            st = {k: f[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3", "mu", "sd")}
            self.goals = {k[len("goals__"):]: f[k] for k in f.files if k.startswith("goals__")}
            self.caps = json.loads(str(f["caps"]))
            self.trusted = set(json.loads(str(f["trusted"]))) if "trusted" in f.files else set(self.goals)
            self.nb = int(f["nb"]) if "nb" in f.files else 0
        if self.inference:
            self.net = FrozenNet(st, QMAP["ensemble"], QMAP["hidden"], QMAP["embed"], nb=self.nb)
        else:
            self.net = Net(QMAP["ensemble"], QMAP["hidden"], QMAP["embed"], nb=self.nb)
            self.net.load(st)
        self.fitted = True


# ------------------------------------------------------------------ the field
class NetField:
    """A schema's field with Phi from the map (ModeField's interface; the validated graph is the base field's)."""

    def __init__(self, base: ModeField, qmap: QMap, name: str):
        self.base, self.name, self.nb = base, name, qmap.nb
        self.st = qmap.net.state()
        self.goal_emb = qmap.goal_embed(name)
        self.tau = QMAP["tau"]
        self.cap = float(max(base.cap, qmap.caps.get(name, base.cap)))
        # the data's extent: a network extrapolates freely away from its samples, so outside the box of the field's
        # samples (per mode, weighted coordinates) Phi is the map at the state clipped into the box plus the field's
        # slope kappa x the clipped distance: it never decreases away from the data
        self.support: dict = {}
        self.support_pooled = None
        if QMAP["support_q"] > 0:
            for m, (tree, _) in base.trees.items():
                self.support[m] = self._box(tree, base.W[m], base.kappas[m])
            if base.pooled is not None:
                tree, _, kp = base.pooled
                self.support_pooled = self._box(tree, base.W_pooled, kp)

    def __getattr__(self, item):
        return getattr(self.base, item)

    @staticmethod
    def _box(tree, W, kappa) -> tuple:
        Zw = np.asarray(tree.data, float)[:, :ZDIM]
        q = QMAP["support_q"]
        return (np.asarray(W, float), np.quantile(Zw, q, axis=0), np.quantile(Zw, 1 - q, axis=0), float(kappa))

    def clip(self, z, mode) -> tuple[np.ndarray, np.ndarray]:
        """(z clipped into its mode's box, kappa x the weighted L1 distance it was moved)."""
        zc = np.array(z, float)
        pen = np.zeros(len(zc))
        known = list(self.support)
        groups = [(np.nonzero(mode == m)[0], v) for m, v in self.support.items()]
        if self.support_pooled is not None:
            groups.append((np.nonzero(~np.isin(mode, known))[0], self.support_pooled))
        for sel, (W, lo, hi, kappa) in groups:
            if not len(sel):
                continue
            zw = zc[sel, :ZDIM] * W
            cw = np.clip(zw, lo, hi)
            pen[sel] = kappa * np.abs(zw - cw).sum(1)
            pos = W > 0
            zc[np.ix_(sel, np.nonzero(pos)[0])] = cw[:, pos] / W[pos]
        return zc, pen

    def value(self, z, mode, rel=None):
        z = np.asarray(z, float)
        shape = z.shape[:-1]
        zf = z.reshape(-1, z.shape[-1])
        mf = np.broadcast_to(np.asarray(mode), shape).reshape(-1)
        zc, pen = self.clip(zf, mf) if self.support or self.support_pooled is not None else (zf, 0.0)
        m, s = np_phi(self.st, self.goal_emb, with_bits(zc, mf, self.nb), self.tau)
        out = m + QMAP["uncertainty"] * s + pen
        prim = self.base.prim
        g = mf == self.base.goal_mode
        if g.any():
            if prim.goal_requires_lift:
                out[g] = 0.0
            else:
                gs = prim.goal_slice
                out[g] = (np.abs(zf[g][:, gs] - prim.goal_center) * self.base.W_goal[gs]).sum(1)
        return np.minimum(out, self.cap).reshape(shape)

    def explored(self, mode):
        return np.ones(np.shape(mode), bool)

    def net_spec(self) -> dict:
        """The network for the native rollouts (bep.motor.native.fused: ``net`` in the spec)."""
        st = self.st
        E, _, h = st["W1"].shape
        return {"E": int(E), "in": FEAT + self.nb, "nb": int(self.nb), "h": int(h), "k2": int(st["W3"].shape[-1]),
                "W1": st["W1"].astype(float).ravel(), "b1": st["b1"].astype(float).ravel(),
                "W2": st["W2"].astype(float).ravel(), "b2": st["b2"].astype(float).ravel(),
                "W3": st["W3"].astype(float).ravel(), "b3": st["b3"].astype(float).ravel(),
                "mu": st["mu"].astype(float), "sd": st["sd"].astype(float), "wz": np.asarray(W_FIXED, float),
                "goals": self.goal_emb.astype(float).ravel(), "G": int(self.goal_emb.shape[1]),
                "tau": float(self.tau), "unc": float(QMAP["uncertainty"]),
                "support": {int(m): _support_spec(v) for m, v in self.support.items()},
                "support_pooled": None if self.support_pooled is None else _support_spec(self.support_pooled)}


def _support_spec(v) -> dict:
    W, lo, hi, kappa = v
    return {"W": W, "lo": lo, "hi": hi, "kappa": kappa}


def install_native() -> None:
    """Native rollouts use the map's Phi when a field has one (bep.motor.native.bridge._field)."""
    from ..motor.native import bridge
    if getattr(bridge, "_loop_net", False):
        return
    orig = bridge._field

    def _field(spec, field):
        orig(spec, field)
        if hasattr(field, "net_spec"):
            spec["net"] = field.net_spec()
            spec["cap"] = float(field.cap)
    bridge._field = _field
    bridge._loop_net = True


def wrap_library(lib: dict, qmap: QMap) -> dict:
    """Every schema the map is trusted for (it predicted time-to-goal better than the schema's own field on
    held-out samples at the last sleep) gets a map field; the others keep their field."""
    for t, (prim, field, goals) in list(lib.items()):
        if t in qmap.trusted and t in qmap.goals and not isinstance(field, NetField):
            lib[t] = (prim, NetField(field, qmap, t), goals)
    return lib


# ------------------------------------------------------------------ the loop extension
class MapExtension:
    """Sleep: refit the map on every schema's archive (warm-started); snapshot: qmap.npz with the epoch."""

    def __init__(self):
        self.qmap = None

    def sleep(self, loop) -> dict:
        path = loop.out / "qmap.npz"
        if self.qmap is None:
            self.qmap = QMap(path if path.exists() else None)
        data = {t: schema_data(ex.archive, ex.prim) for t, ex in loop.explorers.items()}
        info = self.qmap.fit(data)
        if info.get("fitted"):
            info["field_held_out"] = self._field_spearman(loop)
            self.qmap.trusted = {t for t, c in self.qmap.calibration.items()
                                 if c["spearman"] > info["field_held_out"].get(t, -1.0)}
            info["trusted"] = sorted(self.qmap.trusted)
            self.qmap.save(path)
        return {"map": info}

    def _field_spearman(self, loop) -> dict:
        """Rank correlation of each schema's own field (bep.explore.field.ModeField, as used in tasks) with its
        time-to-goal labels, on a sample of its archive's non-goal samples."""
        from scipy.stats import spearmanr
        out, rng = {}, np.random.default_rng(2)
        for t, ex in loop.explorers.items():
            Z, M, L = ex.archive.samples()
            ok = np.isfinite(L) & (M >= 0) & (M != ex.prim.goal_mode)
            if ok.sum() < 20:
                continue
            idx = rng.choice(np.flatnonzero(ok), size=min(2000, int(ok.sum())), replace=False)
            v = field_for(ex.folder / "archive.npz", ex.prim, archive=ex.archive).value(Z[idx], M[idx])
            out[t] = round(float(spearmanr(v, L[idx])[0]), 3)
        return out

    def activities(self, loop) -> list:
        return []

    def snapshot(self, loop, snap: Path) -> None:
        if (loop.out / "qmap.npz").exists():
            import shutil
            shutil.copy2(loop.out / "qmap.npz", snap / "qmap.npz")
