"""Schemas: each primitive with its learned precondition, split or merged at sleep between epochs.

A schema is a primitive (its goal relation, graph and field) together with what play learned about where it
works.  Two things are learned from play's calls:

* a precondition network per schema: P(success | the call's context), a small MLP ensemble (torch) on the
  relational features of bep.play.effect (roles of X and the target in every relation), the contact flags of
  X and its reference at the start (bep.babble.events.flags), and two continuous quantities the schema itself
  provides: Phi at the start and the distance to its validated graph in typical hops.  The ensemble's spread
  is what the schema does not know yet;
* accommodation (Piaget): when a schema's outcomes vary within the contexts its features tell apart, the
  discrete indicator that best separates success from failure (a contact flag at the start, or a relation
  literal in role form, e.g. r0(O, X): something rests on X) splits it into *variants*, each with its own
  success statistics and its own practice: the arbiter can give exploration of a variant its own blocks, which
  start from that variant's failed calls (bep.explore.explore).  A split needs SCHEMA['split_gain'] bits of
  information gain and SCHEMA['split_min'] calls on each side; variants whose rates no longer differ are
  merged back, and two schemas whose goal relations fire on mostly the same states are merged (the one with
  the better executor stays).

Splits and merges happen only at sleep, between epochs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .effect import EffectModel, load, role
from ..babble.events import FLAGS
from ..config import SCHEMA


# ------------------------------------------------------------------ indicators of a call's context
def indicators(rec: dict) -> set[str]:
    """Discrete indicators of a play call's start: contact flags (``fingers=2``, ``xy=1``, ...) and relation
    literals in role form (``r0(O,X)``: a relation between another body and X)."""
    out = set()
    for name, v in zip(FLAGS, rec.get("flags0") or []):
        out.add(f"{name}={int(v)}")
    x, t = rec["x"], rec["target"]
    tgt = t if isinstance(t, str) else (None if t is None else tuple(t))
    for q, u, v in rec["before"]:
        out.add(f"{q}({role(u, x, tgt)},{role(v, x, tgt)})")
    return out


def entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def best_split(recs: list[dict]) -> dict | None:
    """The indicator with the largest information gain about success (both sides >= split_min calls)."""
    y = np.array([r["success"] for r in recs], float)
    if len(y) < SCHEMA["min_calls"] or y.min() == y.max():
        return None
    H = entropy(y.mean())
    ind = [indicators(r) for r in recs]
    cand = Counter(i for s in ind for i in s)
    best = None
    for name, n in cand.items():
        has = np.array([name in s for s in ind])
        if has.sum() < SCHEMA["split_min"] or (~has).sum() < SCHEMA["split_min"]:
            continue
        p1, p0 = y[has].mean(), y[~has].mean()
        gain = H - (has.mean() * entropy(p1) + (1 - has.mean()) * entropy(p0))
        if best is None or gain > best["gain"]:
            best = {"indicator": name, "gain": round(float(gain), 4), "p_with": round(float(p1), 3),
                    "p_without": round(float(p0), 3), "n_with": int(has.sum()), "n_without": int((~has).sum())}
    return best


# ------------------------------------------------------------------ the precondition network
def context_record(env, prim, field, x: str, target, rels, poses: dict) -> dict:
    """A call's context from the current scene, as play records it (relations before, poses, roles, contact
    flags, Phi and the distance to the schema's graph at the start): the precondition's input."""
    from ..babble.events import flags
    from ..world.objects import VIEW_S
    view = env.pair_view(env.sensors(), x, prim.reference(env, x, target))
    rel = prim.rel(env, view, VIEW_S)
    z, m = rel["z"], int(rel["mode"])
    hop = float(field.roadmap.near(z, m))
    return {"before": [list(i) for i in rels], "pose_before": {k: list(map(float, v)) for k, v in poses.items()},
            "x": x, "target": target if isinstance(target, str) or target is None else [target.x, target.y],
            "objects": list(env.cubes), "flags0": [int(v) for v in flags(view)],
            "phi0": float(field.value(z, m)), "hop0": hop if np.isfinite(hop) else None}


def load_preconditions(path) -> dict:
    """The schemas' precondition networks saved at sleep (preconditions.npz), by schema."""
    path = Path(path)
    if not path.exists():
        return {}
    by: dict = {}
    with np.load(path) as f:
        for k in f.files:
            t, key = k.split("__", 1)
            by.setdefault(t, {})[key] = f[k]
    return {t: Precondition.from_state(st) for t, st in by.items()}


def features(model: EffectModel, rec: dict) -> np.ndarray:
    rel = model.phi(rec)[:model.nrel]
    fl = np.zeros(len(FLAGS) + 1)                       # fingers as a count, the rest as booleans
    if rec.get("flags0"):
        fl[:len(FLAGS)] = rec["flags0"]
    phi = rec.get("phi0")
    hop = rec.get("hop0")
    cont = np.array([phi if phi is not None else 30.0, min(hop, 10.0) if hop is not None else 10.0], float)
    return np.concatenate([rel, fl, cont])


class Precondition:
    """P(success | context): an ensemble of small MLPs (1 x SCHEMA['hidden']), bootstrap-resampled."""

    def __init__(self):
        self.st = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "Precondition":
        import torch
        torch.set_num_threads(4)
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        E, h, d = SCHEMA["ensemble"], SCHEMA["hidden"], X.shape[1]
        g = torch.Generator().manual_seed(seed)
        W1 = (torch.randn(E, d, h, generator=g) * (2 / d) ** 0.5).requires_grad_()
        b1 = torch.zeros(E, 1, h, requires_grad=True)
        W2 = (torch.randn(E, h, 1, generator=g) * (1 / h) ** 0.5).requires_grad_()
        b2 = torch.zeros(E, 1, 1, requires_grad=True)
        boot = torch.tensor(np.stack([rng.integers(len(y), size=len(y)) for _ in range(E)]))
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-2, weight_decay=1e-3)
        for _ in range(SCHEMA["steps"]):
            xb, yb = Xn[boot], Y[boot]                                     # (E, N, d), (E, N)
            logit = (torch.bmm(torch.relu(torch.bmm(xb, W1) + b1), W2) + b2)[..., 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.st = {k: v.detach().numpy() for k, v in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2))}
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) over the ensemble of P(success)."""
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        h = np.maximum(np.einsum("ni,eio->eno", Xn, self.st["W1"]) + self.st["b1"], 0.0)
        p = 1 / (1 + np.exp(-(np.einsum("eni,eio->eno", h, self.st["W2"]) + self.st["b2"])[..., 0]))
        return p.mean(0), p.std(0)

    def state(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, **self.st}

    @staticmethod
    def from_state(st: dict) -> "Precondition":
        p = Precondition()
        p.mu, p.sd = st["mu"], st["sd"]
        p.st = {k: st[k] for k in ("W1", "b1", "W2", "b2")}
        return p


def _logloss(y, p) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(model_of, recs: list[dict], k: int = 3) -> dict:
    """Held-out (k-fold) log-loss of the precondition network, the original effect model's logistic success
    model, and the base rate."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(recs))
    y = np.array([r["success"] for r in recs], float)
    out = {"net": [], "effect": [], "base": []}
    for f in np.array_split(idx, k):
        test = set(f.tolist())
        tr = [recs[i] for i in idx if i not in test]
        te = [recs[i] for i in f]
        if len({r["success"] for r in tr}) < 2:
            continue
        m = model_of()
        Xtr = np.array([features(m, r) for r in tr])
        Xte = np.array([features(m, r) for r in te])
        ytr, yte = np.array([r["success"] for r in tr], float), y[f]
        pn, _ = Precondition().fit(Xtr, ytr).predict(Xte)
        em = model_of().fit(tr)
        from .effect import record_shapes
        pe = []
        for r in te:
            em.set_shapes(record_shapes(r), r.get("parts", []))
            pe.append(em.p_ok({tuple(i) for i in r["before"]}, r["pose_before"], r["x"], r["target"]))
        out["net"].append(_logloss(yte, pn))
        out["effect"].append(_logloss(yte, np.array(pe)))
        out["base"].append(_logloss(yte, np.full(len(yte), (ytr.sum() + 1) / (len(ytr) + 2))))
    return {k: round(float(np.mean(v)), 4) if v else None for k, v in out.items()}


# ------------------------------------------------------------------ chunking (macro-schemas)
def chunk_macros(play: list, targets: list, path) -> list:
    """Pairs of schemas that play chains on the same body (a on X, then b on X, in the same episode, a successful)
    at least PLAY['macro_min'] times, b succeeding at least PLAY['macro_p'] of them, become macro-schemas
    ``a>b`` executed as one call (basal-ganglia chunking, Graybiel 1998)."""
    from ..config import PLAY
    n, ok = Counter(), Counter()
    by_ep: dict = {}
    for r in play:
        if r.get("episode") is not None and ">" not in r["primitive"]:
            by_ep.setdefault(r["episode"], []).append(r)
    for recs in by_ep.values():
        recs.sort(key=lambda r: r.get("step", 0))
        for r1, r2 in zip(recs, recs[1:]):
            if r1["success"] and r1["x"] == r2["x"] and r1["primitive"] != r2["primitive"] \
                    and r1["primitive"] in targets and r2["primitive"] in targets:
                k = f"{r1['primitive']}>{r2['primitive']}"
                n[k] += 1
                ok[k] += bool(r2["success"])
    macros = [{"name": k, "n": c, "p": round(ok[k] / c, 3)} for k, c in n.most_common()
              if c >= PLAY["macro_min"] and ok[k] / c >= PLAY["macro_p"]]
    Path(path).write_text(json.dumps(macros, indent=1))
    return [m["name"] for m in macros]


# ------------------------------------------------------------------ the loop extension
class SchemaExtension:
    """Sleep: fit the preconditions, split or merge schemas; activities: exploration of each variant."""

    def __init__(self):
        self.variants: dict[str, dict] = {}            # "t|indicator" -> {"target", "indicator", "with": bool, ...}

    def _context(self, v: dict):
        name, want = v["indicator"], v["with"]
        return lambda info: info is not None and (name in indicators(info)) == want

    def sleep(self, loop) -> dict:
        from ..world.relations import RelationSet
        path = loop.out / "play.jsonl"
        if not path.exists() or not loop.prims:
            return {}
        play = [r for r in load(path) if "flags0" in r]
        R = RelationSet.load(loop.out / "relations")
        report, states = {"splits": [], "merged": [], "preconditions": {}}, {}
        targets = {p["target"] for p in loop.prims}
        # merge back variants whose rates no longer differ; drop variants of schemas that disappeared
        for key, v in list(self.variants.items()):
            recs = [r for r in play if r["primitive"] == v["target"]]
            s = best_split(recs)
            if v["target"] not in targets or s is None or s["indicator"] != v["indicator"] \
                    or s["gain"] < SCHEMA["split_gain"] / 2:
                report["merged"].append(key)
                del self.variants[key]
        for t in sorted(targets):
            recs = [r for r in play if r["primitive"] == t]
            if len(recs) < SCHEMA["min_calls"] or len({r["success"] for r in recs}) < 2:
                continue
            model_of = lambda t=t: EffectModel.for_relations(R, t)              # noqa: E731
            m = model_of()
            X = np.array([features(m, r) for r in recs])
            y = np.array([r["success"] for r in recs], float)
            pre = Precondition().fit(X, y)
            states[t] = pre.state()
            report["preconditions"][t] = {"calls": len(recs), "success": round(float(y.mean()), 3),
                                          "held_out_logloss": evaluate(model_of, recs)}
            s = best_split(recs)
            if s is not None and s["gain"] >= SCHEMA["split_gain"]:
                worse = s["p_with"] < s["p_without"]
                key = f"{t}|{s['indicator']}" if worse else f"{t}|not {s['indicator']}"
                if key not in self.variants:
                    self.variants[key] = {"target": t, "indicator": s["indicator"], "with": worse, **s}
                    report["splits"].append(key)
                    # a variant starts from its schema's competence history, not with optimism: a context where
                    # the schema cannot succeed (e.g. grasping while holding something) earns no attention
                    loop.arbiter.hist[f"explore:{key}"] = list(loop.arbiter.hist.get(f"explore:{t}", []))
        # schemas whose goal relations fire on mostly the same states: the one with the better executor stays
        report["merged"] += self._merge_schemas(loop)
        from ..config import PLAY
        if PLAY["macros"]:
            report["macros"] = chunk_macros(load(path), sorted(targets), loop.out / "macros.json")
        np.savez_compressed(loop.out / "preconditions.npz",
                            **{f"{t}__{k}": v for t, st in states.items() for k, v in st.items()})
        (loop.out / "schemas.json").write_text(json.dumps({"variants": self.variants, "report": report}, indent=1,
                                                          default=float))
        report["variants"] = sorted(self.variants)
        return {"schemas": report}

    def _merge_schemas(self, loop) -> list:
        from ..world.objects import VIEW_S
        from ..world.relations import cc_features
        R = loop.R_all
        prims = [p for p in loop.prims if p["kind"] == "object-object"]
        if len(prims) < 2:
            return []
        f = np.load(loop.out / "probes.npz")
        fc = cc_features(f["s2"], VIEW_S, "AB")
        fires = {p["target"]: R.get(p["target"]).holds_f(fc) for p in prims}
        merged = []
        for i, a in enumerate(prims):
            for b in prims[i + 1:]:
                fa, fb = fires[a["target"]], fires[b["target"]]
                small = min(fa.sum(), fb.sum())
                if small and (fa & fb).sum() >= SCHEMA["merge_overlap"] * small:
                    sa = loop.explorers[a["target"]].history[-1]["success"] if loop.explorers[a["target"]].history else 0
                    sb = loop.explorers[b["target"]].history[-1]["success"] if loop.explorers[b["target"]].history else 0
                    drop = b if sa >= sb else a
                    if drop in loop.prims:
                        loop.prims.remove(drop)
                        merged.append(f"{drop['target']} into {(a if drop is b else b)['target']}")
        if merged:
            (loop.out / "primitives.json").write_text(json.dumps(loop.prims, indent=1))
            loop._explorers()
        return merged

    def activities(self, loop) -> list:
        return [f"explore:{k}" for k, v in self.variants.items() if v["target"] in loop.explorers]

    def run(self, loop, name: str):
        key = name.split(":", 1)[1]
        v = self.variants[key]
        ex = loop.explorers[v["target"]]
        st = ex.run_block(loop.pool, focus=self._context(v))
        st["variant"] = key
        return st, st["competence"]

    def snapshot(self, loop, snap: Path) -> None:
        import shutil
        for name in ("preconditions.npz", "schemas.json"):
            if (loop.out / name).exists():
                shutil.copy2(loop.out / name, snap / name)
