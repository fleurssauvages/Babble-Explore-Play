"""Learned effect model of a discovered primitive, from play data (executed calls).

Abstract state s = (set of discovered relation instances, object positions).
Action a = (primitive, subject X, target T), T = cube or table spot.
Everything below is fitted from the play log; no transition rule is written:

* success      P(ok | s, a): logistic regression on role-relative features
               (per discovered relation r: #r(., X), #r(X, .), #r(., T),
               #r(T, .); hand relations on X / on others; T is a spot;
               |XT|; reach radius of X and T; clearance around T; heights)
* duration     E[t | ok, s, a]: ridge regression on the same features
* relations    for every possible instance (r, u, v) after the call:
               P(holds after | it held before, roles(u, v), outcome) with
               roles in {X, T, O(ther), E(hand)}: one smoothed table for
               persistence and one for appearance, per outcome.
* poses        objects related by a relation follow their partner with the
               relation's median offset, learned in object units so it applies
               to any pair of shapes; X follows T on success; an object that
               loses its support drops to the table.

A call *succeeds* when it finished (the primitive reported its goal) and made the relation instance it
is meant to make (``made_instance``; a placement on a table spot: the goal held relative to the spot).  Play
labels its records with ``call_succeeded`` and the brains learn from real calls with the same test.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from ..config import EXEC, HAND
from ..world.objects import SPOT, kind_shape
from ..world.relations import CC, HC


# ------------------------------------------------------------------ success
def made_instance(kind: str, t: str, x: str, target) -> tuple | None:
    """The relation instance a call (primitive t of the given relation kind, subject x) is meant to make."""
    if kind == "hand-object":
        return (t, HAND, x)
    if kind == "object-state":
        return (t, x, "")
    return (t, x, target) if isinstance(target, str) else None


def call_succeeded(result: str, made, before, after) -> bool:
    """The call finished and made its relation.  Without a relation instance to check (a table spot: no body to
    relate to), finishing is the test: the executor reports done only once the goal held relative to the spot
    """
    before, after = set(map(tuple, before)), set(map(tuple, after))
    if result != "done":
        return False
    return (tuple(made) in after) if made is not None else True


# ------------------------------------------------------------------ helpers
def role(o: str, x: str, t) -> str:
    if o == "":
        return "-"                                  # second slot of a state relation
    if o == HAND:
        return "hand"
    if o == x:
        return "X"
    if isinstance(t, str) and o == t:
        return "T"
    return "O"


def target_pos(t, poses, x=None) -> np.ndarray:
    if isinstance(t, str):
        return np.asarray(poses[t], float)
    if t is None:                                   # state / hand-object primitive: no target body
        return np.asarray(poses[x], float)
    return np.array([t[0], t[1], -SPOT.half[2]])


def shape_of(t, shapes, x=None):
    if t is None:
        return shapes[x]
    return shapes[t] if isinstance(t, str) else SPOT


def pair_scale(sx, st) -> np.ndarray:
    """Object units of 'X relative to T' (same as relations.cc_features)."""
    rx = max(sx.half[0], sx.half[1])
    return np.array([st.half[0] + rx, st.half[1] + rx, sx.half[2] + st.half[2]])


def features(rels: set, poses: dict, x: str, t, cc_names, hc_names, shapes, sc_names=()) -> np.ndarray:
    tname = t if isinstance(t, str) else None
    f = []
    for s in sc_names:                              # states of X, of T, and of whatever X rests on / in
        f += [float((s, x, "") in rels), float((s, tname, "") in rels) if tname else 0.0,
              float(sum((s, v, "") in rels for (q, u, v) in rels if q in cc_names and u == x))]
    for r in cc_names:
        f += [sum(1 for (q, u, v) in rels if q == r and v == x), sum(1 for (q, u, v) in rels if q == r and u == x),
              sum(1 for (q, u, v) in rels if q == r and v == tname) if tname else 0,
              sum(1 for (q, u, v) in rels if q == r and u == tname) if tname else 0]
    for g in hc_names:
        f += [float((g, HAND, x) in rels), float(any(q == g and v != x for (q, u, v) in rels))]
    px = np.asarray(poses[x], float)
    pt = target_pos(t, poses, x)
    sx, st = shapes[x], shape_of(t, shapes, x)
    others = [np.asarray(p, float) for n, p in poses.items() if n not in (x, tname)]
    span = 6 * sx.length
    clear = min([np.hypot(*(pt[:2] - o[:2])) for o in others], default=span)
    f += [float(tname is None), float(np.hypot(*(px[:2] - pt[:2]))), float(np.hypot(*px[:2])),
          float(np.hypot(*pt[:2])), float(min(clear, span)), float(px[2]), float(pt[2])]
    f += [sx.half[2], st.half[2], min(st.half[0], st.half[1])]
    return np.array(f, float)


def candidate_instances(objects, cc_names, hc_names, sc_names=(), parts=()):
    out = [(r, u, v) for r in cc_names for u in objects for v in objects if u != v]
    out += [(g, HAND, u) for g in hc_names for u in objects]
    out += [(s, u, "") for s in sc_names for u in parts]
    return out


# ------------------------------------------------------------------ models
class Logistic:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, y, sw=None):
        sw = np.ones(len(y)) if sw is None else np.asarray(sw, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        for _ in range(50):                                   # Newton / IRLS (sample weights sw)
            p = expit(Z @ w)
            g = Z.T @ (sw * (p - y)) + self.l2 * np.r_[w[:-1], 0]
            H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + self.l2 * np.diag(np.r_[np.ones(len(w) - 1), 1e-6])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.hstack([(np.atleast_2d(X) - self.mu) / self.sd, np.ones((len(np.atleast_2d(X)), 1))])
        return 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))


class Ridge:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, y, sw=None):
        sw = np.ones(len(y)) if sw is None else np.asarray(sw, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        self.w = np.linalg.solve((Z * sw[:, None]).T @ Z + self.l2 * np.eye(Z.shape[1]), (Z * sw[:, None]).T @ y)
        return self

    def predict(self, X):
        Z = np.hstack([(np.atleast_2d(X) - self.mu) / self.sd, np.ones((len(np.atleast_2d(X)), 1))])
        return Z @ self.w


class _Constant:
    def __init__(self, p):
        self.p, self.w = p, np.zeros(1)

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.p)


class EffectModel:
    def __init__(self, cc_names, hc_names, prim_rel: str, prior: float = 2.0, sc_names=(), prim_kind: str = "object-object"):
        self.cc, self.hc, self.prim_rel, self.prior = list(cc_names), list(hc_names), prim_rel, prior
        self.sc, self.prim_kind, self.parts = list(sc_names), prim_kind, []

    @classmethod
    def for_relations(cls, R, prim_rel: str) -> "EffectModel":
        rel = prim_rel
        return cls([r.name for r in R.cc], [r.name for r in R.hc], rel, sc_names=[r.name for r in R.sc],
                   prim_kind=R.get(rel).kind)

    # ---------------------------------------------------------------- fit
    def fit(self, recs: list[dict]) -> "EffectModel":
        """Success and duration, relation effects and offsets, from executed calls: play's (by the learning's
        body) and, when given, the robot's validation calls (source "robot", each weighted by its "weight")."""
        recs = ex = [r for r in recs if r.get("source", "executed") in ("executed", "robot")]
        y = np.array([r["success"] for r in ex], float)
        X = np.array([self.phi(r) for r in ex]) if ex else None
        t = np.array([r["time"] for r in ex], float)
        sw = np.array([float(r.get("weight", 1.0)) for r in ex])
        self._data = (X, y, t, sw)                                 # executed calls: refit when reality adds more
        self._fit_outcomes()
        # relation dynamics tables: key (outcome, held_before, rel, role_u, role_v) -> [count_true, n]
        tab = defaultdict(lambda: [0.0, 0.0])
        offs = defaultdict(list)
        for r in recs:
            sh = record_shapes(r)
            before = {tuple(i) for i in r["before"]}
            after = {tuple(i) for i in r["after"]}
            tgt = r["target"]
            for inst in candidate_instances(r["objects"], self.cc, self.hc, self.sc, r.get("parts", [])):
                q, u, v = inst
                key = (bool(r["success"]), inst in before, q, role(u, r["x"], tgt), role(v, r["x"], tgt))
                w = float(r.get("weight", 1.0))
                tab[key][0] += w * float(inst in after)
                tab[key][1] += w
            for (q, u, v) in after:
                if q in self.cc:
                    pu, pv = np.asarray(r["pose_after"][u]), np.asarray(r["pose_after"][v])
                    offs[q].append((pu - pv) / pair_scale(sh[u], sh[v]))
            if r["success"] and tgt is None:            # state / hand-object primitive: X's own displacement
                offs["self"].append(np.asarray(r["pose_after"][r["x"]]) - np.asarray(r["pose_before"][r["x"]]))
            elif r["success"] and not isinstance(tgt, str):
                offs["spot"].append((np.asarray(r["pose_after"][r["x"]]) - target_pos(tgt, r["pose_after"]))
                                    / pair_scale(sh[r["x"]], SPOT))
        self.table = {k: v for k, v in tab.items()}
        # pooled rates used as smoothing priors
        self.rate = {}
        for held in (True, False):
            for ok_ in (True, False):
                s = [v for k, v in self.table.items() if k[1] == held and k[0] == ok_]
                tot = np.sum(s, axis=0) if s else np.array([0.0, 0.0])
                self.rate[(ok_, held)] = float((tot[0] + 0.5) / (tot[1] + 1.0))
        self.offset = {k: np.median(np.array(v), axis=0) for k, v in offs.items() if len(v)}
        return self

    def fit_success(self, X, y, sw) -> None:
        """Fit the success predictor from features, outcome labels and sample weights."""
        # Laplace-smoothed base rate: a primitive seen only succeeding (or failing) is not certain of it
        self.p_success_base = float((np.sum(sw * y) + 1.0) / (np.sum(sw) + 2.0)) if len(y) else 0.5
        if len(y) and len(set(y)) == 2:
            self.success = Logistic().fit(X[:, :self.nrel], y, sw)
        else:
            self.success = _Constant(self.p_success_base)

    def _fit_outcomes(self) -> None:
        """Success probability and duration from the executed calls (play, plus real calls if any)."""
        X, y, t, sw = self._data
        self.fit_success(X, y, sw)
        ok = y > 0
        self.duration = Ridge(3.0).fit(X[ok], t[ok], sw[ok]) if ok.sum() > 5 else None
        self.t_fail = float(np.average(t[~ok], weights=sw[~ok])) if (~ok).any() else EXEC["timeout_s"]
        self.t_mean = float(np.average(t[ok], weights=sw[ok])) if ok.any() else EXEC["timeout_s"] / 2

    def learn(self, rels, poses, x, t, success: bool, duration: float, weight: float = 1.0) -> None:
        """One real call (the model learns from real experience): appended with ``weight`` and the
        success / duration models refit, so the lesson carries over to every state with similar features
        (e.g. "placing a body that is not held fails")."""
        f = features(rels, poses, x, t, self.cc, self.hc, self.shapes, self.sc)
        X, y, tt, sw = self._data
        X = f[None] if X is None or not len(X) else np.vstack([X, f])
        self._data = (X, np.r_[y, float(success)], np.r_[tt, float(duration)], np.r_[sw, float(weight)])
        self._fit_outcomes()

    def set_shapes(self, shapes: dict, parts=()) -> "EffectModel":
        """Shapes of the bodies of the world the model is used in (name -> objects.Shape), and which of
        them are moving parts."""
        self.shapes, self.parts = dict(shapes), list(parts)
        return self

    def ensemble(self, n: int, rng) -> list:
        """Success models fitted on bootstrap resamples of the executed calls (their disagreement measures
        what the model does not know yet, not what is merely uncertain)."""
        X, y, _, sw = self._data
        out = []
        for _ in range(n):
            if not len(y):
                out.append(_Constant(0.5))
                continue
            i = rng.integers(len(y), size=len(y))
            yi, wi = y[i], sw[i]
            p0 = float((np.sum(wi * yi) + 1.0) / (np.sum(wi) + 2.0))
            out.append(Logistic().fit(X[i][:, :self.nrel], yi, wi) if len(set(yi)) == 2 else _Constant(p0))
        return out

    def phi_now(self, rels, poses, x, t) -> np.ndarray:
        """Success-model input (the relational features) of a call from the given state, as a 1-row matrix."""
        return features(rels, poses, x, t, self.cc, self.hc, self.shapes, self.sc)[None, :self.nrel]

    def phi(self, r) -> np.ndarray:
        return features({tuple(i) for i in r["before"]}, r["pose_before"], r["x"], r["target"], self.cc, self.hc,
                        record_shapes(r), self.sc)

    # ------------------------------------------------------------ predict
    @property
    def nrel(self) -> int:
        """Number of leading relational features (roles of X and the target in the relations): success is
        predicted from these only, which transfer across numbers and layouts of bodies; geometry is left to
        the primitives' own fields (time, coverage)."""
        return 3 * len(self.sc) + 4 * len(self.cc) + 2 * len(self.hc)

    def p_ok(self, rels, poses, x, t) -> float:
        f = features(rels, poses, x, t, self.cc, self.hc, self.shapes, self.sc)
        return float(self.success.predict(f[:self.nrel])[0])

    def time(self, rels, poses, x, t, ok: bool) -> float:
        if not ok or self.duration is None:
            return self.t_fail if not ok else self.t_mean
        return float(np.clip(self.duration.predict(features(rels, poses, x, t, self.cc, self.hc, self.shapes, self.sc))[0],
                             0.0, EXEC["timeout_s"]))

    def p_inst(self, ok: bool, held: bool, q, ru, rv) -> float:
        c, n = self.table.get((ok, held, q, ru, rv), (0.0, 0.0))
        # prior (frame assumption): without evidence a relation keeps its truth value; observed data for
        # this role combination overrides it.  This is what lets a model learned with two bodies be
        # used with many (relations among the others, or of the target with a third body).
        p0 = float(held)
        return float((c + self.prior * p0) / (n + self.prior))

    def predict_relations(self, rels, objects, x, t, ok: bool) -> dict:
        """P(holds after) for every candidate instance, given the outcome."""
        return {inst: self.p_inst(ok, inst in rels, inst[0], role(inst[1], x, t), role(inst[2], x, t))
                for inst in candidate_instances(objects, self.cc, self.hc, self.sc, self.parts)}

    def most_likely(self, rels, objects, x, t, ok: bool) -> frozenset:
        """The most likely relation graph after the call, given its outcome (instances with P > 1/2).  The
        target relation and release are part of a successful call's definition. All grasp relations
        of a single parallel gripper must refer to the same object."""
        probabilities = self.predict_relations(rels, objects, x, t, ok)
        after = {inst for inst, p in probabilities.items() if p > 0.5}
        if ok:
            made = made_instance(self.prim_kind, self.prim_rel, x, t)
            if made is not None:
                after.add(made)
            if self.prim_kind == CC:
                after = {i for i in after if i[0] not in self.hc or i[2] != x}
        grips = [i for i in after if i[0] in self.hc]
        if grips:
            held = x if ok and self.prim_kind == HC else max(
                grips, key=lambda i: (probabilities[i], i in rels, i))[2]
            after.difference_update(i for i in grips if i[2] != held)
        return frozenset(after)

    def update_poses(self, rels, new, poses, x, t, ok) -> dict:
        p = {k: np.asarray(v, float).copy() for k, v in poses.items()}
        if ok and t is None:
            p[x] = p[x] + self.offset.get("self", np.zeros(3))
        elif ok:
            off = self.offset.get("spot" if not isinstance(t, str) else self.prim_rel, np.array([0, 0, 1.0]))
            p[x] = target_pos(t, poses) + off * pair_scale(self.shapes[x], shape_of(t, self.shapes))
        # objects keep following their supports (median relation offset); objects whose
        # support relation vanished drop to the table where they are
        for _ in range(len(p)):
            for (q, u, v) in new:
                if q in self.cc and u != x and u in p and v in p:
                    p[u] = p[v] + self.offset.get(q, np.array([0, 0, 1.0])) * pair_scale(self.shapes[u],
                                                                                          self.shapes[v])
        for (q, u, v) in rels:
            if q in self.cc and (q, u, v) not in new and u != x and u not in self.parts \
                    and not any(i[0] in self.cc and i[1] == u for i in new):
                p[u] = np.array([p[u][0], p[u][1], self.shapes[u].half[2]])
        return p

    def to_json(self) -> dict:
        return {"p_success_base": self.p_success_base,
                "table": {str(k): v for k, v in sorted(self.table.items(), key=lambda kv: str(kv[0]))},
                "offset": {k: np.round(v, 4).tolist() for k, v in self.offset.items()},
                "t_mean": self.t_mean, "t_fail": self.t_fail}


# ------------------------------------------------------------------ evaluation


def record_shapes(r) -> dict:
    """name -> objects.Shape of a play record's bodies (free shapes and moving parts)."""
    return {n: kind_shape(r.get("shapes", {}).get(n, "cube")) for n in r["objects"]}


def load(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
