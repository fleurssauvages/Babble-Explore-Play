"""A learned, factored cost-to-go for the brain's search: how long until each unmet goal literal holds.

The search over relation graphs (bep.brain.brain) ranks a state by its cost so far plus a heuristic.  Without
one, every unmet goal literal costs BRAIN['goal_weight_s'] seconds.  Here that per-literal cost is learned from
the chains of calls play and imagination executed (play.jsonl), and the heuristic is the sum over the goal's unmet
literals: a learned, context-aware version of the additive heuristic of classical planning.

* A literal is an instance to make (a required instance that does not hold) or to break (a forbidden one that
  holds, or a grip while the goal wants the hand free).
* It is read from the state in its own frame: its relation and mode, and every instance of the state in role
  form relative to the literal's bodies (``r0(O,X)``: another body on the literal's first body, X; T its second
  body; hand the hand).  So one term looks at a literal and its relational neighbourhood only, and a cost learned
  with two bodies applies to each literal of a scene with many.  Role literals never seen in training are
  ignored (the model uses what the data supports).
* Targets: along each recorded chain, from every state, the time spent until the literal first holds (to make)
  or first stops holding (to break), counting failed calls; literals the chain never reaches are left out.
  Random play reaches a literal by detours too, so the costs are fitted with an expectile loss of level
  BRAIN['h_expectile'] < 1/2: the model estimates the shorter ways of reaching it, not the average detour.
* A small MLP (torch at fit time, numpy at search time) with a softplus output (a cost is positive).

The heuristic is fitted once per solve from the learning's records; learning itself is unchanged.  It is not
admissible: it guides the search, which still checks every plan step with the effect models.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import BRAIN, HAND
from ..play.effect import candidate_instances, role


def _frame(inst) -> tuple:
    """(X, T) of a literal: its first and second body (a hand-object literal: the body; a state literal: the part)."""
    q, u, v = inst
    if u == HAND:
        return v, None
    return u, (v or None)


def literal_keys(state, inst, mode: str) -> dict:
    """The features of one literal in a state: its relation and mode, and the state's instances in role form."""
    x, t = _frame(inst)
    out = defaultdict(float)
    out[f"{mode}:{inst[0]}"] = 1.0
    for q, u, v in state:
        out[f"{q}({role(u, x, t)},{role(v, x, t)})"] += 1.0
    return out


def chains(recs: list[dict]) -> list[list[dict]]:
    """The recorded chains of calls (play and imagination episodes), each in its order."""
    by = defaultdict(list)
    for r in recs:
        if r.get("source", "executed") != "executed" or "episode" not in r or "step" not in r:
            continue
        by[(r.get("epoch"), r["episode"], "imagined_goal" in r)].append(r)
    return [sorted(c, key=lambda r: r["step"]) for c in by.values()]


def call_costs(chain: list[dict], models: dict | None) -> list[float]:
    """What each call of a chain costs: its time (``models`` None), or, in the planner's units, the expected time
    with retries the effect models give a successful call, (p t_ok + (1 - p) t_fail) / p, and 0 for a failed one
    (its time is part of the retries of the call that succeeds)."""
    if models is None:
        return [float(r["time"]) for r in chain]
    from ..play.effect import record_shapes
    out = []
    for r in chain:
        m = models.get(r["primitive"])
        if m is None:
            out.append(float(r["time"]))
            continue
        if not r["success"]:
            out.append(0.0)
            continue
        m.set_shapes(record_shapes(r), r.get("parts", []))
        rels = {tuple(i) for i in r["before"]}
        p = max(m.p_ok(rels, r["pose_before"], r["x"], r["target"]), BRAIN["min_p"])
        t_ok = m.time(rels, r["pose_before"], r["x"], r["target"], True)
        t_fail = m.time(rels, r["pose_before"], r["x"], r["target"], False)
        out.append((p * t_ok + (1 - p) * t_fail) / p)
    return out


def training_rows(recs: list[dict], R, models: dict | None = None) -> list[tuple[dict, float]]:
    """(features, cost until the literal is made or broken) along every chain (``call_costs``: seconds, or the
    planner's expected cost with ``models``); literals and features over the relations of R."""
    names = {r.name for r in R}
    cc, hc, sc = ([r.name for r in rs] for rs in (R.cc, R.hc, R.sc))
    rows = []
    for c in chains(recs):
        before = [{tuple(i) for i in r["before"] if i[0] in names} for r in c]
        after = [{tuple(i) for i in r["after"] if i[0] in names} for r in c]
        times = np.cumsum([0.0] + call_costs(c, models))
        r0 = c[0]
        insts = candidate_instances(r0["objects"], cc, hc, sc, r0.get("parts", []))
        for i, s in enumerate(before):
            for inst in insts:
                holds = inst in s
                for j in range(i, len(c)):
                    if (inst in after[j]) != holds:
                        rows.append((literal_keys(s, inst, "break" if holds else "make"),
                                     float(times[j + 1] - times[i])))
                        break
    return rows


class LiteralCost:
    """The learned cost of one literal in a state (seconds), and the heuristic of a goal (their sum)."""

    def fit(self, recs: list[dict], R, seed: int = 0, models: dict | None = None) -> "LiteralCost":
        """``models``: the effect models, to label in the planner's units (BRAIN['h_units'] "planner")."""
        import torch
        rows = training_rows(recs, R, models)
        self.n_rows = len(rows)
        self.known = {k for f, _ in rows for k in f if ":" in k}      # (mode, relation) pairs seen in training
        if not rows:
            self.W = None
            return self
        self.keys = sorted({k for f, _ in rows for k in f})
        col = {k: j for j, k in enumerate(self.keys)}
        X = np.zeros((len(rows), len(self.keys)))
        for n, (f, _) in enumerate(rows):
            for k, v in f.items():
                X[n, col[k]] = v
        y = np.array([c for _, c in rows])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        torch.manual_seed(seed)
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        h = BRAIN["h_hidden"]
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
        tau = BRAIN["h_expectile"]
        for _ in range(BRAIN["h_steps"]):
            pred = torch.nn.functional.softplus(net(Xt)[:, 0])
            e = yt - pred
            loss = (torch.where(e > 0, tau, 1 - tau) * e * e).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.loss = float(loss.item())
        self.W = [p.detach().numpy().astype(float) for p in net.parameters()]
        self._col = col
        self._cache: dict = {}
        return self

    def literal(self, state: frozenset, inst, mode: str) -> float:
        """Seconds until the literal is made (mode "make") or broken ("break") from the state."""
        key = (state, inst, mode)
        if key not in self._cache:
            x = np.zeros(len(self.keys))
            for k, v in literal_keys(state, inst, mode).items():
                j = self._col.get(k)
                if j is not None:
                    x[j] = v
            W1, b1, W2, b2 = self.W
            z = np.maximum(((x - self.mu) / self.sd) @ W1.T + b1, 0.0) @ W2.T + b2
            self._cache[key] = float(np.logaddexp(0.0, z[0]))            # softplus
        return self._cache[key]

    def __call__(self, goal, state: frozenset) -> float:
        """The heuristic: the sum of the unmet literals' costs (BRAIN['goal_weight_s'] for a literal whose relation
        and mode training never saw)."""
        def cost(inst, mode):
            if self.W is None or f"{mode}:{inst[0]}" not in self.known:
                return BRAIN["goal_weight_s"]
            return self.literal(state, inst, mode)
        h = sum(cost(i, "make") for i in goal.require if i not in state)
        h += sum(cost(i, "break") for i in goal.forbid if i in state)
        if goal.hand_free:
            h += sum(cost(i, "break") for i in state if i[0] in goal.hc)
        return h

    def report(self) -> dict:
        return {"rows": self.n_rows, "features": len(getattr(self, "keys", [])),
                "loss": round(getattr(self, "loss", float("nan")), 3)}
