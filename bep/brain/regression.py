"""Backward search for the brain: the schemas read as lifted operators, and goal regression through them.

The effect models (bep.play.effect) only predict forward.  For a search that also grows back from the goal, each
schema is read from play's successful calls (play.jsonl) as a lifted operator over role literals (``r0(X,T)``,
``g0(hand,X)``, ``r0(O,X)``: X the subject, T the target body, O any other body, hand the hand):

* positive preconditions: literals present before at least BRAIN['bw_pre'] of its successful calls;
* negative preconditions: role patterns present before at least BRAIN['bw_min'] of its calls, with which it
  succeeded at most BRAIN['bw_never'] of the time (evidence that the context defeats it; a pattern play never met,
  e.g. one that needs a third body, constrains nothing);
* effects: a literal it makes (absent before, present after, in at least half of those successes) and one it
  breaks (present before, absent after, in at least half of those).

A subgoal is a set of required instances and of forbidden patterns.  A pattern's slot is a body, the hand, or a
wildcard over the bodies with exclusions: "no body on X" is (r0, *, X); the goal's free hand is (g, hand, *).
Regressing a subgoal through a call (schema, X, target) that contributes to it (makes a required instance, or
breaks what a forbidden pattern covers) removes what the call achieves and adds its preconditions; the call is
not used when it would break a required instance or make a forbidden one, nor when the result contradicts itself.

The search in bep.brain.brain alternates a forward step and a backward step; a forward state that satisfies a
backward subgoal is a meeting point, and the subgoal's calls are replayed from it with the effect models (each
must be available and its success must lead on) before the plan is accepted, priced by the forward models.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..config import BRAIN, HAND
from ..play.effect import role

ROLES_CC = ("X", "T", "O")


def _role(o, x, t) -> str:
    return role(o, x, t if isinstance(t, str) else None)


def lifted(state, x, t) -> set:
    return {(q, _role(u, x, t), _role(v, x, t)) for q, u, v in map(tuple, state)}


def wild(excl=()) -> tuple:
    return ("*", frozenset(excl))


def is_wild(slot) -> bool:
    return isinstance(slot, tuple)


def slot_match(slot, value) -> bool:
    if is_wild(slot):
        return value not in slot[1] and value not in ("", HAND)
    return slot == value


def matches(pattern, inst) -> bool:
    return pattern[0] == inst[0] and slot_match(pattern[1], inst[1]) and slot_match(pattern[2], inst[2])


class Regression:
    """The schemas as lifted operators (fitted from play's records), and regression of subgoals through calls."""

    def fit(self, recs: list[dict], R) -> "Regression":
        cc, hc, sc = [r.name for r in R.cc], [r.name for r in R.hc], [r.name for r in R.sc]
        patterns = [(q, a, b) for q in cc for a in ROLES_CC for b in ROLES_CC if not (a == b and a != "O")]
        patterns += [(q, "hand", a) for q in hc for a in ROLES_CC]
        patterns += [(q, a, "-") for q in sc for a in ROLES_CC]
        self.ops: dict = {}
        by, tried, won = defaultdict(list), defaultdict(Counter), defaultdict(Counter)
        for r in recs:
            if r.get("source", "executed") != "executed":
                continue
            for lit in lifted(r["before"], r["x"], r["target"]):
                tried[r["primitive"]][lit] += 1
                won[r["primitive"]][lit] += bool(r["success"])
            if r["success"]:
                by[r["primitive"]].append(r)
        n_all = Counter(r["primitive"] for r in recs if r.get("source", "executed") == "executed")
        for t, ok in by.items():
            n = len(ok)
            before = [lifted(r["before"], r["x"], r["target"]) for r in ok]
            after = [lifted(r["after"], r["x"], r["target"]) for r in ok]
            freq = Counter(lit for b in before for lit in b)
            pre_pos = {lit for lit, c in freq.items() if c >= BRAIN["bw_pre"] * n and "O" not in lit[1:]}
            pre_neg = {p for p in patterns if tried[t][p] >= BRAIN["bw_min"]
                       and won[t][p] <= BRAIN["bw_never"] * tried[t][p]}
            made, absent, broke, present = Counter(), Counter(), Counter(), Counter()
            for b, a in zip(before, after):
                for p in patterns:
                    if p in b:
                        present[p] += 1
                        broke[p] += p not in a
                    else:
                        absent[p] += 1
                        made[p] += p in a
            adds = {p for p in patterns if absent[p] and made[p] >= 0.5 * absent[p] and "O" not in p[1:]}
            dels = {p for p in patterns if present[p] and broke[p] >= 0.5 * present[p]}
            rate = n / max(n_all[t], 1)
            mean_t = sum(float(r["time"]) for r in ok) / n
            self.ops[t] = {"pre_pos": pre_pos, "pre_neg": pre_neg, "adds": adds, "dels": dels,
                           "cost": mean_t / max(rate, BRAIN["min_p"])}
        self.mean_cost = (sum(o["cost"] for o in self.ops.values()) / len(self.ops)) if self.ops else 10.0
        return self

    # ------------------------------------------------------------ grounding
    @staticmethod
    def _ground(lit, x, tb, objects=None):
        """A role literal for a call on (x, target body tb): an instance, or a pattern when it names O; None when it
        names T and the call has no target body."""
        out = [lit[0]]
        for r in lit[1:]:
            if r == "X":
                out.append(x)
            elif r == "T":
                if tb is None:
                    return None
                out.append(tb)
            elif r == "hand":
                out.append(HAND)
            elif r == "-":
                out.append("")
            else:
                out.append(wild([x] + ([tb] if tb else [])))
        return tuple(out)

    # ------------------------------------------------------------ subgoals
    def goal_subgoal(self, goal) -> tuple:
        neg = set(goal.forbid)
        if goal.hand_free:
            neg |= {(q, HAND, wild()) for q in goal.hc}
        return frozenset(goal.require), frozenset(neg)

    @staticmethod
    def satisfies(state, G) -> bool:
        pos, neg = G
        return pos <= state and not any(matches(n, i) for n in neg for i in state)

    def unmet(self, G, state) -> int:
        pos, neg = G
        return len(pos - state) + sum(any(matches(n, i) for i in state) for n in neg)

    def regress(self, G, key):
        """The subgoal before the call ``key`` = (schema, X, target) that leads to G, or None (see the module
        docstring)."""
        t, x, tt = key
        op = self.ops.get(t)
        if op is None:
            return None
        tb = tt if isinstance(tt, str) and tt != "spot" else None
        pos, neg = G
        adds = {a for a in (self._ground(p, x, tb) for p in op["adds"]) if a is not None}
        dels = [d for d in (self._ground(p, x, tb) for p in op["dels"]) if d is not None]
        if any(matches(n, a) for n in neg for a in adds) or any(matches(d, i) for d in dels for i in pos):
            return None
        new_neg, changed = set(), False
        for n in neg:
            whole, freed = False, set()
            for d in dels:
                if d[0] != n[0]:
                    continue
                ok, frees = True, None
                for ns, ds in zip(n[1:], d[1:]):
                    if not is_wild(ns):
                        ok &= slot_match(ds, ns)          # the call breaks this very slot's instances
                    elif not is_wild(ds):
                        if slot_match(ns, ds):
                            frees = ds                    # it breaks the instances of one body the pattern covers
                        else:
                            ok = False
                if not ok:
                    continue
                if frees is None:
                    whole = True
                else:
                    freed.add(frees)
            if whole:                                     # whatever the pattern forbids, the call breaks
                changed = True
                continue
            if freed:                                     # before the call, those bodies may still match it
                changed = True
                n = (n[0], *[(("*", ns[1] | frozenset(freed)) if is_wild(ns) else ns) for ns in n[1:]])
            new_neg.add(n)
        if not (adds & pos) and not changed:
            return None                                   # the call contributes nothing
        new_pos = set(pos - adds)
        for p in op["pre_pos"]:
            g = self._ground(p, x, tb)
            if g is not None and not any(is_wild(s) for s in g[1:]):
                new_pos.add(g)
        for p in op["pre_neg"]:
            g = self._ground(p, x, tb)
            if g is not None:
                new_neg.add(g)
        if any(matches(n, i) for n in new_neg for i in new_pos):
            return None
        return frozenset(new_pos), frozenset(new_neg)

    def cost(self, key) -> float:
        op = self.ops.get(key[0])
        return op["cost"] if op else self.mean_cost
