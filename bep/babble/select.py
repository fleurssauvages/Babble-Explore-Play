"""Which discovered relations get a primitive (and so an exploration, a field and a place in play).

* every object-state relation,
* every distinct hand-object relation with evidence that it can carry the body,
* every object-object relation with enough probe support.

``worthy`` is the same test without the primitive's construction; the probe's stop rule compares only
worthy relations between batches (small clusters hovering at the size threshold would otherwise keep
the discovered set from ever settling).
"""
from __future__ import annotations

from ..config import PRIMS
from ..world.relations import CC, HC, SC, RelationSet


def worthy(r) -> bool:
    if r.kind == SC:
        return True
    if r.kind == HC:
        return r.carries
    return len(r.members) >= PRIMS["min_members"]


def choose_primitives(R: RelationSet, probes: str) -> list[dict]:
    """Primitive targets, in exploration order (state and hand-object first), with the body kinds they act on."""
    from ..explore.primitive import DiscoveredPrimitive
    cand = {}
    for r in R:
        if not worthy(r):
            continue
        try:
            p = DiscoveredPrimitive(R, r.name, probes=probes)
        except ValueError:                      # never held with a free subject in the probe
            continue
        r.stats["kinds"] = list(p.kinds)
        r.stats["roles"] = [list(k) for k in p.observed_roles()]
        cand[r.name] = r
    first = [n for n, r in cand.items() if r.kind in (SC, HC)]
    second = [n for n, r in cand.items() if r.kind == CC]
    return [{"target": t, "kind": R.get(t).kind, "kinds": R.get(t).stats["kinds"], "roles": R.get(t).stats["roles"],
             "stage": i}
            for i, group in enumerate((first, second)) for t in group]
