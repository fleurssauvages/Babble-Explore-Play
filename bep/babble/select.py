"""Which discovered relations get a primitive (and so an exploration, a field and a place in play).

Goals (a primitive makes the relation):

* every object-state relation,
* every distinct hand-object relation with evidence that it can carry the body,
* every object-object relation with enough probe support.

Channels (a primitive moves X *through* the relation): a dynamic hand-object relation, through which the hand
moves the body without changing its contacts (e.g. pushing).  A channel primitive is explored backward like
the others, with X as the actor (its graph is in X's space), and executed by the channel's controller, learned
from its probe cluster (bep.motor.hog_motor.PushHop).  A channel is not a goal of
its own (it holds only while the hand moves X): each object-object relation that holds with X resting on a
table spot gets a channel primitive ``<relation>~<channel>``, the relation made relative to a table spot, the
hand never carrying X (whether X gets there is for exploration to find).

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
        return r.carries or is_channel(r)
    return len(r.members) >= PRIMS["min_members"]


def is_channel(r) -> bool:
    """A dynamic hand relation (its effect moves X without changing X's contacts, bep.babble.motion_clusters):
    a channel: its primitive is explored with X as the actor and executed by the channel's controller."""
    return r.kind == HC and bool(r.stats.get("dynamic", False)) and not r.carries


def primitive_relations(R: RelationSet, target: str) -> RelationSet:
    """The relations a primitive's contact modes are made of: every relation except the channels, plus its own
    channel (a channel through which another primitive moves X would only split its graph)."""
    from ..world.relations import channel_of
    own = channel_of(target)[1]
    return RelationSet([r for r in R if not is_channel(r) or r.name == own], R._rest_tilt)


def choose_primitives(R: RelationSet, probes: str) -> list[dict]:
    """Primitive targets, in exploration order (state and hand-object first), with the body kinds they act on;
    then the channel primitives (stage 0: they start from bodies at rest, not from a held body)."""
    from ..explore.primitive import DiscoveredPrimitive
    from ..world.objects import pair_world
    cand = {}
    for r in R:
        if not worthy(r) or is_channel(r):
            continue
        try:
            p = DiscoveredPrimitive(primitive_relations(R, r.name), r.name, probes=probes)
        except ValueError:                      # never held with a free subject in the probe
            continue
        r.stats["kinds"] = list(p.kinds)
        r.stats["roles"] = [list(k) for k in p.observed_roles()]
        cand[r.name] = r
    first = [n for n, r in cand.items() if r.kind in (SC, HC)]
    second = [n for n, r in cand.items() if r.kind == CC]
    out = [{"target": t, "kind": R.get(t).kind, "kinds": R.get(t).stats["kinds"], "roles": R.get(t).stats["roles"],
            "stage": i}
           for i, group in enumerate((first, second)) for t in group]
    for ch in [r for r in R if is_channel(r)]:
        for t in second:
            kx = R.get(t).stats["kinds"][0]
            try:
                p = DiscoveredPrimitive(primitive_relations(R, f"{t}~{ch.name}"), f"{t}~{ch.name}", kinds=(kx, "spot"),
                                        probes=probes)
            except ValueError:
                continue
            if not p.accepts_spot(pair_world(kx, "spot", body="float"), "A"):
                continue                        # the relation does not hold with X resting on the table
            out.append({"target": f"{t}~{ch.name}", "kind": CC, "kinds": [kx, "spot"], "roles": [[kx, "spot"]],
                        "stage": 0, "channel": ch.name})
    return out
