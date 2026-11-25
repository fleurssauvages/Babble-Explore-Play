"""Group probe outcomes by contact signatures and observed motion.

Geometry from each group supplies examples for relation goals and exploration.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from ..config import PROBE
from ..world.geometry import characteristic_length
from ..world.objects import TYPES, VIEW_S
from ..world.relations import (CC, HC, SC, Relation, RelationSet, cc_features, hc_features, member_vector,
                               sc_features)


def lower_mode(x: np.ndarray, bins: int = 256) -> np.ndarray:
    """Values below Otsu's threshold (resting objects are upright or lying on a side: keep the
    undisturbed mode)."""
    h, edges = np.histogram(x, bins=bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    t = c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]
    return x[x <= t]


def membership_radius(X: np.ndarray) -> float:
    if len(X) < 3:
        return 0.1
    d, _ = cKDTree(X).query(X, k=2)
    return float(max(0.08, 1.5 * np.quantile(d[:, 1], 0.95)))


def relation_keys(rels) -> list[tuple]:
    """Discrete grouping keys used to compare discoveries across probe batches."""
    return [(r.kind, tuple(sorted(r.stats.get("cluster_contacts", r.signature).items())),
             r.stats.get("joint_direction", 0)) for r in rels]


def same_relations(a, b) -> bool:
    return Counter(a) == Counter(b)


def discover(records: list[dict], S, *, min_members: int = PROBE["min_members"]):
    s2 = np.array([r["s2"] for r in records])
    s1 = np.array([r["s1"] for r in records])
    moved = np.array([r["moved"] for r in records])
    ev = [e for r in records for e in r["events"]]
    ev_probe = np.repeat(np.arange(len(records)), [len(r["events"]) for r in records])
    E = np.array([e[0] for e in ev]) if ev else np.zeros((0, s2.shape[1]))
    follow = np.array([e[1] for e in ev], bool)
    lift = np.array([e[2] in (0, 2) for e in ev], bool)       # both ends of the lift probe
    relations, report = [], {"n_probes": len(records), "events": len(ev), "followed": int(follow.sum())}

    # ---------------- object-object relations (hand released and gone, scene at rest)
    hand_free = moved & (s2[:, S.lf_A] <= 0) & (s2[:, S.rf_A] <= 0) & (s2[:, S.hand_A] <= 0) & (s2[:, S.hand_B] <= 0)
    v_rest = PROBE["rest_speed"]
    rest = (np.linalg.norm(s2[:, S.A_vel], axis=1) < v_rest * characteristic_length(s2[:, S.A_ext])) & \
           (np.linalg.norm(s2[:, S.B_vel], axis=1) < v_rest * characteristic_length(s2[:, S.B_ext]))
    ok = hand_free & rest
    rows = []                                                   # (probe, order, features)
    for order in ("AB", "BA"):
        f = cc_features(s2, S, order)
        for i in np.nonzero(ok & f["c_xy"])[0]:
            rows.append((i, order, f["p"][i], bool(f["c_xt"][i]), bool(f["c_yt"][i]),
                         float(f["tilt_x"][i]), float(f["tilt_y"][i])))
    report["cc_contact_rows"] = len(rows)
    clusters = []
    for sig in [(a, b) for a in (False, True) for b in (False, True)]:
        idx = [k for k, r in enumerate(rows) if (r[3], r[4]) == sig]
        if len(idx) >= min_members:
            clusters.append({"sig": sig, "rows": idx, "keys": {(rows[k][0], rows[k][1]) for k in idx}})
    dA = np.linalg.norm(s2[:, S.A_pos] - s1[:, S.A_pos], axis=1) / characteristic_length(s2[:, S.A_ext])
    dB = np.linalg.norm(s2[:, S.B_pos] - s1[:, S.B_pos], axis=1) / characteristic_length(s2[:, S.B_ext])

    def mobility(c):
        """Prefer a free, ungrounded, demonstrably moved subject when choosing a relation's direction."""
        parts = [s2[p, S.A_art] > 0 if o == "AB" else s2[p, S.B_art] > 0 for p, o in c["keys"]]
        dx = [dA[p] - dB[p] if o == "AB" else dB[p] - dA[p] for p, o in c["keys"]]
        xt, yt = c["sig"]
        return (float(np.mean(parts)), int(xt) - int(yt), -float(np.median(dx)))

    # merge mirrors: same probes seen with roles swapped
    flip = {"AB": "BA", "BA": "AB"}
    keep = []
    for ci, c in enumerate(clusters):
        mirror = None
        for cj, c2 in enumerate(clusters):
            if cj != ci and len({(p, flip[o]) for p, o in c["keys"]} & c2["keys"]) > 0.5 * len(c["keys"]):
                mirror = cj
        if mirror is not None:
            mine, other = (*mobility(c), ci), (*mobility(clusters[mirror]), mirror)
            if mine > other:
                continue
        keep.append(c)
    for c in keep:
        X = np.array([rows[k][2] for k in c["rows"]])
        tilt = max(max(rows[k][5], rows[k][6]) for k in c["rows"])
        name = f"r{len(relations)}"
        relations.append(Relation(
            name, CC, {"c_xy": True, "c_xt": c["sig"][0], "c_yt": c["sig"][1]}, X, membership_radius(X),
            tilt_max=float(tilt + PROBE["tilt_margin"]),
            stats={"members": len(X), "probes": len({p for p, _ in c["keys"]}),
                   "cluster_contacts": {"c_xy": True, "c_xt": c["sig"][0], "c_yt": c["sig"][1]},
                   "p_median": np.round(np.median(X, 0), 4).tolist(),
                   "p_min": np.round(X.min(0), 4).tolist(), "p_max": np.round(X.max(0), 4).tolist()}))

    # ---------------- hand-object relations (events in which the body followed the hand)
    fh = hc_features(E, S, rotation=False, body=True)
    yaw_members = member_vector(HC, fh)
    bilateral = fh["c_l"] & fh["c_r"]
    groups = [(True, {"c_l": True, "c_r": True}, follow & lift & bilateral)]
    for sig in [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]:
        if any(sig) and not (sig[0] and sig[1]):
            signature = dict(zip(("c_l", "c_r", "c_h"), sig))
            groups.append((False, signature, follow & np.logical_and.reduce([fh[k] == v for k, v in signature.items()])))
    for verified_carry, signature, examples in groups:
        sel = np.flatnonzero(examples)
        if len(sel) < min_members:
            continue
        X = (fh["body"] if verified_carry else yaw_members)[sel]
        rel = Relation(f"g{sum(r.kind == HC for r in relations)}", HC, signature, yaw_members[sel],
                       membership_radius(X), body_members=X if verified_carry else None,
                       stats={"contact_only": True})
        fires = rel.holds_f(fh)
        prec = float(np.mean(follow[fires]))
        lift_fires = fires & lift
        lp = float(np.mean(follow[lift_fires])) if np.any(lift_fires) else 0.0
        if (lp if verified_carry else prec) < PROBE["min_precision"]:
            continue
        lift_probes = np.unique(ev_probe[sel[lift[sel]]])
        subjects = Counter(TYPES[int(records[i]["pair"][0])] for i in lift_probes)
        rel.stats.update(members=len(sel), verified_carry=verified_carry,
                         lift_probes=len(lift_probes), lift_subjects=dict(subjects),
                         precision_on_events=round(prec, 3), lift_precision=round(lp, 3),
                         p_median=np.round(np.median(X[:, :3], 0), 4).tolist())
        relations.append(rel)
    # ---------------- joint endpoints grouped by the direction of the observed motion
    untouched = (s2[:, S.lf_A] <= 0) & (s2[:, S.rf_A] <= 0)
    joint_examples = {-1: [], 1: []}
    for role in ("A", "B"):
        before, after = sc_features(s1, S, role), sc_features(s2, S, role)
        change = after["q"] - before["q"]
        for direction, values in joint_examples.items():
            moved_joint = direction * change > PROBE["joint_motion_min"]
            values.extend(after["q"][moved & untouched & after["art"] & moved_joint])
    for direction, values in joint_examples.items():
        if len(values) >= min_members:
            members = np.asarray(values)[:, None]
            name = f"s{sum(r.kind == SC for r in relations)}"
            relations.append(Relation(name, SC, {"art": True}, members, membership_radius(members),
                                      stats={"members": len(members), "joint_direction": direction}))
    # ---------------- minimal signatures (Occam): drop every contact flag, except the
    # defining contact of the cluster, that is not needed to separate it from all other
    # probe states.  Flags that were merely constant in the probe world (e.g. "Y touches
    # the table" when Y was always on the table) would otherwise not transfer.
    fc = {o: cc_features(s2[ok], S, o) for o in ("AB", "BA")}
    for rel in (r for r in relations if r.kind == CC):
        feats = list(fc.values())
        base = [rel.holds_f(f) for f in feats]
        dropped = []
        for flag in list(rel.signature):
            if flag == "c_xy":
                continue
            val = rel.signature.pop(flag)
            if all(np.array_equal(rel.holds_f(f), b) for f, b in zip(feats, base)):
                dropped.append(flag)
            else:
                rel.signature[flag] = val
        rel.stats["dropped_flags"] = dropped
    report["merged"] = []
    # ---------------- resting tilt: objects at rest and touching nothing but the table
    alone = ok & ~cc_features(s2, S, "AB")["c_xy"]
    tilts = np.concatenate([cc_features(s2[alone], S, o)["tilt_x"] for o in ("AB", "BA")])
    rest = lower_mode(tilts) if len(tilts) else tilts
    rest = rest if len(rest) else tilts                    # all alike (e.g. nothing ever tips): all at rest
    rest_tilt = float(np.quantile(rest, 0.995) + PROBE["tilt_margin"]) if len(rest) else 0.05
    report["rest_tilt"] = rest_tilt
    report["relations"] = [r.to_json() for r in relations]
    return RelationSet(relations, rest_tilt), report


def load_records(f) -> list[dict]:
    recs = [{"s1": a, "s2": b, "attached": bool(c), "moved": bool(d), "pair": e, "events": []}
            for a, b, c, d, e in zip(f["s1"], f["s2"], f["attached"], f["moved"], f["pair"])]
    for s, fl, k, i in zip(f["ev_s"], f["ev_follow"], f["ev_kind"], f["ev_probe"]):
        recs[int(i)]["events"].append((s, bool(fl), int(k)))
    return recs


def created_relations(R, rec) -> set[str]:
    """Relations a probe brought about: object relations holding at rest after it but not before, and
    hand relations holding in an event in which the body followed the hand."""
    h1, h2 = R.holds_all(rec["s1"][None], VIEW_S), R.holds_all(rec["s2"][None], VIEW_S)
    out = {n.split("@")[0] for n in h2 if n not in {r.name for r in R.hc} and h2[n][0] and not h1[n][0]}
    follow = [e[0] for e in rec["events"] if e[1]]
    if follow and R.hc:
        he = R.holds_all(np.array(follow), VIEW_S)
        out |= {r.name for r in R.hc if he[r.name].any()}
    return out
