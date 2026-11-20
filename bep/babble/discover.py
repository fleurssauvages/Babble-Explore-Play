"""Relations from probe data: clustering what holds at rest and what follows the hand.

Discovery (no relation is named or specified in advance):

* object-object: in S2 (hand gone, scene at rest), ordered pairs (X, Y) in contact are grouped by
  contact signature (X-table, Y-table) and clustered in p_X^Y (object units).  Each dense cluster is a
  candidate relation; mirror clusters (the same probes seen as (Y, X)) are merged.
* hand-object: in the probe events, configurations whose body then moved with the hand are clustered
  in (pinch point, closing axis, opening), i.e. by where the fingers touch, per finger-contact signature.
* object-state: joint values moving parts rest at unusually often.

Because features are in object units, a cluster can span every pair of shapes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..config import PROBE
from ..world.objects import VIEW_S
from ..world.relations import CC, HC, SC, Relation, RelationSet, cc_features, hc_features, member_vector, sc_features


def state_clusters(q: np.ndarray, bins: int = 40) -> list[np.ndarray]:
    """Joint values a moving part rests at unusually often: contiguous histogram bins above
    PROBE['state_peak'] times the median bin count."""
    if len(q) < PROBE["min_members"]:
        return []
    h, edges = np.histogram(np.clip(q, 0, 1), bins=bins, range=(0, 1))
    ref = max(float(np.median(h)), 1.0)
    hot = h > PROBE["state_peak"] * ref
    out, k = [], 0
    while k < bins:
        if hot[k]:
            j = k
            while j + 1 < bins and hot[j + 1]:
                j += 1
            members = q[(q >= edges[k]) & (q <= edges[j + 1])]
            if len(members) >= PROBE["min_members"]:
                out.append(members)
            k = j + 1
        else:
            k += 1
    return out


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


def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    labels = np.full(len(X), -1)
    if len(X) == 0:
        return labels
    tree = cKDTree(X)
    nbrs = tree.query_ball_point(X, eps)
    core = np.array([len(n) >= min_samples for n in nbrs])
    c = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = c
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in nbrs[j]:
                if labels[k] == -1:
                    labels[k] = c
                    stack.append(k)
        c += 1
    return labels


def membership_radius(X: np.ndarray) -> float:
    if len(X) < 3:
        return 0.1
    d, _ = cKDTree(X).query(X, k=2)
    return float(max(0.08, 1.5 * np.quantile(d[:, 1], 0.95)))


def relation_keys(rels) -> list[tuple]:
    """(kind, signature, centre) of each relation, to compare discoveries across batches."""
    return [(r.kind, tuple(sorted(r.signature.items())), np.atleast_1d(r.center)) for r in rels]


def same_relations(a, b, tol: float = 0.5) -> bool:
    """Same kinds, each centre within tol (object units) of a distinct counterpart.  Signatures are not
    compared: their minimal form (redundant contact flags dropped) is sensitive to a few probes."""
    if len(a) != len(b):
        return False
    used = set()
    for k, s, c in a:
        j = next((j for j, (k2, s2, c2) in enumerate(b) if j not in used and k2 == k
                  and len(c2) == len(c) and np.abs(c2 - c).sum() < tol), None)
        if j is None:
            return False
        used.add(j)
    return True


def discover(records: list[dict], S, *, eps: float = PROBE["dbscan_eps"], min_samples: int = PROBE["dbscan_min"],
             min_members: int = PROBE["min_members"]):
    s2 = np.array([r["s2"] for r in records])
    moved = np.array([r["moved"] for r in records])
    ev = [e for r in records for e in r["events"]]
    E = np.array([e[0] for e in ev]) if ev else np.zeros((0, s2.shape[1]))
    follow = np.array([e[1] for e in ev], bool)
    lift = np.array([e[2] == 0 for e in ev], bool)            # grasp-probe events (close and lift)
    relations, report = [], {"n_probes": len(records), "events": len(ev), "followed": int(follow.sum())}

    # ---------------- object-object relations (hand released and gone, scene at rest)
    hand_free = moved & (s2[:, S.lf_A] <= 0) & (s2[:, S.rf_A] <= 0) & (s2[:, S.hand_A] <= 0) & (s2[:, S.hand_B] <= 0)
    v_rest = PROBE["rest_speed"]
    rest = (np.linalg.norm(s2[:, S.A_vel], axis=1) < v_rest) & (np.linalg.norm(s2[:, S.B_vel], axis=1) < v_rest)
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
        if not idx:
            continue
        X = np.array([rows[k][2] for k in idx])
        lab = dbscan(X, eps, max(min_samples, int(PROBE["dbscan_frac"] * len(X))))
        for c in range(lab.max() + 1):
            mem = [idx[k] for k in np.nonzero(lab == c)[0]]
            if len(mem) >= max(min_members, PROBE["min_frac"] * len(idx)):
                clusters.append({"sig": sig, "rows": mem,
                                 "keys": {(rows[k][0], rows[k][1]) for k in mem}})
    def x_part(c):
        """Fraction of a cluster's rows whose subject X is a moving part."""
        return float(np.mean([s2[p, S.A_art] > 0 if o == "AB" else s2[p, S.B_art] > 0 for p, o in c["keys"]]))

    # merge mirrors: same probes seen with roles swapped
    flip = {"AB": "BA", "BA": "AB"}
    keep = []
    for ci, c in enumerate(clusters):
        mirror = None
        for cj, c2 in enumerate(clusters):
            if cj != ci and len({(p, flip[o]) for p, o in c["keys"]} & c2["keys"]) > 0.5 * len(c["keys"]):
                mirror = cj
        if mirror is not None:
            # keep the orientation whose subject X is a free body (a primitive can place it); between two
            # free bodies, the one with X above
            mine, other = (x_part(c), -np.median([rows[k][2][2] for k in c["rows"]]), ci), \
                (x_part(clusters[mirror]), -np.median([rows[k][2][2] for k in clusters[mirror]["rows"]]), mirror)
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
                   "p_median": np.round(np.median(X, 0), 4).tolist(),
                   "p_min": np.round(X.min(0), 4).tolist(), "p_max": np.round(X.max(0), 4).tolist()}))

    # ---------------- hand-object relations (events in which the body followed the hand)
    fh = hc_features(E, S)
    for sig in [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)]:
        if not any(sig):
            continue                          # the hand moves a body only by touching it
        sel = np.nonzero(follow & (fh["c_l"] == sig[0]) & (fh["c_r"] == sig[1]) & (fh["c_h"] == sig[2]))[0]
        if len(sel) < min_members:
            continue
        X = member_vector(HC, fh)[sel]                       # pinch point, closing axis, opening: the contacts
        lab = dbscan(X, 2 * eps, max(min_samples, int(PROBE["dbscan_frac"] * len(X))))
        for c in range(lab.max() + 1):
            mem = sel[lab == c]
            if len(mem) < max(min_members, PROBE["min_frac"] * len(sel)):
                continue
            Xm = X[lab == c]
            rel = Relation(f"g{sum(r.kind == HC for r in relations)}", HC,
                           {"c_l": sig[0], "c_r": sig[1], "c_h": sig[2]}, Xm, membership_radius(Xm))
            # purity on the events: how often does the body follow when the classifier fires?
            fires = rel.holds_f(fh)
            prec = float(np.mean(follow[fires])) if np.any(fires) else float("nan")
            if not prec >= PROBE["min_precision"]:
                continue                      # mostly fires where the body did not follow: not a relation
            lp = float(np.mean(follow[fires & lift])) if np.any(fires & lift) else 0.0
            rel.stats = {"members": int(len(mem)), "precision_on_events": round(prec, 3),
                         "lift_precision": round(lp, 3),    # the body comes along when the hand lifts: carrying
                         "p_median": np.round(np.median(Xm[:, :3], 0), 4).tolist(),
                         "axis_median": np.round(np.median(Xm[:, 3:6], 0), 3).tolist(),
                         "width_range": [round(float(Xm[:, -1].min()), 4), round(float(Xm[:, -1].max()), 4)]}
            relations.append(rel)
    # ---------------- object-state relations (joint values moving parts rest at)
    q_rest = []
    for role in ("A", "B"):
        fsc = sc_features(s2, S, role)
        sel = moved & fsc["art"] & (s2[:, S.lf_A] <= 0) & (s2[:, S.rf_A] <= 0)
        q_rest.append(fsc["q"][sel])
    for members in state_clusters(np.concatenate(q_rest) if q_rest else np.zeros(0)):
        rel = Relation(f"s{len([r for r in relations if r.kind == SC])}", SC, {"art": True}, members[:, None],
                       max(0.02, membership_radius(members[:, None])))
        rel.stats = {"members": int(len(members)), "q_range": [round(float(members.min()), 3),
                                                                round(float(members.max()), 3)]}
        relations.append(rel)
    # ---------------- minimal signatures (Occam): drop every contact flag, except the
    # defining contact of the cluster, that is not needed to separate it from all other
    # probe states.  Flags that were merely constant in the probe world (e.g. "Y touches
    # the table" when Y was always on the table) would otherwise not transfer.
    fc = {o: cc_features(s2[ok], S, o) for o in ("AB", "BA")}
    for rel in [r for r in relations if r.kind != SC]:
        defining = "c_xy" if rel.kind == CC else None
        feats = list(fc.values()) if rel.kind == CC else [fh]
        base = [rel.holds_f(f) for f in feats]
        dropped = []
        for flag in list(rel.signature):
            if flag == defining:
                continue
            val = rel.signature.pop(flag)
            if all(np.array_equal(rel.holds_f(f), b) for f, b in zip(feats, base)):
                dropped.append(flag)
            else:
                rel.signature[flag] = val
        rel.stats["dropped_flags"] = dropped
    # ---------------- duplicates: two relations of a kind that fire on mostly the same probe states are one
    # relation (e.g. a grasp split by the clustering into two clusters); merged, their members pooled
    relations = merge_duplicates(relations, {CC: list(fc.values()), HC: [fh]}, follow, lift)
    report["merged"] = [r.name for r in relations if r.stats.get("merged_from")]
    # ---------------- resting tilt: objects at rest and touching nothing but the table
    alone = ok & ~cc_features(s2, S, "AB")["c_xy"]
    tilts = np.concatenate([cc_features(s2[alone], S, o)["tilt_x"] for o in ("AB", "BA")])
    rest = lower_mode(tilts) if len(tilts) else tilts
    rest = rest if len(rest) else tilts                    # all alike (e.g. nothing ever tips): all at rest
    rest_tilt = float(np.quantile(rest, 0.995) + PROBE["tilt_margin"]) if len(rest) else 0.05
    report["rest_tilt"] = rest_tilt
    report["relations"] = [r.to_json() for r in relations]
    return RelationSet(relations, rest_tilt), report


def merge_duplicates(relations, feats: dict, follow=None, lift=None, overlap: float = 0.5) -> list:
    """Merge relations of the same kind whose firing sets on the probe states overlap by more than
    ``overlap`` of the smaller one; the merged relation keeps the flags both agree on, pools the members
    and takes the larger radius.  Relations are renumbered per kind afterwards."""
    rels = list(relations)
    fires = {id(r): np.concatenate([r.holds_f(f).ravel() for f in feats[r.kind]]) for r in rels if r.kind in feats}
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(rels):
            for b in rels[i + 1:]:
                if a.kind != b.kind or a.kind not in feats or a.members.shape[1] != b.members.shape[1]:
                    continue
                fa, fb = fires[id(a)], fires[id(b)]
                small = min(fa.sum(), fb.sum())
                if small == 0 or (fa & fb).sum() <= overlap * small:
                    continue
                sig = {k: v for k, v in a.signature.items() if b.signature.get(k, None) == v}
                m = Relation(a.name, a.kind, sig, np.concatenate([a.members, b.members]), max(a.delta, b.delta),
                             tilt_max=max(a.tilt_max, b.tilt_max))
                m.stats = {**a.stats, "members": int(len(m.members)),
                           "merged_from": a.stats.get("merged_from", [a.name]) + b.stats.get("merged_from", [b.name])}
                if a.kind == HC and follow is not None:
                    fh = m.holds_f(feats[HC][0])
                    m.stats["precision_on_events"] = round(float(np.mean(follow[fh])), 3) if np.any(fh) else 0.0
                    m.stats["lift_precision"] = round(float(np.mean(follow[fh & lift])), 3) if np.any(fh & lift) else 0.0
                fires[id(m)] = np.concatenate([m.holds_f(f).ravel() for f in feats[m.kind]])
                rels = [m if r is a else r for r in rels if r is not b]
                merged = True
                break
            if merged:
                break
    prefix = {CC: "r", HC: "g", SC: "s"}
    count: dict = {}
    for r in rels:
        r.name = f"{prefix[r.kind]}{count.setdefault(r.kind, 0)}"
        count[r.kind] += 1
    return rels


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
