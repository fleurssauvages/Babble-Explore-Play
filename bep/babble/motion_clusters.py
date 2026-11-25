"""Clusters of the hand probes' motions (grasp and interaction probes mixed).

    python -m bep.babble.motion_clusters --run runs/pair_cubes_hand

For every recorded motion (bep.babble.hand_probe) and every free body X that the actor touches at the start or
at the end of the motion -- the actor is the hand together with every body it carries during the motion, so
that a body pushed by a carried body (a tool) is handled like a body pushed by the hand:

* precondition (at the motion's first contact change, else its start): how many actor bodies touch X (0, 1, 2 or more), whether two of them press
  from opposite sides, the centroid of their contacts on X (X's half extents, folded by X's yaw symmetry),
  whether X rested on something else at the start;
* effect: whether X's joint moved, whether X's support contact (anything other than the actor) was made / kept /
  lost, whether X followed the actor (from the first contact change of the motion to its end, X moved at least
  FOLLOW of the hand's displacement along it), X's
  displacement and its displacement relative to the hand (X's lengths).

Clusters are keyed by the effect only (``effect_key``): joint moved; support made; X follows the actor, keeping
its support; X follows the actor without support.  Motions without effect, or whose effect the actor did not
produce reproducibly (X knocked, falling, lagging), are left out.  The contacts are not part of the key: each
cluster's precondition is what it turns out to require.

The relations are read from the clusters (``relations_from_clusters``: grasp and place from the states their
motions end in, push as a channel from the states where it starts) and saved for exploration, with the probes
in bep.babble.probe's format (probes.npz: the rest state after each motion as a settled scene, the grasp
probe's lifts and the interaction probe's motions as events), which exploration reads for goal seeds and start
states.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import PROBE
from ..world.objects import TYPES, VIEW_S, pair_world, use_body
from ..world.geometry import quat_to_mat
from ..world.relations import (CC, CLOSING, HC, SIG_KEYS, Relation, RelationSet, cc_features, hc_features,
                               joint_norm, member_vector)
from .hand_probe import restore, roles
from .probe import _follows

SUPPORT = ("none", "kept", "lost", "made")


def _fold(p, sym):
    p = np.array(p, float)
    px, py = abs(p[0]), abs(p[1])
    if sym <= 0:
        p[0], p[1] = np.hypot(px, py), 0.0
    elif abs(sym - np.pi / 2) < 1e-6:
        p[0], p[1] = max(px, py), min(px, py)
    elif abs(sym - np.pi) < 1e-6:
        p[0], p[1] = px, py
    return p


MOVED = 0.3                    # X moved: beyond this share of its length; X follows: at least FOLLOW of the
FOLLOW = 0.5                   # actor's displacement along it (the actor moved beyond MOVED)


def _contact_list(env) -> list[tuple]:
    """Contacts on free bodies: (X, partner body, normal into X, point in X's frame)."""
    m, d, r = env.model, env.data, roles(env)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        n = np.array(c.frame[:3])                                        # points from geom1 to geom2
        for bx, partner, sign in ((b2, b1, 1.0), (b1, b2, -1.0)):
            if r[bx] in env.free and partner != bx:
                out.append((r[bx], partner, sign * n,
                            d.xmat[bx].reshape(3, 3).T @ (np.array(c.pos) - d.xpos[bx])))
    return out


def _snapshot(env) -> dict:
    d = env.data
    views = {x: env.pair_view(env.sensors(), x, "B" if x == "A" else "A") for x in env.free}
    return {"p": {x: d.xpos[env._hp_bid[x]].copy() for x in env.free}, "pE": env.hand_pose()[0],
            "w": env.gripper_width(), "q": {x: float(joint_norm(v, VIEW_S, "A")) for x, v in views.items()},
            "contacts": _contact_list(env), "views": views}


def _touch(snap, x, actors) -> tuple:
    """(actor bodies touching X, their normals into X, their contact points, X touches anything else)."""
    mine = [c for c in snap["contacts"] if c[0] == x]
    act = [c for c in mine if c[1] in actors]
    return ({c[1] for c in act}, np.array([c[2] for c in act]).reshape(-1, 3),
            np.array([c[3] for c in act]).reshape(-1, 3), any(c[1] not in actors for c in mine))


def _follows_hand(a, b, x, L) -> bool:
    dE, dX = b["pE"] - a["pE"], b["p"][x] - a["p"][x]
    nE = float(np.linalg.norm(dE))
    return bool(nE > MOVED * L and dX @ dE >= FOLLOW * nE * nE)


def features(env, m: dict) -> list[dict]:
    """One row per free body X that the actor touches at the start or the end of the motion.  The actor is the
    hand together with every body it carries during the motion (a body that follows the hand without any other
    support): a body pushed by a carried body is pushed by the actor, as by a tool."""
    r = roles(env)
    hand = {b for b, role in r.items() if role.startswith("E.")}
    restore(env, (m["s0"], m["c0"]))
    a = _snapshot(env)
    restore(env, (m["sc"], m["cc"]))                      # the first contact change: the interaction starts here
    c = _snapshot(env)
    restore(env, (m["s1"], m["c1"]))
    b = _snapshot(env)
    # carried: held by the hand and touching nothing else at the start, and following the hand (what it runs into
    # during the motion is what it acts on)
    carried = {o for o in env.free if _follows_hand(c, b, o, env.shape(o).length)
               and _touch(a, o, hand)[0] and not _touch(a, o, hand)[3]}
    rows = []
    for x in env.free:
        tools = {env._hp_bid[o] for o in carried if o != x}
        actors = hand | tools
        t0, tc, t1 = _touch(a, x, actors), _touch(c, x, actors), _touch(b, x, actors)
        if not (t0[0] or tc[0] or t1[0]):
            continue
        via = "hand" if (t0[0] | tc[0] | t1[0]) & hand else "carried body"
        shape = env.shape(x)
        L = shape.length
        bodies, normals, points, _ = tc                 # the precondition: at the first contact change
        sup0, sup1 = t0[3], t1[3]                       # the support transition: from the start to the end
        support = "none" if not (sup0 or sup1) else ("kept" if sup0 and sup1 else ("lost" if sup0 else "made"))
        rows.append(dict(
            x=x, via=via, probe=str(m["probe"]), kind=str(m["kind"]),
            contacts=min(len(bodies), 2), opposed=bool(len(normals) > 1 and (normals @ normals.T).min() < -0.5),
            where=_fold(points.mean(0) / np.asarray(shape.half, float), shape.sym) if len(points) else np.zeros(3),
            support=support, follows=_follows_hand(c, b, x, L), dq=b["q"][x] - a["q"][x],
            abs=(b["p"][x] - a["p"][x]) / L, rel=((b["p"][x] - b["pE"]) - (a["p"][x] - a["pE"])) / L,
            action=np.r_[(b["pE"] - a["pE"]) / L, (b["w"] - a["w"]) / L], views=(a["views"][x], b["views"][x]), contact_view=c["views"][x]))
    return rows


# ------------------------------------------------------------------ clustering: by effect
def effect_key(r) -> str | None:
    """What the motion did to X, coded by its effect only (the contacts before are the key's precondition, learned
    after): a joint moved; a support contact was made; X followed the actor keeping / without its support.
    Anything else is no effect, or an effect the actor did not produce reproducibly (X knocked, falling,
    lagging behind): None."""
    if abs(r["dq"]) > PROBE["joint_motion_min"]:
        return "joint moved"
    if r["support"] == "made":
        return "support made"
    if r["follows"]:
        return "follows, support kept" if r["support"] == "kept" else "follows, no support"
    return None


def cluster(rows, min_members=PROBE["min_members"]) -> list[dict]:
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        k = effect_key(r)
        if k is not None:
            groups[k].append(i)
    return [{"key": k, "members": np.array(v)} for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
            if len(v) >= min_members]


def describe(rows, cl) -> dict:
    """A cluster: its effect, and what held before (its precondition, as observed)."""
    m = cl["members"]
    get = lambda k: np.array([rows[i][k] for i in m])                                  # noqa: E731
    pct = lambda v: round(float(np.mean(v)), 2)                                        # noqa: E731
    return {"effect": cl["key"], "n": int(len(m)),
            "precondition": {"actor_contacts": dict(Counter(int(c) for c in get("contacts"))),
                             "opposed": pct(get("opposed")),
                             "on_a_support_before": pct([rows[i]["support"] in ("kept", "lost") for i in m]),
                             "where_median": np.round(np.median(get("where"), 0), 2).tolist()},
            "via": dict(Counter(rows[i]["via"] for i in m)),
            "probes": dict(Counter(rows[i]["probe"] for i in m)),
            "motions": dict(Counter(rows[i]["kind"] for i in m).most_common(4)),
            "abs_median": np.round(np.median(get("abs"), 0), 2).tolist(),
            "rel_norm_median": round(float(np.median(np.linalg.norm(get("rel"), axis=1))), 2)}


# ------------------------------------------------------------------ relations from the clusters
CONSTANT = 0.95                # a contact flag is part of a relation's signature when this share of its states agree


def _signature(f: dict, keys) -> dict:
    """The contact flags (SIG_KEYS) that (almost) all the states agree on."""
    sig = {}
    for k in keys:
        v = np.asarray(f[k], bool)
        share = float(v.mean()) if len(v) else 0.0
        if share >= CONSTANT or share <= 1 - CONSTANT:
            sig[k] = share >= CONSTANT
    return sig


def push_controller(rows: list[dict]) -> dict:
    """How the hand made X follow, from the push cluster: where the hand was relative to X at the first contact
    change, in the frame of the horizontal displacement X then made (x along it: negative = behind X, y to the
    left, z up; X's lengths), and how far X went per unit of the hand's displacement along it."""
    offs, gains, axes, widths, down = [], [], [], [], []
    for r in rows:
        v0, v1 = r["contact_view"], r["views"][1]
        dX, dE = v1[VIEW_S.A_pos] - v0[VIEW_S.A_pos], v1[VIEW_S.ee_pos] - v0[VIEW_S.ee_pos]
        L = float(2.0 * np.cbrt(np.prod(v0[VIEW_S.A_ext])))
        n = float(np.linalg.norm(dX[:2]))
        if n < 1e-6:
            continue
        u = np.r_[dX[:2] / n, 0.0]
        lat = np.cross([0.0, 0.0, 1.0], u)
        o = v0[VIEW_S.ee_pos] - v0[VIEW_S.A_pos]
        offs.append([o @ u / L, o @ lat / L, o[2] / L])
        RE = quat_to_mat(v0[VIEW_S.ee_quat])
        axes.append(abs(float(RE[:, CLOSING] @ u)))                       # closing axis along the push: 1
        widths.append(float(v0[VIEW_S.finger1] + v0[VIEW_S.finger2]) / L)
        down.append(float(-RE[2, 2]))                                      # approach axis pointing down: 1
        along = float(dE @ u)
        if along > 1e-6:
            gains.append(n / along)
    offs = np.array(offs).reshape(-1, 3)
    return {"offset": np.round(np.median(offs, 0), 3).tolist() if len(offs) else [-0.7, 0.0, 0.0],
            "offset_q25": np.round(np.quantile(offs, 0.25, axis=0), 3).tolist() if len(offs) else None,
            "offset_q75": np.round(np.quantile(offs, 0.75, axis=0), 3).tolist() if len(offs) else None,
            "gain": round(float(np.median(gains)), 3) if gains else 1.0, "n": int(len(offs)),
            "closing_along_push": round(float(np.median(axes)), 3) if axes else 1.0,
            "opening": round(float(np.median(widths)), 3) if widths else 1.0,
            "approach_down": round(float(np.median(down)), 3) if down else 1.0}


def relations_from_clusters(rows: list[dict], recs: list[dict]) -> RelationSet:
    """The relations of the three effect clusters, each read from the states its motions produce:

    * grasp (hand-object, carrying): the end states of 'follows, no support' motions (X held by the hand, nothing
      else touching it), with bilateral contact;
    * place (object-object): the end states of 'support made' motions in which X rests on the other body: X's
      position relative to it (one relation: the cluster defines it);
    * push (hand-object channel): the hand on X at the first contact change of 'follows, support kept' motions,
      in the frame of the motion (relations.motion_offset), with the signature its states agree on.

    How often each relation's motions had its effect where it held (precision) is recorded in its stats."""
    from .discover import lower_mode, membership_radius
    rels = []
    key = [effect_key(r) for r in rows]
    # static or dynamic: does the cluster's effect change X's contact relations (a support made, X held up with
    # no support), or only move X while every contact it had is kept?  Dynamic relations are explored forward.
    dynamic = {k: float(np.mean([r["support"] == "kept" for r, q in zip(rows, key) if q == k])) >= 0.5
               for k in set(key) if k is not None}
    # grasp
    held = [r["views"][1] for r, k in zip(rows, key) if k == "follows, no support" and r["via"] == "hand"]
    if len(held) >= PROBE["min_members"]:
        V = np.array(held)
        fh = hc_features(V, VIEW_S, rotation=False, body=True)
        sel = np.flatnonzero(fh["c_l"] & fh["c_r"])
    else:
        sel = np.empty(0, dtype=int)
    if len(sel) >= PROBE["min_members"]:
        sig = {"c_l": True, "c_r": True}
        g = Relation("g0", HC, sig, member_vector(HC, fh)[sel], membership_radius(fh["body"][sel]),
                     body_members=fh["body"][sel], stats={"contact_only": True})
        lifts = [r for r in rows if r["kind"] == "lift" and r["via"] == "hand"]
        fires = g.holds_f(hc_features(np.array([r["contact_view"] for r in lifts]), VIEW_S, rotation=False, body=True)) \
            if lifts else np.zeros(0, bool)
        carried = np.array([effect_key(r) == "follows, no support" for r in lifts], bool)
        prec = float(carried[fires].mean()) if np.any(fires) else 0.0
        g.stats.update(members=len(sel), verified_carry=True,
                       dynamic=dynamic.get("follows, no support", False),
                       lift_precision=round(prec, 3), precision_on_events=round(prec, 3))
        g.stats["from"] = "follows, no support (end states)"
        rels.append(g)
    # place
    placed = [r["views"][1] for r, k in zip(rows, key) if k == "support made"]
    if placed:
        V = np.array(placed)
        fc = cc_features(V, VIEW_S, "AB")
        sel = np.flatnonzero(fc["c_xy"])
        if len(sel) >= PROBE["min_members"]:
            m = sel
            tilt = float(np.maximum(fc["tilt_x"][m], fc["tilt_y"][m]).max())
            r = Relation("r0", CC, {"c_xy": True}, fc["p"][m], membership_radius(fc["p"][m]),
                         tilt_max=tilt + PROBE["tilt_margin"])
            r.stats = {"members": int(len(m)), "from": "support made (end states, X on the other body)",
                       "dynamic": dynamic.get("support made", False),
                       "p_median": np.round(np.median(fc["p"][m], 0), 3).tolist(),
                       "p_min": np.round(fc["p"][m].min(0), 3).tolist(), "p_max": np.round(fc["p"][m].max(0), 3).tolist()}
            rels.append(r)
    # push (channel)
    pushed = [r["contact_view"] for r, k in zip(rows, key) if k == "follows, support kept" and r["via"] == "hand"]
    if len(pushed) >= PROBE["min_members"]:
        V = np.array(pushed)
        fh = hc_features(V, VIEW_S, rotation=False)
        sel = np.flatnonzero(fh["c_any"])
        if len(sel) >= PROBE["min_members"]:
            sig = _signature({k: fh[k][sel] for k in SIG_KEYS[HC]}, SIG_KEYS[HC])
            ch = Relation(f"g{sum(q.kind == HC for q in rels)}", HC, sig, fh["motion"][sel],
                          membership_radius(fh["motion"][sel]), stats={"motion_frame": True})
            hand_rows = [r for r in rows if r["via"] == "hand" and r["probe"] == "interact"]
            fires = ch.holds_f(hc_features(np.array([r["contact_view"] for r in hand_rows]), VIEW_S, rotation=False))
            pushed_ok = np.array([effect_key(r) == "follows, support kept" for r in hand_rows], bool)
            ch.stats.update(members=int(len(sel)), verified_carry=False, lift_precision=0.0,
                            dynamic=dynamic.get("follows, support kept", True),
                            precision_on_events=round(float(pushed_ok[fires].mean()) if np.any(fires) else 0.0, 3),
                            offset_median=np.round(np.median(fh["motion"][sel], 0), 3).tolist(),
                            **{"from": "follows, support kept (contact states)"})
            ch.stats["push"] = push_controller([r for r, k in zip(rows, key)
                                                if k == "follows, support kept" and r["via"] == "hand"])
            rels.append(ch)
    # resting tilt: bodies at rest touching nothing but the table (as bep.babble.discover)
    s2 = np.array([r["s2"] for r in recs])
    f = cc_features(s2, VIEW_S, "AB")
    alone = (~f["c_xy"]) & (s2[:, VIEW_S.lf_A] <= 0) & (s2[:, VIEW_S.rf_A] <= 0) & (s2[:, VIEW_S.hand_A] <= 0)
    tilts = np.concatenate([cc_features(s2[alone], VIEW_S, o)["tilt_x"] for o in ("AB", "BA")])
    rest = lower_mode(tilts) if len(tilts) else tilts
    rest_tilt = float(np.quantile(rest, 0.995) + PROBE["tilt_margin"]) if len(rest) else 0.05
    return RelationSet(rels, rest_tilt)


# ------------------------------------------------------------------ bep.babble.probe's format
def probe_records(env, f, rows_by_motion) -> list[dict]:
    """The rest state after each motion is a settled scene (s2, s1 its start); the grasp probe's lifts are lift
    events (kind 0 before, kind 2 in flight when X came along), the interaction probe's motions that touched X
    are motion events (kind 1, X followed the hand: bep.babble.probe._follows)."""
    cube = TYPES.index("cube")
    recs = []
    for i in range(len(f["kind"])):
        restore(env, (f["s0"][i], f["c0"][i]))
        s1 = env.pair_view(env.sensors(), "A", "B")
        restore(env, (f["s_rest"][i], f["c1"][i]))
        rec = {"s1": s1, "s2": env.pair_view(env.sensors(), "A", "B"), "attached": False, "moved": True,
               "pair": [cube, cube], "events": []}
        for r in rows_by_motion.get(i, []):
            if r["via"] != "hand":
                continue
            v0, v1 = r["views"]
            dE, dX = v1[VIEW_S.ee_pos] - v0[VIEW_S.ee_pos], v1[VIEW_S.A_pos] - v0[VIEW_S.A_pos]
            segment = PROBE["segment"] * env.shape(r["x"]).length
            if str(f["kind"][i]) == "lift":
                followed = _follows(dX, dE, segment)
                rec["events"].append((v0, bool(followed), 0))
                if followed:
                    rec["events"].append((v1, bool(v1[VIEW_S.lf_A] > 0 and v1[VIEW_S.rf_A] > 0), 2))
            elif str(f["probe"][i]).startswith("interact"):
                rec["events"].append((v1, bool(_follows(dX, dE, segment)), 1))
        recs.append(rec)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/pair_cubes_hand")
    args = ap.parse_args()
    out = Path(args.run)
    use_body("float")
    env = pair_world("cube", "cube")
    roles(env)
    f = dict(np.load(out / "motions.npz"))
    rows, by_motion = [], defaultdict(list)
    for i in range(len(f["kind"])):
        for r in features(env, {k: f[k][i] for k in ("s0", "c0", "sc", "cc", "s1", "c1", "key0", "key1", "probe", "kind")}):
            rows.append(r)
            by_motion[i].append(r)
    clusters = [describe(rows, c) for c in cluster(rows)]
    report = {"motions": int(len(f["kind"])), "rows": len(rows),
              "effects": dict(Counter(str(effect_key(r)) for r in rows).most_common()),
              "clusters": clusters}
    (out / "motion_clusters.json").write_text(json.dumps(report, indent=1))
    recs = probe_records(env, f, by_motion)
    ev = [(i, e) for i, r in enumerate(recs) for e in r["events"]]
    np.savez_compressed(out / "probes.npz", s1=np.array([r["s1"] for r in recs]), s2=np.array([r["s2"] for r in recs]),
                        attached=np.zeros(len(recs), bool), moved=np.ones(len(recs), bool),
                        pair=np.array([r["pair"] for r in recs]), types=np.array(TYPES),
                        focus=np.zeros(len(recs), bool), ev_s=np.array([e[0] for _, e in ev]),
                        ev_follow=np.array([e[1] for _, e in ev], bool), ev_kind=np.array([e[2] for _, e in ev]),
                        ev_probe=np.array([i for i, _ in ev]))
    rels = relations_from_clusters(rows, recs)
    rels.save(out / "relations")
    rep = {"relations": [r.to_json() for r in rels], "rest_tilt": rels.rest_tilt}
    (out / "probe_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=1))
    for c in clusters:
        print(json.dumps(c))
    for r in rep["relations"]:
        print(json.dumps({k: r[k] for k in ("name", "kind", "signature", "n_members", "stats")}, default=float))


if __name__ == "__main__":
    main()
