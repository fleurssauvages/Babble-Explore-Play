"""Events: what a motor command changed, as discrete indicators only, and the relations they make.

An *event* is the change between two settled states (bep.babble.probe): the settled state before a command
and the settled state after it.  It is read for every ordered pair of bodies (X, Y) through the pair views
(bep.world.objects.ObjWorld.pair_view), as a key made of discrete indicators only:

* contact flags, before -> after (keys are factored: one key per factor of the state the event changed): how many fingers touch X (0, 1, 2: the gripper's two fingers are
  interchangeable), the hand touches X, X touches Y, X touches the table, Y touches the table, the hand
  touches Y.  The key holds the flags that changed, with their new value;
* how X moved during the command: it followed the hand, it tipped, it moved otherwise, or it stayed still;
* whether a moving part's joint moved, and which way.

Keys are not chosen in advance: every key observed often enough is a candidate.  A candidate is kept when
the hand caused it (the hand touched X or Y around the command) and it is reproducible: in some
precondition context (the settled flags before the command and a discrete class of the command) raises its
rate well above its base rate (``EVENTS['lift']``), as infants keep the effects that are contingent on their
own actions.

Relations are read from the kept keys by which pair of roles the change concerns:

* X now touches Y (object-object): once the hand can cause that contact (the key is kept), the relation is
  read from every scene at rest after the hand let go and withdrew in which X touches Y (what stays when
  let go), grouped by the other contact flags at rest (X on the table or not, Y on the table or not);
* X left its support and followed the hand, held by both fingers (hand-object, carrying): read from the
  settled states after the event;
* a joint moved (object-state): the joint values reached, per direction.

Mirror keys (the same events seen with X and Y exchanged) are merged, keeping the order in which X is the
body that moved.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from ..config import EVENTS, PROBE
from ..world.geometry import characteristic_length, quat_to_mat
from ..world.objects import TYPES, VIEW_S, pair_world
from ..world.relations import CC, HC, SC, Relation, RelationSet, cc_features, hc_features, member_vector, tilt_of

FLAGS = ("fingers", "hand", "xy", "xt", "yt", "hy")
ORDERS = (("A", "B"), ("B", "A"))


# ------------------------------------------------------------------ helpers
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


# ------------------------------------------------------------------ indicators
def flags(v: np.ndarray) -> np.ndarray:
    """(..., 6) integer contact flags of pair views (FLAGS order)."""
    S = VIEW_S
    fingers = (v[..., S.lf_A] > 0).astype(int) + (v[..., S.rf_A] > 0).astype(int)
    return np.stack([fingers, v[..., S.hand_A] > 0, v[..., S.A_B] > 0, v[..., S.A_table] > 0,
                     v[..., S.B_table] > 0, v[..., S.hand_B] > 0], axis=-1).astype(int)


def _tilt(v):
    return tilt_of(quat_to_mat(v[..., VIEW_S.A_quat]))


def motion(v0, v1, s0, s1) -> np.ndarray:
    """How X moved during a command (start view v0, end view v1; settled views s0, s1 for the tilt):
    0 still, 1 moved, 2 tipped, 3 followed the hand."""
    S = VIEW_S
    L = characteristic_length(v0[..., S.A_ext])
    dE = v1[..., S.ee_pos] - v0[..., S.ee_pos]
    dX = v1[..., S.A_pos] - v0[..., S.A_pos]
    nE = np.linalg.norm(dE, axis=-1)
    follows = (nE > EVENTS["moved"] * L) & ((dX * dE).sum(-1) >= EVENTS["follow_frac"] * nE * nE)
    moved = np.linalg.norm(dX, axis=-1) > EVENTS["moved"] * L
    tipped = np.abs(_tilt(s1) - _tilt(s0)) > EVENTS["tipped"]
    return np.where(follows, 3, np.where(tipped, 2, np.where(moved, 1, 0)))


MOTION = ("still", "moved", "tipped", "follows")


def joint_dir(s0, s1) -> np.ndarray:
    from ..world.relations import joint_norm
    dq = joint_norm(s1, VIEW_S, "A") - joint_norm(s0, VIEW_S, "A")
    art = s1[..., VIEW_S.A_art] > 0
    return np.where(art & (dq > PROBE["joint_motion_min"]), 1, np.where(art & (dq < -PROBE["joint_motion_min"]), -1, 0))


def key_of(f0, f1, mo, jd) -> str | None:
    """The discrete key of one event (None: nothing changed and X stayed still)."""
    changed = [f"{FLAGS[i]}={int(f1[i])}" for i in range(len(FLAGS)) if f0[i] != f1[i]]
    if not changed and mo == 0 and jd == 0:
        return None
    return "|".join(changed + [MOTION[mo]] + ([f"joint{'+' if jd > 0 else '-'}"] if jd else []))


def factor_keys(f0, f1, mo, jd) -> list[str]:
    """The event's effect on each factor of the state it changed (factored effects): X-Y contact, X's support
    (with how X moved), the grip on X, Y's support, the hand on Y, a joint; X moving without any of its
    contacts changing (pushed, tipped) is an effect of its own."""
    out = []
    if f0[2] != f1[2]:
        out.append(f"xy={int(f1[2])}")
    if f0[3] != f1[3]:
        out.append(f"xt={int(f1[3])}|{MOTION[mo]}")
    if f0[0] != f1[0] or f0[1] != f1[1]:
        out.append(f"fingers={int(f1[0])}|hand={int(f1[1])}")
    if f0[4] != f1[4]:
        out.append(f"yt={int(f1[4])}")
    if f0[5] != f1[5]:
        out.append(f"hy={int(f1[5])}")
    if jd:
        out.append(f"joint{'+' if jd > 0 else '-'}")
    if mo and f0[2] == f1[2] and f0[3] == f1[3]:
        out.append(f"X:{MOTION[mo]}")
    return out


def cell_of(f, p) -> str:
    """A Go-Explore cell of a settled state (for the novelty archive only, never a symbol): its contact flags
    and X's position relative to Y in object units, binned by one unit horizontally (folded: |x|, |y|) and
    half a unit vertically, clipped to +-3 units."""
    b = np.clip(np.round([abs(p[0]), abs(p[1]), 2 * p[2]]), -3, 3).astype(int)
    return "".join(map(str, f)) + ":" + ",".join(map(str, b))


def command_class(kind: str, dE) -> str:
    """A discrete class of the motor command (the action half of a precondition context)."""
    if kind != "move":
        return kind
    n = float(np.linalg.norm(dE))
    if n < 1e-6:
        return "move-none"
    return "move-up" if dE[2] > 0.3 * n else ("move-down" if dE[2] < -0.3 * n else "move-flat")


# ------------------------------------------------------------------ the event store
class Events:
    """Every probe's commands, read as events for both role orders (rows)."""

    def __init__(self):
        self.recs: list[dict] = []
        self.rows: dict = {}

    def add(self, recs: list[dict]) -> None:
        base = len(self.recs)
        for k, r in enumerate(recs):
            r["uid"] = base + k
        self.recs += recs
        self._rows(recs)

    def _rows(self, recs) -> None:
        """Per command and role order: views, flags, motion, key (appended to self.rows)."""
        new = defaultdict(list)
        for r in recs:
            env = pair_world(*r["pair"])
            cmds = r["commands"]
            if not cmds:
                continue
            sets = np.array([r["s_init"]] + [c["s_set"] for c in cmds])
            st = np.array([c["s_start"] for c in cmds])
            en = np.array([c["s_end"] for c in cmds])
            rest = r["s_rest"]
            for o, (x, y) in enumerate(ORDERS):
                vs = env.pair_view(sets, x, y)
                v0, v1 = env.pair_view(st, x, y), env.pair_view(en, x, y)
                vr = env.pair_view(rest, x, y)
                f = flags(vs)
                mo = motion(v0, v1, vs[:-1], vs[1:])
                jd = joint_dir(vs[:-1], vs[1:])
                pv = cc_features(vs, VIEW_S, "AB")["p"]
                pair = tuple(r["pair"]) if o == 0 else tuple(r["pair"][::-1])
                for j in range(len(cmds)):
                    new["uid"].append(r["uid"])
                    new["cmd"].append(j)
                    new["order"].append(o)
                    new["pair"].append(pair)
                    new["kind"].append(cmds[j]["kind"])
                    new["reset"].append(cmds[j]["reset"])
                    new["f0"].append(f[j])
                    new["f1"].append(f[j + 1])
                    new["mo"].append(int(mo[j]))
                    new["jd"].append(int(jd[j]))
                    new["key"].append(key_of(f[j], f[j + 1], int(mo[j]), int(jd[j])))
                    new["v_pre"].append(vs[j])
                    new["v_start"].append(v0[j])
                    new["v_end"].append(v1[j])
                    new["v_set"].append(vs[j + 1])
                    new["v_rest"].append(vr)
                    dE = v1[j][VIEW_S.ee_pos] - v0[j][VIEW_S.ee_pos]
                    new["dEz"].append(float(dE[2]))
                    new["cmdc"].append(command_class(cmds[j]["kind"], dE))
                    new["factors"].append(tuple(factor_keys(f[j], f[j + 1], int(mo[j]), int(jd[j]))))
                    new["cell"].append(cell_of(f[j + 1], pv[j + 1]))
        for k, v in new.items():
            if k in ("key", "pair", "kind", "cmdc", "factors", "cell"):
                arr = np.empty(len(v), dtype=object)
                arr[:] = v
            else:
                arr = np.array(v)
            self.rows[k] = arr if k not in self.rows else np.concatenate([self.rows[k], arr])

    def __len__(self):
        return len(self.rows.get("key", []))

    # --------------------------------------------------------- the novelty archive
    def restart_bases(self, rng, n: int) -> list:
        """n archived end states to restart probes from, as Go-Explore returns to cells: a cell (the settled
        contact flags and a coarse bin of X's position relative to Y, ``cell_of``) is chosen with weight
        1 / sqrt(1 + its count), then a state in it."""
        if not len(self):
            return []
        cell = self.rows["cell"]
        idx = np.flatnonzero(self.rows["order"] == 0)[-EVENTS_POOL():]
        if not len(idx):
            return []
        cells = defaultdict(list)
        for i in idx:
            cells[cell[i]].append(i)
        names = sorted(cells)
        counts = Counter(cell[self.rows["order"] == 0])
        w = np.array([1.0 / np.sqrt(1.0 + counts[k]) for k in names])
        pick = [cells[names[c]][int(rng.integers(len(cells[names[c]])))]
                for c in rng.choice(len(names), size=n, p=w / w.sum())]
        out = []
        for i in pick:
            r = self.recs[self.rows["uid"][i]]
            c = r["commands"][self.rows["cmd"][i]]
            out.append({"pair": tuple(r["pair"]), "state": c["state"], "ctrl": c["ctrl"]})
        return out

    def trim(self, max_probes: int) -> None:
        """Keep the most recent ``max_probes`` probes (and their rows): memory over long runs."""
        if len(self.recs) <= max_probes:
            return
        cut = len(self.recs) - max_probes
        keep = self.rows["uid"] >= cut
        self.rows = {k: v[keep] for k, v in self.rows.items()}
        self.rows["uid"] = self.rows["uid"] - cut
        self.recs = self.recs[cut:]
        for k, r in enumerate(self.recs):
            r["uid"] = k

    def novelty(self, start: int) -> float:
        """Share of the events since row ``start`` whose key had been seen fewer than min_members times before
        them (what is still new)."""
        keys = self.rows["key"]
        seen = Counter(k for k in keys[:start] if k is not None)
        new = [k for k in keys[start:] if k is not None]
        if not new:
            return 0.0
        rare = 0
        for k in new:
            rare += seen[k] < EVENTS["min_members"]
            seen[k] += 1
        return rare / len(new)


def EVENTS_POOL() -> int:
    from ..config import BABBLE
    return BABBLE["restart_pool"]


# ------------------------------------------------------------------ keys: candidates, contingency, chunks
def actor_involved(rows) -> np.ndarray:
    """The hand touched X or Y before, at the end of or after the command (the effect is the hand's)."""
    S = VIEW_S
    touch = lambda v: (v[:, S.hand_A] > 0) | (v[:, S.hand_B] > 0)          # noqa: E731
    return touch(rows["v_pre"]) | touch(rows["v_end"]) | touch(rows["v_set"])


def key_table(ev: Events) -> list[dict]:
    """Every key: its count, its best precondition context and rate there, and whether it is kept."""
    R = ev.rows
    if not len(ev):
        return []
    ctx = [(*map(int, f), c) for f, c in zip(R["f0"], R["cmdc"])]
    actor = actor_involved(R)
    n_ctx = Counter(ctx)
    n_key, n_kc = Counter(), Counter()
    for fk, c, a in zip(R["factors"], ctx, actor):
        if a:
            for k in fk:
                n_key[k] += 1
                n_kc[(k, c)] += 1
    total = len(ctx)
    out = []
    for k, n in n_key.most_common():
        base = n / total
        # contingency: some context (flags before + command) raises the effect's rate well above its base rate,
        # on enough events there to tell
        rates = [(n_kc[(k, c)] / n_ctx[c], c) for c in n_ctx
                 if n_ctx[c] >= EVENTS["min_context"] and n_kc[(k, c)] >= EVENTS["min_in_context"]]
        best = max(rates) if rates else (0.0, None)
        lift = best[0] / base if base > 0 else 0.0
        out.append({"key": k, "n": n, "base_rate": round(base, 4), "best_rate": round(best[0], 3),
                    "lift": round(lift, 2),
                    "best_context": ({**dict(zip(FLAGS, best[1][:-1])), "command": best[1][-1]} if best[1] else None),
                    "kept": n >= EVENTS["min_members"] and lift >= EVENTS["lift"] and best[0] >= EVENTS["precision"]})
    return out


def chunks(ev: Events, kept: set) -> list[dict]:
    """Pairs of kept keys that follow each other within a probe (same role order) reliably."""
    R = ev.rows
    nxt = Counter()
    first = Counter()
    by = defaultdict(dict)
    for i, (u, j, o, fk) in enumerate(zip(R["uid"], R["cmd"], R["order"], R["factors"])):
        by[(u, o)][j] = [k for k in fk if k in kept]
    for seq in by.values():
        for j, ks in seq.items():
            for k in ks:
                first[k] += 1
                for k2 in seq.get(j + 1, []):
                    nxt[(k, k2)] += 1
    return [{"first": a, "then": b, "n": n, "p": round(n / first[a], 3)} for (a, b), n in nxt.most_common()
            if n >= EVENTS["min_members"] and n / first[a] >= EVENTS["chunk_p"]]


# ------------------------------------------------------------------ relations from the kept keys
def _signature(f: dict, keys) -> dict:
    sig = {}
    for k in keys:
        v = np.asarray(f[k], bool)
        share = float(v.mean()) if len(v) else 0.0
        if share >= EVENTS["constant"] or share <= 1 - EVENTS["constant"]:
            sig[k] = share >= EVENTS["constant"]
    return sig


def _hand_free_at_rest(v) -> np.ndarray:
    S = VIEW_S
    return (v[:, S.lf_A] <= 0) & (v[:, S.rf_A] <= 0) & (v[:, S.hand_A] <= 0) & (v[:, S.hand_B] <= 0)


def _at_rest(v) -> np.ndarray:
    S = VIEW_S
    L = characteristic_length(v[:, S.A_ext])
    LB = characteristic_length(v[:, S.B_ext])
    return (np.linalg.norm(v[:, S.A_vel], axis=1) < PROBE["rest_speed"] * L) & \
           (np.linalg.norm(v[:, S.B_vel], axis=1) < PROBE["rest_speed"] * LB)


def relations_from_events(ev: Events, old: RelationSet | None = None) -> tuple[RelationSet, dict]:
    """The relations of the kept keys (see the module docstring), and a report.  ``old``: the relations of
    the previous sleep; a new relation that fires on mostly the same states as an old one keeps its name."""
    R = ev.rows
    table = key_table(ev)
    kept = {t["key"] for t in table if t["kept"]}
    keys = R["key"]
    has = lambda k: np.array([k in fk for fk in R["factors"]], bool) if len(keys) else np.zeros(0, bool)   # noqa: E731
    rels, report = [], {"keys": table, "chunks": chunks(ev, kept), "relations": []}
    # ---------- object-object: X now touches Y, read at rest after letting go (both role orders)
    cc_groups = defaultdict(set)                         # signature -> {(uid, order)}
    rest = R["v_rest"]
    fr = cc_features(rest, VIEW_S, "AB")
    stable = fr["c_xy"] & _hand_free_at_rest(rest) & _at_rest(rest)
    if "xy=1" in kept:              # the hand can make X touch Y: the contact's states, wherever it holds at rest
        for i in np.flatnonzero(stable):
            sig = (bool(fr["c_xt"][i]), bool(fr["c_yt"][i]))
            cc_groups[sig].add((int(R["uid"][i]), int(R["order"][i])))
    first_row = {}
    for i, (u, o) in enumerate(zip(R["uid"], R["order"])):
        first_row.setdefault((int(u), int(o)), i)
    clusters = []
    for sig, members in cc_groups.items():
        if len(members) >= EVENTS["min_members"]:
            clusters.append({"sig": sig, "members": members})
    # mirrors: the same probes with X and Y exchanged; keep the order in which X moved more
    def mobility(c):
        d = [float(np.linalg.norm(R["v_rest"][first_row[m]][VIEW_S.A_pos] - R["v_pre"][first_row[m]][VIEW_S.A_pos]))
             for m in c["members"]]
        return float(np.median(d))
    keep = []
    for ci, c in enumerate(clusters):
        mirror = next((cj for cj, c2 in enumerate(clusters) if cj != ci and
                       len({(u, 1 - o) for u, o in c["members"]} & c2["members"]) > EVENTS["mirror"] * len(c["members"])),
                      None)
        if mirror is not None and (mobility(clusters[mirror]), mirror) > (mobility(c), ci):
            continue
        keep.append(c)
    for c in keep:
        idx = np.array([first_row[m] for m in sorted(c["members"])])
        V = rest[idx]
        f = cc_features(V, VIEW_S, "AB")
        X = f["p"]
        tilt = float(np.maximum(f["tilt_x"], f["tilt_y"]).max())
        sig = {"c_xy": True, **_signature({"c_xt": f["c_xt"], "c_yt": f["c_yt"]}, ("c_xt", "c_yt"))}
        rels.append(Relation("r?", CC, sig, X, membership_radius(X), tilt_max=tilt + PROBE["tilt_margin"],
                             stats={"members": len(X), "probes": len({m[0] for m in c["members"]}),
                                    "from": "X touches Y, kept at rest after letting go",
                                    "p_median": np.round(np.median(X, 0), 3).tolist()}))
    # ---------- hand-object: X left its support and follows the hand, held by both fingers
    held = has("xt=0|follows") if "xt=0|follows" in kept else np.zeros(len(keys), bool)
    if len(keys):
        held &= np.asarray(R["f1"])[:, 0] == 2
    sel = np.flatnonzero(held)
    if len(sel) >= EVENTS["min_members"]:
        V = R["v_set"][sel]
        fh = hc_features(V, VIEW_S, rotation=False, body=True)
        # how often gripping X and moving up carries it (the affordance's precision)
        f0 = np.array(list(R["f0"]))
        L = characteristic_length(R["v_start"][:, VIEW_S.A_ext])
        lifting = (f0[:, 0] == 2) & (f0[:, 3] == 1) & (R["dEz"] > EVENTS["moved"] * L) & (R["kind"] == "move")
        prec = float(held[lifting].mean()) if lifting.any() else 0.0
        g = Relation("g?", HC, {"c_l": True, "c_r": True}, member_vector(HC, fh), membership_radius(fh["body"]),
                     body_members=fh["body"], stats={"contact_only": True})
        g.stats.update(members=int(len(sel)), verified_carry=True, lift_precision=round(prec, 3),
                       precision_on_events=round(prec, 3), dynamic=False,
                       lift_subjects=dict(Counter(R["pair"][i][0] for i in sel)),
                       **{"from": "X left its support and follows the hand, both fingers on X"})
        rels.append(g)
    # ---------- object-state: a joint moved, per direction
    for direction in (-1, 1):
        sel = np.flatnonzero(has(f"joint{'+' if direction > 0 else '-'}")) \
            if f"joint{'+' if direction > 0 else '-'}" in kept else np.zeros(0, int)
        if len(sel) >= EVENTS["min_members"]:
            from ..world.relations import joint_norm
            q = joint_norm(R["v_set"][sel], VIEW_S, "A")[:, None]
            rels.append(Relation("s?", SC, {"art": True}, q, membership_radius(q),
                                 stats={"members": int(len(q)), "joint_direction": direction}))
    # ---------- Occam: drop contact flags of object-object signatures not needed to tell them apart
    rest_ok = stable | (~fr["c_xy"] & _hand_free_at_rest(rest) & _at_rest(rest))
    fc = [cc_features(rest[rest_ok], VIEW_S, "AB")]
    for rel in (r for r in rels if r.kind == CC):
        base = [rel.holds_f(f) for f in fc]
        dropped = []
        for flag in [k for k in rel.signature if k != "c_xy"]:
            val = rel.signature.pop(flag)
            if all(np.array_equal(rel.holds_f(f), b) for f, b in zip(fc, base)):
                dropped.append(flag)
            else:
                rel.signature[flag] = val
        rel.stats["dropped_flags"] = dropped
    _name(rels, old, ev)
    # ---------- resting tilt: bodies at rest touching nothing but the table
    alone = ~fr["c_xy"] & _hand_free_at_rest(rest) & _at_rest(rest)
    tilts = cc_features(rest[alone], VIEW_S, "AB")["tilt_x"]
    low = lower_mode(tilts) if len(tilts) else tilts
    low = low if len(low) else tilts
    rest_tilt = float(np.quantile(low, 0.995) + PROBE["tilt_margin"]) if len(low) else 0.05
    out = RelationSet(rels, rest_tilt)
    report["relations"] = [r.to_json() for r in rels]
    report["rest_tilt"] = rest_tilt
    report["events"] = int(len(keys))
    report["kept_keys"] = sorted(kept)
    return out, report


def _name(rels, old, ev) -> None:
    """r0, r1, ... / g0, ... / s0, ...; a relation that fires on mostly the same rest states (object-object) or
    settled states (hand-object) as a relation of ``old`` keeps that name."""
    prefix = {CC: "r", HC: "g", SC: "s"}
    used = set()
    if old is not None and len(ev):
        views = {CC: cc_features(ev.rows["v_rest"], VIEW_S, "AB"),
                 HC: hc_features(ev.rows["v_set"], VIEW_S, rotation=False, body=True),
                 SC: None}
        for r in rels:
            if r.kind == SC:
                continue
            fr = r.holds_f(views[r.kind])
            best, score = None, 0.0
            for o in old.relations:
                if o.kind != r.kind or o.name in used:
                    continue
                fo = o.holds_f(views[o.kind])
                small = min(fr.sum(), fo.sum())
                s = (fr & fo).sum() / small if small else 0.0
                if s > score:
                    best, score = o, s
            if best is not None and score > 0.5:
                r.name = best.name
                used.add(best.name)
    for r in rels:
        if r.name.endswith("?"):
            k = 0
            while f"{prefix[r.kind]}{k}" in used:
                k += 1
            r.name = f"{prefix[r.kind]}{k}"
            used.add(r.name)
    rels.sort(key=lambda r: ({CC: 0, HC: 1, SC: 2}[r.kind], r.name))


# ------------------------------------------------------------------ bep.babble.probe's file format
def save_probes(ev: Events, path) -> None:
    """probes.npz in bep.babble.probe's format, one record per probe and role order: s1 the settled start, s2
    the rest after letting go; events: the carrying events (kind 0 before, kind 2 settled after, followed =
    both fingers still on X) and the other moves touching X (kind 1, followed = X followed the hand)."""
    R = ev.rows
    recs = {}
    for i, (u, o) in enumerate(zip(R["uid"], R["order"])):
        key = (int(u), int(o))
        if key not in recs:
            r = ev.recs[int(u)]
            env = pair_world(*r["pair"])
            x, y = ORDERS[o]
            recs[key] = {"s1": env.pair_view(r["s_init"], x, y), "s2": R["v_rest"][i],
                         "pair": [TYPES.index(R["pair"][i][0]), TYPES.index(R["pair"][i][1])], "events": []}
        if R["kind"][i] != "move":
            continue
        f0, f1 = R["f0"][i], R["f1"][i]
        if R["mo"][i] == 3 and f0[3] == 1 and f1[3] == 0:        # X left its support following the hand
            recs[key]["events"].append((R["v_start"][i], True, 0))
            recs[key]["events"].append((R["v_set"][i], bool(f1[0] == 2), 2))
        elif f0[1] or f1[1]:
            recs[key]["events"].append((R["v_end"][i], bool(R["mo"][i] == 3), 1))
    order = sorted(recs)
    rr = [recs[k] for k in order]
    evs = [(n, e) for n, r in enumerate(rr) for e in r["events"]]
    from ..world.objects import VIEW_DIM
    np.savez_compressed(path, s1=np.array([r["s1"] for r in rr]), s2=np.array([r["s2"] for r in rr]),
                        attached=np.zeros(len(rr), bool), moved=np.ones(len(rr), bool),
                        pair=np.array([r["pair"] for r in rr]), types=np.array(TYPES), focus=np.zeros(len(rr), bool),
                        ev_s=np.array([e[0] for _, e in evs]).reshape(-1, VIEW_DIM),
                        ev_follow=np.array([e[1] for _, e in evs], bool),
                        ev_kind=np.array([e[2] for _, e in evs], int), ev_probe=np.array([n for n, _ in evs], int))
