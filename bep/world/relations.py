"""Discovered relations: classifiers over object-relative features + contacts.

Features are computed from the two-object pair view (``objects.VIEW_S``) of any
world, and are expressed in the objects' own units, so one relation can hold
between objects of different shapes:

* object-object, ordered pair (X, Y): X's centre in Y's gravity-aligned frame,
  divided per axis by the combined extents (x, y: Y's half extent + X's
  horizontal radius; z: the two half heights), so X resting on top of Y is
  z = 1 for every pair of shapes; contacts (X-Y, X-table, Y-table); tilts.
* hand-object (E, X): what fixes the contact points of the parallel gripper on X,
  and nothing else: the pinch point (between the finger pads) in X's frame
  divided by X's half extents, the closing axis (the line the fingers move along)
  in X's frame, without sign (the gripper is symmetric under a half turn) and
  folded by X's own symmetry (a cube grasped along x or along y is one grasp),
  and the finger opening divided by X's graspable width; contacts of each finger
  and of the hand body (so grasping and pushing are different signatures).  The
  rotation of the hand about its closing axis (approaching from above or from the
  side) leaves the contact points unchanged and is not part of the state: the
  controller chooses it.
* object state (X): the joint value of a moving part (drawer, door), normalised
  by its range; 0 for free bodies.
* yaw codes use each shape's own symmetry (none for a cylinder).

A relation is not a hand-written predicate: it is a cluster found by the probe,
a contact signature plus member feature vectors; ``holds`` = signature matches
AND the feature lies within ``delta`` of some member AND tilts are no larger
than observed.  A carrying grasp (a hand-object relation the probe verified by
lifting) is binary: it holds when its contact signature (both fingers) holds and
the line between the fingers (through the pinch point, along the closing axis)
crosses X; where on X the fingers touch does not matter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import quat_to_mat, rotate_into_yaw_frame

CC, HC, SC = "object-object", "hand-object", "object-state"


# --------------------------------------------------------------- features
def tilt_of(R):
    """1 - |cos| of the angle between vertical and the body axis closest to it: 0 for a body resting on
    any face (a box on its side is not tipped), 1 for one standing on an edge at 45 degrees x 2."""
    return 1.0 - np.clip(np.abs(R[..., 2, :]).max(axis=-1), 0.0, 1.0)


def _yaw_tilt(q):
    R = quat_to_mat(q)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0]), tilt_of(R)


def wrap_sym(theta, period):
    """Angle modulo a symmetry period (0 = full symmetry: the angle carries no information)."""
    period = np.asarray(period, float)
    safe = np.where(period > 0, period, 1.0)
    return np.where(period > 0, np.mod(theta + safe / 2, safe) - safe / 2, 0.0)


def sym_code(theta, period) -> np.ndarray:
    """2-D code of an angle with the given symmetry period, |d code| ~ |d theta| / 4 * (2 pi / period)."""
    period = np.asarray(period, float)
    k = np.where(period > 0, 2 * np.pi / np.where(period > 0, period, 1.0), 0.0)
    return np.stack([np.cos(k * theta) / 4.0, np.sin(k * theta) / 4.0], axis=-1)


def _pair(sens, S, order):
    a, b = ("A", "B") if order == "AB" else ("B", "A")
    g = lambda k: sens[..., getattr(S, k)]  # noqa: E731
    return (g(f"{a}_pos"), g(f"{a}_quat"), g(f"{a}_ext"), g(f"{a}_sym"),
            g(f"{b}_pos"), g(f"{b}_quat"), g(f"{b}_ext"), g(f"{b}_sym"))


def cc_features(sens: np.ndarray, S, order: str = "AB") -> dict:
    """Ordered pair features; ``order='AB'`` gives X=A, Y=B, 'BA' the swap."""
    pX, qX, eX, symX, pY, qY, eY, symY = _pair(sens, S, order)
    yawX, tiltX = _yaw_tilt(qX)
    yawY, tiltY = _yaw_tilt(qY)
    d = rotate_into_yaw_frame(pX - pY, yawY)
    rX = np.maximum(eX[..., 0], eX[..., 1])
    scale = np.stack([eY[..., 0] + rX, eY[..., 1] + rX, eX[..., 2] + eY[..., 2]], axis=-1)
    tX, tY = sens[..., S.A_table] > 0, sens[..., S.B_table] > 0
    if order == "BA":
        tX, tY = tY, tX
    return {"p": d / scale, "p_m": d, "yaw": wrap_sym(yawX - yawY, symX), "sym": symX,
            "c_xy": sens[..., S.A_B] > 0, "c_xt": tX, "c_yt": tY, "tilt_x": tiltX, "tilt_y": tiltY}


CLOSING = 1                  # the fingers slide along the hand frame's y axis (column 1 of its rotation)


def axis_code(a, symA) -> np.ndarray:
    """Code of the gripper's closing axis ``a`` (unit, in X's yaw frame): without sign (half-turn symmetric
    gripper), and folded by X's symmetry about the vertical: a square section (period pi/2) does not tell x
    from y, a round one (period 0) keeps only the horizontal magnitude."""
    a = np.abs(np.asarray(a, float))
    symA = np.asarray(symA, float)
    hx, hy, az = a[..., 0], a[..., 1], a[..., 2]
    square = np.abs(symA - np.pi / 2) < 1e-6
    round_ = symA <= 0
    h = np.hypot(hx, hy)
    c0 = np.where(round_, h, np.where(square, np.maximum(hx, hy), hx))
    c1 = np.where(round_, 0.0, np.where(square, np.minimum(hx, hy), hy))
    return np.stack([c0, c1, az], axis=-1)


def hc_features(sens: np.ndarray, S, *, rotation: bool = True, body: bool = False) -> dict:
    """Hand relative to object A (role X): pinch point, closing axis, opening (the contact points)."""
    pE, pA, eA, symA = sens[..., S.ee_pos], sens[..., S.A_pos], sens[..., S.A_ext], sens[..., S.A_sym]
    RA = quat_to_mat(sens[..., S.A_quat])
    yawA = np.arctan2(RA[..., 1, 0], RA[..., 0, 0])
    RE = quat_to_mat(sens[..., S.ee_quat])
    width = sens[..., S.finger1] + sens[..., S.finger2]
    d = rotate_into_yaw_frame(pE - pA, yawA)
    axis = rotate_into_yaw_frame(RE[..., :, CLOSING], yawA)
    c_l, c_r, c_h = sens[..., S.lf_A] > 0, sens[..., S.rf_A] > 0, sens[..., S.hand_A] > 0
    out = {"p": d / eA, "p_m": d, "axis": axis_code(axis, symA),
           "width": width / sens[..., S.A_gw], "width_m": width, "c_l": c_l, "c_r": c_r, "c_h": c_h,
           # contacts folded by the gripper's half-turn symmetry (it swaps the fingers): both fingers, any contact
           "c_2": c_l & c_r, "c_any": c_l | c_r | c_h}
    if rotation:                                     # full hand rotation used to reconstruct configurations
        c, s_ = np.cos(yawA), np.sin(yawA)
        Ry = np.zeros(np.shape(yawA) + (3, 3))
        Ry[..., 0, 0], Ry[..., 0, 1], Ry[..., 1, 0], Ry[..., 1, 1], Ry[..., 2, 2] = c, s_, -s_, c, 1.0
        out["R_m"] = Ry @ RE
    if body:
        # Acquisition geometry follows the object when it tilts. The gravity-aligned
        # coordinates above still describe movements and archived hand poses.
        p = np.einsum("...ij,...i->...j", RA, pE - pA) / eA
        a = np.einsum("...ij,...i->...j", RA, RE[..., :, CLOSING])
        px, py = np.abs(p[..., 0]), np.abs(p[..., 1])
        square = np.abs(symA - np.pi / 2) < 1e-6
        rectangle = np.abs(symA - np.pi) < 1e-6
        axial = symA <= 0
        p[..., 0] = np.where(axial, np.hypot(px, py),
                             np.where(square, np.maximum(px, py), np.where(rectangle, px, p[..., 0])))
        p[..., 1] = np.where(axial, 0.0, np.where(square, np.minimum(px, py), np.where(rectangle, py, p[..., 1])))
        out["body"] = np.concatenate([p, axis_code(a, symA), out["width"][..., None]], axis=-1)
    return out


def pinch_outside(body) -> np.ndarray:
    """How far the line between the fingers misses X, in X's half extents (0: it crosses X): the pinch point
    (midway between the finger pads) against X's box, from the ``body`` grasp features (the pinch point in X's
    full frame, folded by X's symmetry: radial distance for a round section)."""
    return np.maximum(np.max(np.abs(np.asarray(body)[..., :3]), axis=-1) - 1.0, 0.0)


def joint_norm(sens, S, role: str = "A"):
    """Joint value of role A (or B) normalised by its range; 0 for bodies without a joint."""
    q, rng, art = sens[..., getattr(S, f"{role}_q")], sens[..., getattr(S, f"{role}_qrange")], \
        sens[..., getattr(S, f"{role}_art")]
    return np.where(art > 0, (q - rng[..., 0]) / np.maximum(rng[..., 1] - rng[..., 0], 1e-9), 0.0)


def sc_features(sens: np.ndarray, S, role: str = "A") -> dict:
    return {"q": joint_norm(sens, S, role), "art": sens[..., getattr(S, f"{role}_art")] > 0}


# layout of the relational state z
Z_PXY, Z_YXY, Z_PEX, Z_AXIS, Z_OPEN, Z_QX, Z_QY = slice(0, 3), slice(3, 5), slice(5, 8), slice(8, 11), 11, 12, 13
ZDIM = 14


def to_metric_shapes(pn, pen, sx, sy):
    """Object-unit relative positions (X relative to Y, the hand relative to X) -> metres, for shapes sx, sy
    (inverse of the features)."""
    ex, ey = sx.ext, sy.ext
    rx = max(ex[0], ex[1])
    return (np.asarray(pn, float) * np.array([ey[0] + rx, ey[1] + rx, ex[2] + ey[2]]),
            np.asarray(pen, float) * ex)


def pair_features(sens: np.ndarray, S, *, body_hand: bool = False) -> tuple[dict, dict, dict, dict]:
    return (cc_features(sens, S), hc_features(sens, S, rotation=False, body=body_hand),
            sc_features(sens, S, "A"), sc_features(sens, S, "B"))


def relational_z(sens: np.ndarray, S, *, features=None) -> np.ndarray:
    """14-D relational state of roles (E, X=A, Y=B), in the bodies' own units:
    [p_X^Y (3), yaw_XY code (2), pinch point p_E^X (3), closing axis code (3), opening, joint X, joint Y]."""
    c, h, x, y = pair_features(sens, S) if features is None else features
    return np.concatenate([c["p"], sym_code(c["yaw"], c["sym"]), h["p"], h["axis"],
                           h["width"][..., None], x["q"][..., None], y["q"][..., None]], axis=-1)


# --------------------------------------------------------------- relations
SIG_KEYS = {CC: ("c_xy", "c_xt", "c_yt"), HC: ("c_l", "c_r", "c_h", "c_2", "c_any"), SC: ("art",)}


def member_vector(kind: str, f: dict) -> np.ndarray:
    if kind == CC:
        return f["p"]
    if kind == SC:
        return f["q"][..., None]
    return np.concatenate([f["p"], f["axis"], f["width"][..., None]], axis=-1)     # the contact points


@dataclass
class Relation:
    name: str                       # r0, r1, ... (no semantics)
    kind: str                       # CC or HC
    signature: dict                 # contact flag -> bool
    members: np.ndarray             # (M, d) member feature vectors
    delta: float                    # k-NN membership radius
    tilt_max: float = 1.0
    stats: dict = field(default_factory=dict)
    body_members: np.ndarray | None = None

    def __post_init__(self):
        self.members = np.asarray(self.members, float)
        if self.body_members is not None:
            self.body_members = np.asarray(self.body_members, float)
        self._tree = cKDTree(self.members if self.body_members is None else self.body_members)
        self.center = np.median(self.members, axis=0)

    kinds = None                    # (X kind, Y kind) pairs of the learning it comes from (None: any)
    covered = frozenset()           # pairs any merged learning saw (bep.explore.library.load_library)

    @property
    def carries(self) -> bool:
        from ..config import PRIMS
        return (self.kind == HC and self.stats.get("verified_carry", True)
                and self.stats.get("lift_precision", 0.0) >= PRIMS["min_lift_precision"])

    @property
    def contact_only(self) -> bool:
        return bool(self.stats.get("contact_only", False))

    @property
    def held(self) -> bool:
        """A carrying grasp: binary, both fingers on X and the line between them crossing X."""
        return self.kind == HC and self.body_members is not None

    def _membership_vector(self, f: dict) -> np.ndarray:
        return f["body"] if self.body_members is not None else member_vector(self.kind, f)

    def applies(self, kx: str, ky: str | None = None) -> bool:
        """Is the relation read on bodies of these kinds?  Yes on kinds its learning saw, and on kinds no
        learning saw (extrapolation); not on kinds another learning owns.  Object-object relations look at
        the pair, the others at X's kind."""
        if not self.kinds:
            return True
        if self.kind == CC and ky is not None:
            return (kx, ky) in self.kinds or (kx, ky) not in self.covered
        return kx in {p[0] for p in self.kinds} or kx not in {p[0] for p in self.covered}

    def holds_f(self, f: dict) -> np.ndarray:
        if np.ndim(f[SIG_KEYS[self.kind][0]]) == 0:
            return self._holds_one(f)
        ok = np.ones(np.shape(f[SIG_KEYS[self.kind][0]]), dtype=bool)
        for k, v in self.signature.items():
            ok &= (np.asarray(f[k]) > 0) == bool(v)
        if self.contact_only:
            return ok
        if self.kind == CC:
            ok &= (f["tilt_x"] <= self.tilt_max) & (f["tilt_y"] <= self.tilt_max)
        if self.held:
            return ok & (pinch_outside(f["body"]) <= 0.0)
        out = np.zeros(ok.size, dtype=bool)
        sel = np.nonzero(ok.reshape(-1))[0]
        if len(sel):
            x = self._membership_vector(f)
            flat = x.reshape(-1, x.shape[-1])
            d, _ = self._tree.query(flat[sel], k=1)
            out[sel] = d <= self.delta
        return out.reshape(ok.shape)

    def _holds_one(self, f: dict) -> np.ndarray:
        """holds_f for one state (every control step): the same tests on scalars."""
        for k, v in self.signature.items():
            if (np.asarray(f[k]) > 0) != bool(v):
                return np.array(False)
        if self.contact_only:
            return np.array(True)
        if self.kind == CC and not (f["tilt_x"] <= self.tilt_max and f["tilt_y"] <= self.tilt_max):
            return np.array(False)
        if self.held:
            return np.array(bool(pinch_outside(f["body"]) <= 0.0))
        d, _ = self._tree.query(np.asarray(self._membership_vector(f)).reshape(-1), k=1)
        return np.array(bool(d <= self.delta))

    def residual_f(self, f: dict) -> np.ndarray:
        """Distance (object units) of the relation's feature to its member set (0 inside)."""
        if self.contact_only:
            return (~self.holds_f(f)).astype(float)
        if self.held:
            return pinch_outside(f["body"])
        x = self._membership_vector(f)
        flat = x.reshape(-1, x.shape[-1])
        d, _ = self._tree.query(flat, k=1)
        return np.maximum(d - self.delta, 0.0).reshape(x.shape[:-1])

    def to_json(self) -> dict:
        return {"name": self.name, "kind": self.kind, "signature": {k: bool(v) for k, v in self.signature.items()},
                "delta": self.delta, "tilt_max": self.tilt_max, "n_members": int(len(self.members)),
                "center": np.round(self.center, 4).tolist(), "stats": self.stats}


class RelationSet:
    def __init__(self, relations: list[Relation], rest_tilt: float | None = None) -> None:
        self.relations = list(relations)
        self._rest_tilt = rest_tilt
        self.body_hand = any(r.body_members is not None for r in self.relations)

    def __iter__(self):
        return iter(self.relations)

    def __len__(self):
        return len(self.relations)

    def get(self, name: str) -> Relation:
        return next(r for r in self.relations if r.name == name)

    @property
    def cc(self) -> list[Relation]:
        return [r for r in self.relations if r.kind == CC]

    @property
    def hc(self) -> list[Relation]:
        return [r for r in self.relations if r.kind == HC]

    @property
    def sc(self) -> list[Relation]:
        return [r for r in self.relations if r.kind == SC]

    @property
    def rest_tilt(self) -> float:
        """Tilt of an undisturbed resting object (learned from the probe); beyond it an object
        counts as knocked over / not settled."""
        if self._rest_tilt is not None:
            return self._rest_tilt
        return min([r.tilt_max for r in self.cc], default=0.05)

    def bit_names(self) -> list[str]:
        """Bit order of mode masks: object-object r(X,Y), hand-object g(E,X), state s(X), state s(Y)
        (state bits of Y are written "s@Y")."""
        return [r.name for r in self.cc] + [r.name for r in self.hc] + [r.name for r in self.sc] + \
            [f"{r.name}@Y" for r in self.sc]

    def holds_all(self, sens: np.ndarray, S, *, features=None) -> dict[str, np.ndarray]:
        out = {}
        cc, hc, sc = self.cc, self.hc, self.sc
        if cc:
            fc = cc_features(sens, S) if features is None else features[0]
            for r in cc:
                out[r.name] = r.holds_f(fc)
        if hc:
            fh = hc_features(sens, S, rotation=False, body=self.body_hand) if features is None else features[1]
            for r in hc:
                out[r.name] = r.holds_f(fh)
        if sc:
            fx, fy = (sc_features(sens, S, "A"), sc_features(sens, S, "B")) if features is None else features[2:]
            for r in sc:
                out[r.name] = r.holds_f(fx)
                out[f"{r.name}@Y"] = r.holds_f(fy)
        return out

    def mode(self, sens: np.ndarray, S, *, features=None) -> np.ndarray:
        h = self.holds_all(sens, S, features=features)
        m = np.zeros(sens.shape[:-1], dtype=np.int64)
        for b, name in enumerate(self.bit_names()):
            m |= h[name].astype(np.int64) << b
        return m

    def bit(self, name: str) -> int:
        return 1 << self.bit_names().index(name)

    # ------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        path = Path(path)
        arrays = {f"members_{r.name}": r.members for r in self.relations}
        arrays.update({f"body_{r.name}": r.body_members for r in self.relations if r.body_members is not None})
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps({"rest_tilt": self._rest_tilt,
                                                         "relations": [r.to_json() for r in self.relations]},
                                                        indent=1))

    @staticmethod
    def load(path: str | Path) -> "RelationSet":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        rest = meta.get("rest_tilt") if isinstance(meta, dict) else None
        meta = meta["relations"] if isinstance(meta, dict) else meta
        with np.load(path.with_suffix(".npz")) as arr:
            rels = [Relation(m["name"], m["kind"], m["signature"], arr[f"members_{m['name']}"], m["delta"],
                             m["tilt_max"], m.get("stats", {}), arr.get(f"body_{m['name']}")) for m in meta]
        return RelationSet(rels, rest)
