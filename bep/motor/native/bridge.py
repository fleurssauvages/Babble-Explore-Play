"""Fused native rollouts: simulate and evaluate the primitives' costs in C++ (fused.cpp, next to this file).

``attach(cost)`` gives a cost object (bep.motor.costs.PrimitiveCost, or LocalTargetCost used in
exploration) a ``native`` evaluator; bep.motor.mppi.SpikeMPPI then calls it instead of mujoco.rollout +
the Python cost (SpikeMPPI attaches it on first use).  The C++ code mirrors the
Python implementation term by term (identical costs to ~1e-15).  Set BEP_NATIVE=0 to disable.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ...config import EXEC
from ...world.geometry import characteristic_length
from ...world.objects import VIEW_DIM, VIEW_SLICES
from ...world.relations import CC, HC, SC, ZDIM

VIEW_FIELDS = ["ee_pos", "ee_quat", "A_pos", "A_quat", "B_pos", "B_quat", "finger1", "finger2", "lf_A", "rf_A", "A_B",
               "A_table", "B_table", "hand_B", "hand_table", "A_vel", "B_vel", "A_ext", "B_ext", "A_sym", "B_sym",
               "hand_A", "A_q", "B_q", "A_qrange", "B_qrange", "A_art", "B_art", "A_gw", "B_gw"]
_SIG = {CC: ["c_xy", "c_xt", "c_yt"], HC: ["c_l", "c_r", "c_h", "c_2", "c_any"], SC: ["art"]}
_KIND = {CC: 0, HC: 1, SC: 2}
_MOD = None
_ENGINES: dict = {}


def _module():
    global _MOD
    if _MOD is None:
        import fcntl
        here = Path(__file__).resolve().parent
        with open(here / ".build.lock", "w") as lock:       # many workers may start at once: one builds
            fcntl.flock(lock, fcntl.LOCK_EX)
            so = list(here.glob("_fused*.so"))
            if not so or min(f.stat().st_mtime for f in so) < (here / "fused.cpp").stat().st_mtime:   # missing or stale
                subprocess.check_call([sys.executable, str(here / "setup.py"), "build_ext", "--inplace"],
                                      stdout=subprocess.DEVNULL)
        sys.path.insert(0, str(here))
        import _fused  # noqa: E402
        _MOD = _fused
    return _MOD


def enabled() -> bool:
    return os.environ.get("BEP_NATIVE", "1") != "0"


def _pair(env, x, target) -> dict:
    const, dst, src = env.pview_spec(x, target)
    return {"const": const, "src": np.asarray(src, np.int32), "dst": np.asarray(dst, np.int32)}


def _relations(R) -> tuple[list, np.ndarray, np.ndarray]:
    rels = _relations_list(R.relations)
    names = [r.name for r in R.relations]
    bit_rel, bit_role = [], []
    for b in R.bit_names():
        role = 1 if b.endswith("@Y") else 0
        bit_rel.append(names.index(b[:-2] if role else b))
        bit_role.append(role)
    return rels, np.array(bit_rel, np.int32), np.array(bit_role, np.int32)


def _relations_list(relations) -> list:
    rels = []
    for r in relations:
        sig = [(_SIG[r.kind].index(k), int(bool(v))) for k, v in r.signature.items()]
        members = r.members if r.body_members is None else r.body_members
        rels.append({"kind": _KIND[r.kind], "dim": int(r.members.shape[1]), "delta": float(r.delta),
                     "tilt_max": float(r.tilt_max), "members": np.ascontiguousarray(members, float),
                     "body_hand": r.body_members is not None, "held": bool(r.held),
                     "contact_only": r.contact_only,
                     "sig_flag": np.array([s[0] for s in sig], np.int32), "sig_val": np.array([s[1] for s in sig], np.int32),
                     "center": np.atleast_1d(np.asarray(r.center, float))})
    return rels


def _tree(tree, W, kappa, Lmin) -> dict:
    data = np.asarray(tree.data, float)
    return {"W": np.asarray(W, float), "Zw": np.ascontiguousarray(data[:, :ZDIM]),
            "Laug": np.ascontiguousarray(data[:, ZDIM]),
            "kappa": float(kappa), "Lmin": float(Lmin)}


def _base(env, prim, x, target) -> dict:
    rels, bit_rel, bit_role = _relations(prim.R)
    pair = _pair(env, x, prim.reference(env, x, target))
    return {"off": np.array([VIEW_SLICES[f][0] for f in VIEW_FIELDS], np.int32), "view_dim": int(VIEW_DIM),
            "main": pair,
            "hc_zero": bool(prim.kind == HC and not prim.x_is_part), "rels": rels, "bit_rel": bit_rel,
            "bit_role": bit_role, "goal_mode": int(prim.goal_mode), "goal_slice": np.asarray(prim.goal_slice, np.int32),
            "carry_mask": int(prim.carry_mask), "carry_mode": 0, "release_goal": prim.kind == CC,
            "lift_goal": prim.goal_requires_lift, "lift_contacts": np.zeros(0, np.int32),
            "transport_scale": np.zeros(len(prim.goal_slice), float),
            "goal_center": np.atleast_1d(np.asarray(prim.goal_center, float)), "rest_tilt": float(prim.R.rest_tilt),
            "speed_scale": float(characteristic_length(pair["const"][VIEW_SLICES["B_ext"]])),
            "keep_rel": np.zeros(0, np.int32), "keep_pair": [], "leave_quat": np.zeros(0, np.int32),
            "leave_vel": np.zeros(0, np.int32), "w_break": 0.0, "w_step": 0.0, "leave_w": 0.0,
            "leave_length": np.zeros(0),
            "trees": {}, "pooled": None, "W_goal": np.zeros(len(np.atleast_1d(prim.goal_slice))), "tau": 0.05, "unexplored": 0.0,
            "cap": 1e9, "k": 1, "z_target": np.zeros(ZDIM), "W_dist": np.ones(ZDIM), "m_target": -1,
            "mode_penalty": 0.0, "terminal": 0.0, "phi_w": 0.0, "safety_w": 0.0, "cost_type": 0, "wp_resid": 0, "progress_final": 0,
            "barrier_d": np.zeros(0, np.int32), "barrier_w": 0.0, "goal_reward": 0.0, "typ_sigma": 1.0,
            "typ_W": np.zeros(len(np.atleast_1d(prim.goal_slice))), "typ_center": np.zeros(len(np.atleast_1d(prim.goal_slice)))}


def _field(spec, field) -> None:
    spec["trees"] = {int(m): _tree(t, field.W[m], field.kappas[m], Lmin) for m, (t, Lmin) in field.trees.items()}
    if field.pooled is not None:
        t, Lmin, kp = field.pooled
        spec["pooled"] = _tree(t, field.W_pooled, kp, Lmin)
    spec["W_goal"] = np.asarray(field.W_goal, float)[np.asarray(spec["goal_slice"])]
    from ...config import FIELD
    spec.update(tau=float(field.tau), cap=float(field.cap), k=int(field.k), unexplored=float(FIELD["unexplored_s"]))


class Engine:
    """One fused evaluator (per thread count), the spec set lazily for whichever cost is evaluated."""

    def __init__(self, threads: int):
        self.f = _module().Fused(int(threads))
        self.current, self.ver = None, 0

    def evaluate(self, cost, model, x0, controls, substeps, mocap: bool = False):
        ver = getattr(cost, "native_ver", 0)
        if self.current is not cost or self.ver != ver:
            self.f.set_spec(cost.native_spec)
            self.current, self.ver = cost, ver
        return self.f.evaluate(int(model._address), np.ascontiguousarray(x0, float),
                               np.ascontiguousarray(controls, float), int(substeps), bool(mocap))


def engine(threads: int) -> Engine:
    if threads not in _ENGINES:
        _ENGINES[threads] = Engine(threads)
    return _ENGINES[threads]


def attach(cost):
    """Give a cost object its native specification (returns the cost)."""
    cost.native = False
    if not enabled() or not hasattr(getattr(cost, "prim", None), "metric_of"):     # a discovered primitive
        return cost
    name = type(cost).__name__
    if name == "PrimitiveCost":
        spec = _base(cost.env, cost.prim, cost.x, cost._target_arg)
        _field(spec, cost.field)
        env = cost.env
        spec.update(carry_mode=int(cost.carry_mode), lift_contacts=cost.lift_contacts.astype(np.int32),
                    phi_w=cost.dt * EXEC["phi_weight"],
                    safety_w=EXEC["side_effect_weight"] * EXEC["phi_weight"] * cost.dt * EXEC["side_effects"],
                    terminal=EXEC["terminal_weight"], w_break=float(cost.w_break), w_step=float(cost.w_step),
                    leave_w=float(cost.w_step))
        if cost.transport_scale is not None:
            spec["transport_scale"] = np.asarray(cost.transport_scale, float)
        names = [r.name for r in cost.prim.R.relations]
        Rall = getattr(cost.prim, "Rall", cost.prim.R)
        extra = [r for r in Rall.relations if r.name not in names and any(k.rel == r.name for k in cost.keep)]
        if extra:                                      # kept relations of other learnings: appended, no mode bit
            spec["rels"] = spec["rels"] + _relations_list(extra)
            names = names + [r.name for r in extra]
        keep_rel, keep_pair = [], []
        for k in cost.keep:
            r = Rall.get(k.rel)
            other = next(n for n in env.cubes if n != k.x) if r.kind == SC else k.y
            keep_rel.append(names.index(k.rel))
            keep_pair.append(_pair(env, k.x, other))
        spec["keep_rel"], spec["keep_pair"] = np.array(keep_rel, np.int32), keep_pair
        spec["leave_quat"] = np.array([env.sidx[f"{n}_quat"][0] for n in cost.others], np.int32)
        spec["leave_vel"] = np.array([env.sidx[f"{n}_vel"][0] for n in cost.others], np.int32)
        spec["leave_length"] = np.array([env.shape(n).length for n in cost.others])
        spec["goal_center"] = np.asarray(cost.center, float)
        spec["progress_final"] = 1                  # progress on the final snapshot only (world.PrimitiveCost)
        W, sigma = cost.prim.typical_scale()
        spec.update(barrier_d=np.asarray(cost.barrier_d, np.int32),
                    barrier_t=np.asarray(cost.barrier_t, np.int32),
                    barrier_sigma_table=float(EXEC["barrier_sigma_table"]),
                    blend_off=np.array([o for o, _ in cost.blend], np.int32),
                    blend_len=np.array([len(i) for _, i in cost.blend], np.int32),
                    blend_idx=np.concatenate([i for _, i in cost.blend] or [np.zeros(0, int)]).astype(np.int32),
                    blend_eps=float(EXEC["mode_blend"]),
                    barrier_sigma=float(EXEC["barrier_sigma"]))
        spec.update(barrier_w=float(EXEC["barrier_weight"]),
                    goal_reward=float(EXEC["goal_reward"]), typ_sigma=float(sigma), typ_W=np.asarray(W, float),
                    typ_center=np.asarray(cost.center, float))
    elif name == "LocalTargetCost":                   # reverse Go-Explore validation toward a node
        from ...config import CONTROL_DT
        spec = _base(cost.env, cost.prim, "A", "B")
        spec.update(cost_type=1, wp_resid=int(cost.m == cost.prim.goal_mode and not cost.prim.goal_requires_lift),
                    z_target=np.asarray(cost.z, float), W_dist=np.asarray(cost.prim.metric_of(cost.m), float),
                    m_target=int(cost.m),
                    mode_penalty=float(cost.mode_penalty), terminal=float(cost.terminal_weight),
                    safety_w=EXEC["side_effect_weight"] * EXEC["phi_weight"] * CONTROL_DT)
    else:
        return cost
    cost.native_spec = spec
    cost.native = True
    if getattr(cost, "wp", None) is not None:          # a waypoint set before the first evaluation
        cost.set_waypoint(*cost.wp)
    return cost
