"""Hand of God for two bodies (roles X = A, Y = B) and the hand E.

Each role is a free body of some shape or the moving part of a fixture.  There is no
intended contact mode: the configuration is placed, settled under gravity with the given
gripper command, and whatever physics produces is measured.

Placement rule (the same for every pairing): a fixture's base stands at the context pose and
its joint is set to the configuration's normalised value; a free Y stands on the table at
the context pose; a free X is placed relative to Y's frame (``pXY``); a free Y paired with a
fixture X is placed so that X relative to Y is ``pXY``.  The hand is placed relative to X.

``precheck`` runs the purely geometric tests without simulating, so proposals can be
resampled cheaply; ``realize`` simulates the settle and rejects IK failures, contact
penetration and scenes still moving.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .env import width_to_ctrl, yaw_rot
from ..config import GRIP_MAX, PROBE, REACH
from .objects import VIEW_S, is_fixture, kind_shape, overlap, pair_world

GRIP_OPEN, GRIP_CLOSE, GRIP_HOLD = 0, 1, 2
PARK = (0.95, -0.55, 0.0)             # where an uninvolved free body waits (out of reach)


@dataclass
class Region:
    """Workspace region from which the anchor body's world pose is drawn (x, y, yaw)."""
    x: tuple = (0.45, 0.60)
    y: tuple = (-0.10, 0.10)
    yaw: tuple = (-0.3, 0.3)

    def sample(self, rng) -> np.ndarray:
        return np.array([rng.uniform(*self.x), rng.uniform(*self.y), rng.uniform(*self.yaw)])


TRAIN_REGION = Region()
_UNSETTLED = 4.0 * PROBE["rest_speed"]
HAND_MIN_Z = 0.012                    # pinch point above the table (gripper geometry)


@dataclass
class GConfig:
    pXY: np.ndarray        # X relative to Y (Y's gravity-aligned frame), metres
    yawXY: float
    pEX: np.ndarray        # hand (pinch point) relative to X, metres, in X's yaw frame
    rEX: np.ndarray        # the hand's full rotation in X's yaw frame (any orientation: 6-DoF Hand of God)
    width: float           # initial finger opening (m)
    grip: int              # gripper command: open / close / hold the opening
    pair: tuple = ("cube", "cube")     # kinds of X and Y (shape or fixture)
    qx: float = 0.0        # normalised joint value of X / Y when they are moving parts
    qy: float = 0.0
    park: bool = False     # place a free Y out of reach (state primitives: Y plays no part)
    virtual_y: bool = False  # Y is a virtual reference on the table at the context pose (X placed relative to it)

    def copy(self) -> "GConfig":
        return replace(self, pXY=np.array(self.pXY, float), pEX=np.array(self.pEX, float),
                       rEX=np.array(self.rEX, float))

    def grip_ctrl(self) -> float:
        if self.grip == GRIP_CLOSE:
            return 0.0
        if self.grip == GRIP_OPEN:
            return 255.0
        return float(width_to_ctrl(self.width))


def _frame(env, n):
    """World position and yaw of a body's frame (after kinematics)."""
    bid = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"cube{n}"))
    R = env.data.xmat[bid].reshape(3, 3)
    return env.data.xpos[bid].copy(), float(np.arctan2(R[1, 0], R[0, 0]))


def _place(env, cfg: GConfig, ctx):
    """Place both roles (and fixtures' joints) in env.data; return (pA, yawA) of X's frame."""
    tx, ty = cfg.pair
    sx, sy = kind_shape(tx), kind_shape(ty)
    x0, y0, yaw0 = ctx[:3]
    d, m = env.data, env.model
    env.ref = "B"                                    # Y's role in pair views (a Spot when there is no body B)
    if ty == "spot":
        from .objects import SPOT, Spot
        env.ref = Spot(float(x0), float(y0), float(yaw0))
        if is_fixture(tx):
            env.place_fixture("A", x0, y0, yaw0)
            env.set_joint("A", cfg.qx)
            mujoco.mj_kinematics(m, d)
            return _frame(env, "A")
        pB, yawB = np.array([x0, y0, -SPOT.half[2]]), yaw0
        pA = pB + yaw_rot(yawB) @ np.array(cfg.pXY, float)
        pA[2] = max(pA[2], sx.half[2])
        env.place("A", pA[0], pA[1], yawB + cfg.yawXY, pA[2])
        return pA, yawB + cfg.yawXY
    if is_fixture(ty):
        env.place_fixture("B", x0, y0, yaw0)
        env.set_joint("B", cfg.qy)
    if is_fixture(tx):
        env.place_fixture("A", x0, y0, yaw0)
        env.set_joint("A", cfg.qx)
    mujoco.mj_kinematics(m, d)
    if is_fixture(tx):
        pA, yawA = _frame(env, "A")
        if cfg.park and not is_fixture(ty):
            env.place("B", *PARK)
        elif not is_fixture(ty):                     # free Y such that X relative to Y is pXY
            yawB = yawA - cfg.yawXY
            pB = pA - yaw_rot(yawB) @ np.asarray(cfg.pXY, float)
            pB[2] = max(pB[2], sy.half[2])
            env.place("B", pB[0], pB[1], yawB, pB[2])
        return pA, yawA
    if cfg.virtual_y:
        from .objects import SPOT
        pB, yawB = np.array([x0, y0, -SPOT.half[2]]), yaw0
        if not is_fixture(ty):
            env.place("B", *PARK)
    elif is_fixture(ty):
        pB, yawB = _frame(env, "B")
    else:
        pB, yawB = np.array([x0, y0, sy.half[2]]), yaw0
        env.place("B", x0, y0, yaw0)
    pXY = np.array(cfg.pXY, float)
    pA = pB + yaw_rot(yawB) @ pXY
    pA[2] = max(pA[2], sx.half[2])                   # X cannot be below the table
    yawA = yawB + cfg.yawXY
    env.place("A", pA[0], pA[1], yawA, pA[2])
    return pA, yawA


HALF_TURN = np.diag([-1.0, -1.0, 1.0])     # about the approach axis: the same grasp, fingers swapped


def teleport_ik(env, pE, RE, q_hint=None):
    """Joint angles putting the hand at (pE, RE) (the arm is teleported there, no motion is planned): tried from
    the hint and from home, with the hand as given and turned by half a turn about its approach axis (the
    symmetric gripper makes the same contacts)."""
    from .env import HOME_Q
    for R in (RE, RE @ HALF_TURN):
        for q0 in ((q_hint, HOME_Q) if q_hint is not None else (HOME_Q,)):
            q, ok = env.solve_ik(pE, R, q0)
            if ok:
                return q, True
    return None, False


def precheck(cfg: GConfig, ctx) -> str | None:
    """Geometric tests that need no simulation (free-body interpenetration, reach)."""
    tx, ty = cfg.pair
    if not is_fixture(tx) and not is_fixture(ty) and not cfg.virtual_y:
        sx, sy = kind_shape(tx), kind_shape(ty)
        pXY = np.array(cfg.pXY, float)
        pXY[2] = max(pXY[2], sx.half[2] - sy.half[2])
        if overlap(pXY, cfg.yawXY, sx, sy):
            return "overlap"
        pA = np.array(ctx[:2]) + yaw_rot(ctx[2])[:2, :2] @ pXY[:2]
        if not (REACH[0] < np.hypot(pA[0], pA[1]) < REACH[1]):
            return "out_of_reach"
    return None


def realize(cfg: GConfig, ctx, *, q_hint=None, settle_s: float = 0.15, check_speed: bool = True):
    """Return (env, state, ctrl, pair-view sensors) or a failure string."""
    why = precheck(cfg, ctx)
    if why:
        return why
    env = pair_world(*cfg.pair)
    d, m = env.data, env.model
    mujoco.mj_resetData(m, d)
    pA, yawA = _place(env, cfg, ctx)
    if not (REACH[0] < np.hypot(pA[0], pA[1]) < REACH[1] + 0.1):
        return "out_of_reach"
    pE = pA + yaw_rot(yawA) @ np.asarray(cfg.pEX, float)
    RE = yaw_rot(yawA) @ np.asarray(cfg.rEX, float)
    if pE[2] < HAND_MIN_Z:
        return "ee_below_table"
    if env.floating:                         # the floating hand goes anywhere the arm could take it
        from .reach import reach_map
        if not bool(reach_map().reachable(pE, RE)):
            return "unreachable"
        env.set_hand(pE, RE)
    else:                                    # the arm teleported there
        q, ok = teleport_ik(env, pE, RE, q_hint)
        if not ok:
            return "ik"
        d.qpos[env.arm_qadr] = q
        d.ctrl[:7] = q
    d.qpos[env.finger_qadr] = float(np.clip(cfg.width, 0.0, GRIP_MAX)) / 2
    d.ctrl[env.gi] = cfg.grip_ctrl()
    mujoco.mj_forward(m, d)
    if env.bad_penetration(d):
        return "penetration"
    mujoco.mj_step(m, d, nstep=max(1, int(settle_s / env.dt)))
    view = env.pair_view(env.sensors(d), "A", getattr(env, "ref", "B"))
    if check_speed and max(np.linalg.norm(view[VIEW_S.A_vel]), np.linalg.norm(view[VIEW_S.B_vel])) > _UNSETTLED:
        return "unsettled"
    d.time = 0.0
    return env, env.get_state(d), d.ctrl.copy(), view
