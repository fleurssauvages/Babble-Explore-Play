"""MuJoCo two-cube stacking environment: Franka Panda + parallel gripper.

Conventions
-----------
* The table top is the plane z = 0; the robot base is at the world origin.
* Cube B is the reference (base) cube, cube A is stacked on B.
* The control vector is the MuJoCo actuator vector: 7 arm joint position
  targets + one gripper command (0..255 maps to a 0..0.08 m finger opening).
* A tool-centre-point site ``pinch`` sits between the finger pads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
SCENE_XML = ASSET_DIR / "stacking_scene.xml"

CUBE_HALF = 0.025
CUBE_SIZE = 2.0 * CUBE_HALF
GRIPPER_MAX_WIDTH = 0.08
HOME_Q = np.array([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7853981634], dtype=np.float64)
# Arm links whose collision meshes stay active. Proximal links never come near
# the cubes in this workspace; disabling them makes rollouts ~2x faster.
ACTIVE_COLLISION_BODIES = ("link6", "link7", "hand", "left_finger", "right_finger")


def width_to_ctrl(width):
    return np.clip(np.asarray(width) / GRIPPER_MAX_WIDTH * 255.0, 0.0, 255.0)


def yaw_rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def ee_down_rot(yaw: float) -> np.ndarray:
    """Gripper pointing down (pinch z = -world z) with its x-axis at ``yaw``."""
    return yaw_rot(yaw) @ np.diag([1.0, -1.0, -1.0])


@dataclass
class SensorLayout:
    ee_pos: slice
    ee_quat: slice
    A_pos: slice
    A_quat: slice
    B_pos: slice
    B_quat: slice
    finger1: int
    finger2: int
    lf_A: int
    rf_A: int
    A_B: int
    A_table: int
    B_table: int
    hand_B: int
    hand_table: int
    A_vel: slice
    B_vel: slice


class StackEnv:
    """Model, data, sensor layout, state (de)serialisation and IK."""

    def __init__(self, xml_path: str | Path = SCENE_XML) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        name2id = lambda t, n: int(mujoco.mj_name2id(m, t, n))  # noqa: E731
        self.arm_qadr = np.array([m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                                  for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                                  for i in range(1, 8)])
        self.finger_qadr = np.array([m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")]
                                     for i in (1, 2)])
        self.A_qadr = int(m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, "cubeA_free")])
        self.B_qadr = int(m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, "cubeB_free")])
        self.A_dadr = int(m.jnt_dofadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, "cubeA_free")])
        self.B_dadr = int(m.jnt_dofadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, "cubeB_free")])
        self.pinch_sid = name2id(mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.table_gid = name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.A_bid = name2id(mujoco.mjtObj.mjOBJ_BODY, "cubeA")
        self.B_bid = name2id(mujoco.mjtObj.mjOBJ_BODY, "cubeB")
        self.A_gid = name2id(mujoco.mjtObj.mjOBJ_GEOM, "cubeA_geom")
        self.B_gid = name2id(mujoco.mjtObj.mjOBJ_GEOM, "cubeB_geom")
        self.finger_bids = (name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                            name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"))
        link0 = name2id(mujoco.mjtObj.mjOBJ_BODY, "link0")
        active = {name2id(mujoco.mjtObj.mjOBJ_BODY, n) for n in ACTIVE_COLLISION_BODIES}
        self.robot_bids = set()
        for b in range(m.nbody):
            anc = b
            while anc > 0 and anc != link0:
                anc = int(m.body_parentid[anc])
            if anc == link0:
                self.robot_bids.add(b)  # gravcomp="1" is set in the XML (compile-time)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in self.robot_bids and b not in active and m.geom_contype[g] != 0:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        # Finger contact through the pad boxes only: the convex hull of the finger
        # mesh touches the cube first and makes grasps slip.
        for g in range(m.ngeom):
            if (int(m.geom_bodyid[g]) in self.finger_bids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    and m.geom_contype[g] != 0):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
            if int(m.geom_bodyid[g]) in self.finger_bids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_friction[g] = (1.2, 0.02, 0.002)
                m.geom_condim[g] = 4
        self.jnt_low = np.array([m.jnt_range[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"), 0] for i in range(1, 8)])
        self.jnt_high = np.array([m.jnt_range[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"), 1] for i in range(1, 8)])
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()

        def sl(name):
            sid = name2id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            a, d = int(m.sensor_adr[sid]), int(m.sensor_dim[sid])
            return slice(a, a + d) if d > 1 else a
        self.S = SensorLayout(**{f: sl(f) for f in SensorLayout.__dataclass_fields__})
        self.nsensordata = int(m.nsensordata)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = int(mujoco.mj_stateSize(m, self.state_spec))
        self._ik_data = mujoco.MjData(m)
        self.reset_home()

    # ------------------------------------------------------------------ state
    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def get_state(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        out = np.empty(self.nstate)
        mujoco.mj_getState(self.model, d, out, self.state_spec)
        return out

    def set_state(self, state: np.ndarray, ctrl: np.ndarray | None = None, data=None) -> None:
        d = self.data if data is None else data
        mujoco.mj_setState(self.model, d, np.asarray(state, dtype=np.float64), self.state_spec)
        if ctrl is not None:
            d.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, d)

    def sensors(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.asarray(d.sensordata, dtype=np.float64).copy()

    def arm_q(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.asarray(d.qpos[self.arm_qadr]).copy()

    def gripper_width(self, data=None) -> float:
        d = self.data if data is None else data
        return float(np.sum(d.qpos[self.finger_qadr]))

    def reset_home(self, A_pose=(0.5, -0.12, 0.0), B_pose=(0.5, 0.10, 0.0)) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr] = GRIPPER_MAX_WIDTH / 2
        self.place_cube("A", *A_pose)
        self.place_cube("B", *B_pose)
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, d)

    def place_cube(self, which: str, x: float, y: float, yaw: float, z: float = CUBE_HALF, data=None,
                   quat=None) -> None:
        d = self.data if data is None else data
        adr = self.A_qadr if which == "A" else self.B_qadr
        dadr = self.A_dadr if which == "A" else self.B_dadr
        d.qpos[adr:adr + 3] = (x, y, z)
        d.qpos[adr + 3:adr + 7] = yaw_quat(yaw) if quat is None else quat
        d.qvel[dadr:dadr + 6] = 0.0

    # --------------------------------------------------------------- kinematics
    def pinch_pose(self, data=None) -> tuple[np.ndarray, np.ndarray]:
        d = self.data if data is None else data
        return d.site_xpos[self.pinch_sid].copy(), d.site_xmat[self.pinch_sid].reshape(3, 3).copy()

    def pinch_jacobian(self, data) -> np.ndarray:
        """6x7 arm Jacobian (linear; angular) of the pinch site."""
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, self.pinch_sid)
        return np.vstack([jp[:, self.arm_dadr], jr[:, self.arm_dadr]])

    def solve_ik(self, pos, rot, q_init=None, *, iters: int = 120, tol_pos=1e-3, tol_rot=1e-2,
                 posture_gain: float = 0.05):
        """Damped least-squares IK for the pinch site. Returns (q, success)."""
        d = self._ik_data
        q = np.array(HOME_Q if q_init is None else q_init, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p, R = d.site_xpos[self.pinch_sid], d.site_xmat[self.pinch_sid].reshape(3, 3)
            ep = pos - p
            er = 0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1]) + np.cross(R[:, 2], rot[:, 2]))
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(er) < tol_rot:
                return q, True
            J = self.pinch_jacobian(d)
            e = np.concatenate([ep, er])
            lam = 1e-3
            JJt = J @ J.T + lam * np.eye(6)
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq = Jpinv @ e
            null = np.eye(7) - Jpinv @ J
            dq += null @ (posture_gain * (HOME_Q - q))
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.jnt_low + 1e-3, self.jnt_high - 1e-3)
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        p, R = d.site_xpos[self.pinch_sid], d.site_xmat[self.pinch_sid].reshape(3, 3)
        ep = np.linalg.norm(pos - p)
        er = np.linalg.norm(0.5 * (np.cross(R[:, 0], rot[:, 0]) + np.cross(R[:, 1], rot[:, 1])
                                   + np.cross(R[:, 2], rot[:, 2])))
        return q, bool(ep < 3 * tol_pos and er < 3 * tol_rot)

    # ------------------------------------------------------------- contacts
    def bad_penetration(self, data=None, *, allow_finger_A: bool = False, depth: float = 0.002) -> bool:
        """True if any contact penetrates more than ``depth`` (except allowed pairs)."""
        d = self.data if data is None else data
        m = self.model
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -depth:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if allow_finger_A and ({b1, b2} & set(self.finger_bids)) and self.A_bid in (b1, b2):
                continue
            return True
        return False
