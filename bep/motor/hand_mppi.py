"""Spike-MPPI for the floating hand: the robot's sampling planner on the learning's ideal body.

Same noise (the Spike motor-pool proposal) over the same channels as bep.motor.mppi.SpikeMPPI with 6-DoF
hand motions: pinch-point velocity (3), yaw rate, finger opening rate, roll and pitch rates.  Without an
arm there is no Jacobian: a rollout integrates the rates into a trajectory of hand poses, given to the
physics as the floating hand's mocap target (mujoco.rollout with mocap pose as control), and the cost is any
function of the rollout sensor data (in C++ when the cost has a native form: bep.motor.native.bridge), as for the robot (bep.motor.costs.PrimitiveCost: the primitive's field
Phi, the side-effect terms, the contact barrier and the discrete goal reward).  ``step`` / ``apply`` / ``reset`` match SpikeMPPI's,
so the same executor (bep.motor.execute) drives either body.
"""
from __future__ import annotations

import mujoco
import mujoco.rollout
import numpy as np

from ..world.env import GRIPPER_MAX_WIDTH, width_to_ctrl
from ..world.objects import PINCH
from .lbps import optimize_lbps_temperature
from .mppi import MPPIConfig, limit_transport_speed
from .spike_motor_pool import SpikeMotorPoolSampler

NCH = 7                       # vx vy vz (m/s), yaw rate, finger opening rate, roll and pitch rates (rad/s)
_SPEC = int(mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_MOCAP_POS | mujoco.mjtState.mjSTATE_MOCAP_QUAT)


def _quats(R: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) of rotation matrices (..., 3, 3)."""
    w = np.sqrt(np.maximum(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 1e-12)) / 2
    x = np.copysign(np.sqrt(np.maximum(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0)) / 2, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 0)) / 2, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], 0)) / 2, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _rot(w: np.ndarray) -> np.ndarray:
    """Rotation matrices of rotation vectors (..., 3) (Rodrigues)."""
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = w / np.maximum(th, 1e-12)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * K @ K


class HandMPPI:
    def __init__(self, env, cfg: MPPIConfig | None = None) -> None:
        assert env.floating, "HandMPPI moves the floating hand"
        self.env = env
        self.cfg = cfg = cfg or MPPIConfig()
        self.rng = np.random.default_rng(cfg.seed)
        env.model.opt.timestep = cfg.control_dt / cfg.substeps
        self.sub = int(cfg.substeps)
        self.std = np.array([cfg.lin_std] * 3 + [cfg.yaw_std, cfg.grip_std] + [cfg.tilt_std] * 2)
        self.vmax = np.array([cfg.max_lin] * 3 + [cfg.max_yaw, cfg.max_grip] + [cfg.max_tilt] * 2)
        if cfg.sampling == "spike":
            self.sampler = SpikeMotorPoolSampler(
                n=cfg.num_rollouts, h=cfg.horizon, nu=NCH, dt=cfg.control_dt,
                levels=cfg.recruitment_levels, base_rate_hz=cfg.base_rate_hz,
                twitch_rise_s=cfg.twitch_rise_s, twitch_decay_s=cfg.twitch_decay_s,
                twitch_duration_s=cfg.twitch_duration_s, backend="auto")
            self.spike_scale = 1.0 / max(self.sampler.estimate_unit_rms(), 1e-12)
        n, steps = cfg.num_rollouts, cfg.horizon * self.sub
        self.runner = mujoco.rollout.Rollout(nthread=cfg.threads)
        self._models = [env.model] * n
        self._datas = [mujoco.MjData(env.model) for _ in range(cfg.threads)]
        self._sens_out = np.empty((n, steps, env.model.nsensordata))
        self.reset()

    def reset(self, q_cmd=None, g_cmd=None) -> None:
        self.U = np.zeros((self.cfg.horizon, NCH))
        self.g_cmd = None if g_cmd is None else float(g_cmd)

    def _noise(self) -> np.ndarray:
        n, h = self.cfg.num_rollouts, self.cfg.horizon
        if self.cfg.sampling == "spike":
            z, _ = self.sampler.sample(self.rng)
            eps = z * self.spike_scale
        else:
            eps = self.rng.standard_normal((n, h, NCH))
        eps = eps * self.std
        eps[0] = 0.0                     # the warm-started nominal itself
        return eps

    def _target(self, data):
        """The mocap target's pinch point and rotation now."""
        env = self.env
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, data.mocap_quat[env.mocap_id])
        R = R.reshape(3, 3)
        return data.mocap_pos[env.mocap_id] + R @ PINCH, R

    def _integrate(self, p0, R0, g0, V, grip_limit=GRIPPER_MAX_WIDTH):
        """Hand poses and gripper commands along rate plans V (n, h, NCH)."""
        dt = self.cfg.control_dt
        n, h = V.shape[:2]
        P = p0 + np.cumsum(V[:, :, :3], axis=1) * dt
        W = np.stack([V[:, :, 5], V[:, :, 6], V[:, :, 3]], axis=-1) * dt      # world-frame rotation per step
        increments = _rot(W)
        R = np.empty((n, h, 3, 3))
        cur = np.broadcast_to(R0, (n, 3, 3))
        for t in range(h):
            cur = increments[:, t] @ cur
            R[:, t] = cur
        G = np.clip(g0 + np.cumsum(V[:, :, 4], axis=1) * dt, 0.0, grip_limit)
        return P, R, G

    def step(self, data, cost_fn):
        env, cfg = self.env, self.cfg
        n, h = cfg.num_rollouts, cfg.horizon
        p0, R0 = self._target(data)
        g0 = env.gripper_width(data) if self.g_cmd is None else self.g_cmd
        grip_limit = getattr(cost_fn, "grip_limit", GRIPPER_MAX_WIDTH)
        V = np.clip(self.U[None] + self._noise(), -self.vmax, self.vmax)
        limit_transport_speed(V, cost_fn)
        V[1] = 0.0                       # a 'hold still' candidate is always available
        P, R, G = self._integrate(p0, R0, g0, V, grip_limit)
        body = P - np.einsum("nhij,j->nhi", R, PINCH)                    # mocap = the hand body's frame
        ctrl = np.concatenate([width_to_ctrl(G)[..., None], body, _quats(R)], axis=-1)
        if not hasattr(cost_fn, "native"):
            from .native.bridge import attach
            attach(cost_fn)
        if cost_fn.native:
            # fused native rollouts: physics and cost in one C++ pass (bep.motor.native.bridge), as on the arm
            from .native.bridge import engine
            costs = np.array(engine(cfg.threads).evaluate(cost_fn, env.model, env.get_state(data), ctrl, self.sub,
                                                          mocap=True))
        else:
            ctrl = np.ascontiguousarray(np.repeat(ctrl, self.sub, axis=1))
            x0 = np.ascontiguousarray(np.tile(env.get_state(data), (n, 1)))
            self.runner.rollout(self._models, self._datas, x0, ctrl, control_spec=_SPEC, skip_checks=True,
                                nstep=h * self.sub, sensordata=self._sens_out,
                                chunk_size=max(1, n // cfg.threads))
            costs = np.asarray(cost_fn(self._sens_out[:, self.sub - 1::self.sub].copy()), float)
        costs = costs + cfg.effort_weight * np.sum((V / self.vmax) ** 2, axis=(1, 2))
        costs = np.where(np.isfinite(costs), costs, np.inf)
        lb = optimize_lbps_temperature(costs, delta=cfg.lbps_delta, fallback_temperature=1.0)
        w = np.zeros(n)
        finite = np.isfinite(costs)
        if finite.any():
            c = costs[finite]
            w[finite] = np.exp(-(c - c.min()) / max(lb.temperature, 1e-12))
            w /= w.sum()
        else:
            w[0] = 1.0
        U_new = np.einsum("n,ntc->tc", w, V)
        p1, R1, g1 = self._integrate(p0, R0, g0, U_new[None, :1], grip_limit)
        self.g_cmd = float(g1[0, 0])
        self.U = np.vstack([U_new[1:], U_new[-1:]])
        return (p1[0, 0], R1[0, 0], self.g_cmd), {"best_cost": float(costs.min()), "temperature": float(lb.temperature)}

    def apply(self, data, action) -> None:
        p, R, g = action
        self.env.move_hand(p, R, data)
        data.ctrl[self.env.gi] = float(width_to_ctrl(g))
        mujoco.mj_step(self.env.model, data, nstep=self.sub)
