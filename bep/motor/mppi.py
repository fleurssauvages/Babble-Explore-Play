"""Task-space Spike-MPPI for the Franka + gripper.

Motion channels: end-effector linear velocity and angular rates. The gripper
samples open/close schedules. Motion noise is drawn by the Spike motor-pool proposal (one
antagonistic motor pool per channel) or by white Gaussian noise (baseline).

The velocity plan is integrated into joint position targets with the pinch
Jacobian evaluated along the nominal trajectory (the same linearisation as the
"effector" noise space of Franka_MPPI), keeping the gripper pointing down and a
null-space posture term.  Rollouts are full MuJoCo physics via the native
multithreaded ``mujoco.rollout.Rollout``; the cost is any function of the
rollout sensor data, so the controller is agnostic to where the cost comes
from (hand-designed, local transition target, or the learned field).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable

import mujoco
import mujoco.rollout
import numpy as np

from ..config import HOG
from ..world.env import GRIPPER_MAX_WIDTH, StackEnv, ee_down_rot
from .lbps import optimize_lbps_temperature
from .spike_motor_pool import SpikeMotorPoolSampler

NCH = 4          # vx vy vz (m/s), yaw rate (rad/s); + roll and pitch rates with free_tilt


@dataclass
class MPPIConfig:
    sampling: str = "spike"          # spike | gaussian
    horizon: int = 50
    num_rollouts: int = 32
    control_dt: float = 0.01         # 100 Hz control (as Franka_MPPI)
    substeps: int = 2                # physics steps per control step: timestep = control_dt / substeps
    rollout_substeps: int | None = 2 # the rollouts' physics steps per control step (None: substeps): fewer, coarser
                                     # steps look as far ahead for less; the real simulation keeps ``substeps``
    lin_std: float = 0.20            # m/s
    yaw_std: float = 1.0             # rad/s
    max_lin: float = 0.35
    max_yaw: float = 1.5
    ori_gain: float = 8.0            # keep the gripper pointing down (1/s)
    elbow_gain: float = 1.0          # null space (N = I - J+ J): qdot_des = elbow_gain (elbow_q - q) on joint elbow_joint
    elbow_joint: int = 0             # only (elbow up), the hand's motion untouched (1/s)
    elbow_q: float = 0.0
    tilt_gain: float = 1.0           # free_tilt: weak pull of roll and pitch toward pointing down (1/s), added to the
                                     # sampled rates (undirected noise decays; a tilt the cost pays for is still taken)
    manip_gain: float = 0.0          # null-space ascent of log det(J J^T) of the pinch Jacobian (rad/s per unit)
    effort_weight: float = 0.002     # cost on commanded velocity (per step)
    lbps_delta: float = 0.95
    lead_limit: float = 0.15         # anti-windup: |q_cmd - q| (rad)
    threads: int = 8
    # Spike proposal (defaults of Franka_MPPI)
    base_rate_hz: float = 8.0
    recruitment_levels: int = 6
    twitch_rise_s: float = 0.016
    twitch_decay_s: float = 0.064
    twitch_duration_s: float = 0.200
    seed: int = 0
    # orientation: 6-DoF by default (free_tilt: roll and pitch rates are two more sampled channels, nothing
    # servos them); free_tilt=False keeps the gripper pointing down (roll and pitch servoed by ori_gain).
    free_tilt: bool = True
    # the rollouts' physics: no-slip iterations while simulating the rollouts (None: the model's own).  The real
    # simulation keeps the model's; the rollouts skip that second solver pass (~35% of their time)
    rollout_noslip: int | None = 0
    tilt_std: float = 0.5            # rad/s
    max_tilt: float = 1.0            # rad/s


CostFn = Callable[[np.ndarray], np.ndarray]   # sensordata (N, H, ns) -> costs (N,)


def rollout_substeps(cfg) -> int:
    return int(cfg.rollout_substeps or cfg.substeps)


@contextmanager
def _nothing():
    yield


def rollout_options(model, cfg):
    """The model's physics options for the rollouts (cfg.rollout_noslip; the timestep of cfg.rollout_substeps),
    restored afterwards for the real step."""
    if cfg.rollout_noslip is None and rollout_substeps(cfg) == int(cfg.substeps):
        return _nothing()
    return _rollout_options(model, cfg)


@contextmanager
def _rollout_options(model, cfg):
    old = model.opt.noslip_iterations, model.opt.timestep
    if cfg.rollout_noslip is not None:
        model.opt.noslip_iterations = int(cfg.rollout_noslip)
    model.opt.timestep = cfg.control_dt / rollout_substeps(cfg)
    try:
        yield
    finally:
        model.opt.noslip_iterations, model.opt.timestep = old


class BinaryGripper:
    """Sample open/close schedules and keep the selected sequence for the next horizon."""

    def __init__(self, rng, horizon: int, width: float | None):
        self.rng = rng
        self.plan = None if width is None else np.full(horizon, width >= GRIPPER_MAX_WIDTH / 2)

    def sample(self, n: int, h: int, command: float, carrying: bool) -> np.ndarray:
        if carrying:
            return np.zeros((n, h), dtype=bool)
        open_now = command >= 127.5
        first = np.arange(n) % 2 == 0
        switch = self.rng.integers(1, h + 1, size=n)
        switch[np.arange(n) % 4 < 2] = h
        plans = first[:, None] ^ (np.arange(h) >= switch[:, None])
        if n > 2:
            plans[0] = open_now if self.plan is None else self.plan
            plans[1] = open_now
            plans[2] = not open_now
        return plans

    @staticmethod
    def switch_cost(plans: np.ndarray, command: float) -> np.ndarray:
        """HOG['grip_switch_cost'] per change of the gripper along each sampled schedule (from the current command):
        where the gripper does not matter (approaching, nothing between the fingers) keeping it is strictly
        cheaper, so it is not switched at random; a switch the goal reward pays for still wins."""
        open_now = command >= 127.5
        n_switch = (plans[:, 0] != open_now).astype(float) + (plans[:, 1:] != plans[:, :-1]).sum(axis=1)
        return HOG["grip_switch_cost"] * n_switch

    def select(self, plans: np.ndarray, weights: np.ndarray, command: float) -> tuple[bool, np.ndarray]:
        """Choose the first command by posterior mass, then average motion conditional on it."""
        opens = plans[:, 0]
        mass = float(weights @ opens)
        open_next = mass > 0.5 if mass != 0.5 else command >= 127.5
        conditional = weights * (opens == open_next)
        conditional /= conditional.sum()
        best = plans[np.argmax(conditional)]
        self.plan = np.r_[best[1:], best[-1]]
        return open_next, conditional


def limit_transport_speed(velocities, cost) -> None:
    """Stay within the speed envelope used to validate carrying motions."""
    if getattr(cost, "carry_mode", 0):
        linear = velocities[..., :3]
        scale = np.maximum(1.0, np.linalg.norm(linear, axis=-1, keepdims=True) / HOG["v_max"])
        linear /= scale


class SpikeMPPI:
    def __init__(self, env: StackEnv, cfg: MPPIConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg = cfg or MPPIConfig()
        self.rng = np.random.default_rng(cfg.seed)
        self._native = None        # fused native kinematics (bep.motor.native.bridge), resolved on first use
        env.model.opt.timestep = cfg.control_dt / cfg.substeps
        self.sub = int(cfg.substeps)                    # the real simulation's physics steps per control step
        self.rsub = rollout_substeps(cfg)               # the rollouts'
        self.nch = NCH + 2 * bool(cfg.free_tilt)
        self.std = np.array([cfg.lin_std] * 3 + [cfg.yaw_std] + [cfg.tilt_std] * 2 * cfg.free_tilt)
        self.vmax = np.array([cfg.max_lin] * 3 + [cfg.max_yaw] + [cfg.max_tilt] * 2 * cfg.free_tilt)
        if cfg.sampling == "spike":
            self.sampler = SpikeMotorPoolSampler(
                n=cfg.num_rollouts, h=cfg.horizon, nu=self.nch, dt=cfg.control_dt,
                levels=cfg.recruitment_levels, base_rate_hz=cfg.base_rate_hz,
                twitch_rise_s=cfg.twitch_rise_s, twitch_decay_s=cfg.twitch_decay_s,
                twitch_duration_s=cfg.twitch_duration_s, backend="auto")
            self.spike_scale = 1.0 / max(self.sampler.estimate_unit_rms(), 1e-12)
        elif cfg.sampling != "gaussian":
            raise ValueError("sampling must be spike or gaussian")
        self.runner = mujoco.rollout.Rollout(nthread=cfg.threads)
        self._kin = mujoco.MjData(env.model)
        self.set_rollout_world(None)
        self.reset()

    def set_rollout_world(self, model=None, state_fn=None) -> None:
        """Simulate the rollouts in another (smaller) model: ``state_fn(data)`` maps the real state to it.
        None restores the controlled model itself."""
        model = self.env.model if model is None else model
        model.opt.timestep = self.cfg.control_dt / self.cfg.substeps
        n, steps = self.cfg.num_rollouts, self.cfg.horizon * self.rsub
        self._rmodel, self._state_fn = model, state_fn
        self._models = [model] * n
        self._datas = [mujoco.MjData(model) for _ in range(self.cfg.threads)]
        self._sens_out = np.empty((n, steps, model.nsensordata))

    def close(self) -> None:
        try:
            self.runner.close()
        except Exception:
            pass

    def reset(self, q_cmd=None, g_cmd=None) -> None:
        self.noise_scale = 1.0     # set per step by primitives acting on constrained objects
        self.U = np.zeros((self.cfg.horizon, self.nch))
        self.q_cmd = None if q_cmd is None else np.asarray(q_cmd, dtype=float).copy()
        self.gripper = BinaryGripper(self.rng, self.cfg.horizon, g_cmd)

    # ------------------------------------------------------------ sampling
    def _noise(self) -> np.ndarray:
        n, h = self.cfg.num_rollouts, self.cfg.horizon
        if self.cfg.sampling == "spike":
            z, _ = self.sampler.sample(self.rng)
            eps = z * self.spike_scale
        else:
            eps = self.rng.standard_normal((n, h, self.nch))
        eps = eps * self.std * getattr(self, "noise_scale", 1.0)
        eps[0] = 0.0                     # the warm-started nominal itself
        return eps

    # ------------------------------------------------------- kinematic map
    def _nominal_maps(self, q_base: np.ndarray, U: np.ndarray):
        """Integrate the nominal plan; return per-step 7 x (hand channels) maps and joint targets."""
        env, cfg, d = self.env, self.cfg, self._kin
        h = cfg.horizon
        if self._native is None:
            from .native.bridge import enabled, engine
            self._native = engine(cfg.threads).f if enabled() else False
            self._nargs = (np.asarray(env.arm_qadr, np.int32), np.asarray(env.arm_dadr, np.int32), int(env.pinch_sid),
                           float(cfg.ori_gain), float(cfg.tilt_gain), float(cfg.elbow_gain), int(cfg.elbow_joint),
                           float(cfg.elbow_q),
                           float(cfg.control_dt), np.asarray(env.jnt_low, float), np.asarray(env.jnt_high, float))
        if self._native:
            a = self._nargs
            return self._native.nominal_maps(env.model._address, np.asarray(q_base, float), np.ascontiguousarray(U),
                                             *a, float(cfg.manip_gain))
        P = np.empty((h, 7, self.nch))
        Qn = np.empty((h, 7))
        q = q_base.copy()
        for t in range(h):
            d.qpos[env.arm_qadr] = q
            mujoco.mj_kinematics(env.model, d)
            mujoco.mj_comPos(env.model, d)
            R = d.site_xmat[env.pinch_sid].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            Rd = ee_down_rot(yaw)
            er = 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
            J = env.pinch_jacobian(d)
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            if cfg.free_tilt:                      # roll and pitch rates are commanded, not servoed
                P[t] = Jp[:, [0, 1, 2, 5, 3, 4]]
                twist = np.array([U[t, 0], U[t, 1], U[t, 2], U[t, 4] + cfg.tilt_gain * er[0],
                                  U[t, 5] + cfg.tilt_gain * er[1], U[t, 3]])
            else:
                P[t] = Jp[:, [0, 1, 2, 5]]
                twist = np.array([U[t, 0], U[t, 1], U[t, 2], cfg.ori_gain * er[0], cfg.ori_gain * er[1], U[t, 3]])
            # null space (N = I - J+ J, the hand's motion untouched): qdot_des toward elbow_q on joint elbow_joint only
            # (elbow up), and optionally ascend the manipulability log det(J J^T)
            null = np.zeros(7)
            null[cfg.elbow_joint] = cfg.elbow_gain * (cfg.elbow_q - q[cfg.elbow_joint])
            if cfg.manip_gain > 0:
                null = null + cfg.manip_gain * self._manip_grad(d, q)
            dq = Jp @ twist + (np.eye(7) - Jp @ J) @ null
            q = np.clip(q + dq * cfg.control_dt, env.jnt_low, env.jnt_high)
            Qn[t] = q
        return P, Qn

    def _manip_grad(self, d, q, h: float = 1e-3) -> np.ndarray:
        """Gradient of log det(J J^T) (pinch-site Jacobian) w.r.t. the arm joints, by finite differences."""
        env = self.env

        def logdet(qq):
            d.qpos[env.arm_qadr] = qq
            mujoco.mj_kinematics(env.model, d)
            mujoco.mj_comPos(env.model, d)
            J = env.pinch_jacobian(d)
            return float(np.linalg.slogdet(J @ J.T + 1e-6 * np.eye(6))[1])
        f0 = logdet(q)
        g = np.array([(logdet(q + h * np.eye(7)[i]) - f0) / h for i in range(7)])
        logdet(q)                                      # leave d at q
        return g

    # --------------------------------------------------------------- step
    def step(self, data, cost_fn: CostFn) -> tuple[np.ndarray, dict]:
        env, cfg = self.env, self.cfg
        n, h, dt = cfg.num_rollouts, cfg.horizon, cfg.control_dt
        q_meas = env.arm_q(data)
        q_base = q_meas if self.q_cmd is None else np.clip(
            self.q_cmd, q_meas - cfg.lead_limit, q_meas + cfg.lead_limit)
        command = float(data.ctrl[-1])

        eps = self._noise()
        V = np.clip(self.U[None] + eps, -self.vmax, self.vmax)
        limit_transport_speed(V, cost_fn)
        V[1] = 0.0                       # a 'hold still' candidate is always available
        dV = V - self.U[None]            # effective perturbation after clipping
        P, Qn = self._nominal_maps(q_base, self.U)
        # Linearised joint targets: nominal + cumulative mapped perturbation.
        dq = np.einsum("tjc,ntc->ntj", P, dV) * dt
        Q = Qn[None] + np.cumsum(dq, axis=1)
        np.clip(Q, env.jnt_low, env.jnt_high, out=Q)
        grip_plans = self.gripper.sample(n, h, command, bool(getattr(cost_fn, "carry_mode", 0)))

        ctrl = np.empty((n, h, env.model.nu))
        ctrl[:, :, :7] = Q
        ctrl[:, :, 7] = 255.0 * grip_plans
        s0 = env.get_state(data) if self._state_fn is None else self._state_fn(data)
        if not hasattr(cost_fn, "native"):
            from .native.bridge import attach
            attach(cost_fn)
        with rollout_options(self._rmodel, cfg):     # the rollouts' physics (cfg.rollout_noslip)
            if cost_fn.native:
                # fused native rollouts: physics and cost in one C++ pass (bep.motor.native.bridge)
                from .native.bridge import engine
                costs = np.array(engine(cfg.threads).evaluate(cost_fn, self._rmodel, s0, ctrl, self.rsub))
                sens = None
            else:
                ctrl = np.ascontiguousarray(np.repeat(ctrl, self.rsub, axis=1))
                x0 = np.ascontiguousarray(np.tile(s0, (n, 1)))
                self.runner.rollout(self._models, self._datas, x0, ctrl, skip_checks=True,
                                    nstep=h * self.rsub, sensordata=self._sens_out,
                                    chunk_size=max(1, n // cfg.threads))
                sens = self._sens_out[:, self.rsub - 1::self.rsub].copy()   # after each control step
                costs = np.asarray(cost_fn(sens), dtype=float)
        costs = costs + cfg.effort_weight * np.sum((V / self.vmax) ** 2, axis=(1, 2))
        costs = costs + self.gripper.switch_cost(grip_plans, command)
        costs = np.where(np.isfinite(costs), costs, np.inf)

        lb = optimize_lbps_temperature(costs, delta=cfg.lbps_delta, fallback_temperature=1.0)
        finite = np.isfinite(costs)
        w = np.zeros(n)
        if np.any(finite):
            c = costs[finite]
            w[finite] = np.exp(-(c - c.min()) / max(lb.temperature, 1e-12))
            w /= w.sum()
        else:
            w[0] = 1.0
        open_next, motion_weights = self.gripper.select(grip_plans, w, command)
        U_new = np.einsum("n,ntc->tc", motion_weights, V)

        # Apply the first step of the updated plan.
        q_next = Qn[0] + P[0] @ (U_new[0] - self.U[0]) * dt
        q_next = np.clip(q_next, env.jnt_low, env.jnt_high)
        self.q_cmd = q_next
        self.U = np.vstack([U_new[1:], U_new[-1:]])
        action = np.concatenate([q_next, [255.0 * open_next]])
        best = int(np.argmin(costs))
        info = {"best_cost": float(costs[best]), "mean_cost": float(np.sum(w * np.where(finite, costs, 0))),
                "temperature": float(lb.temperature), "ess": float(lb.ess),
                "best_sens": None if sens is None else sens[best]}
        return action, info

    def apply(self, data, action: np.ndarray) -> None:
        data.ctrl[:] = action
        mujoco.mj_step(self.env.model, data, nstep=self.sub)
