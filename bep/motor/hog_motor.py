"""The Hand-of-God motor: an ideal hand that goes where it is told, within the robot's reach.

Learning (explore, play) moves the floating hand (bep.world.objects, body "float") with this motor instead
of Spike-MPPI; the robot and its controller come in at validation and in the final tasks.

* ``HandMotor``: the hand moves in a straight line, speed-limited (``HOG['v_max']``, ``HOG['w_max']``),
  toward a hand pose.  A hop's target must lie in the robot's reachability map (bep.world.reach); the short
  straight path to it is not checked (a hand resting just outside the map must be able to come back).
* ``hand_target``: where a graph node of a primitive puts the hand, given the current scene.  The node's
  configuration says where X is relative to Y and the hand relative to X (bep.explore.primitive).  If the
  hand holds X (a hand relation on X holds), X is carried to its place relative to Y with the current grip;
  otherwise the hand goes to its place relative to X.  Dimensions the primitive does not use (a hand-object
  primitive's X position on the table) keep their current value.
* a *hop* to a node: its hand pose (fixed in the world when the hop starts) and its gripper command
  (open, close, hold an opening): a body in hand is gripped tighter before it is carried; otherwise the
  command is applied on arrival.
* ``hog_validate``: exploration's forward validation of a backward candidate (one hop toward the target
  node), with the same success test as with Spike-MPPI; ``run_primitive_hog``: a whole call, validated
  motions replayed exactly (straight hops along the primitive's roadmap) and everything else (reaching the
  graph) planned by Spike-MPPI on the floating hand with Phi and the barrier (bep.motor.hand_mppi).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..config import CONTROL_DT, EXEC, EXPLORE, GRIP_MAX, HOG, SUBSTEPS
from ..world.env import yaw_rot
from ..world.geometry import mat_rotvec, quat_to_mat, rotvec_mat
from ..world.graph import relation_graph
from ..world.objects import VIEW_S
from ..world.reach import reach_map
from ..world.relations import CC, HC


class HandMotor:
    def __init__(self, env):
        assert env.floating, "the Hand-of-God motor moves the floating hand"
        self.env = env
        d = env.data
        self.p = d.mocap_pos[env.mocap_id].copy()
        q = d.mocap_quat[env.mocap_id].copy()
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        from ..world.objects import PINCH
        self.R = R.reshape(3, 3)
        self.p = self.p + self.R @ PINCH                     # the target's pinch point
        self.dt = CONTROL_DT

    def step_toward(self, p_t, R_t) -> bool:
        """One control step toward the hand pose (p_t, R_t).  Returns arrived."""
        dp = np.asarray(p_t, float) - self.p
        n = float(np.linalg.norm(dp))
        vmax = HOG["v_max"] * self.dt
        p = self.p + (dp if n <= vmax else dp * vmax / n)
        w = mat_rotvec(np.asarray(R_t, float) @ self.R.T)
        a = float(np.linalg.norm(w))
        wmax = HOG["w_max"] * self.dt
        R = (rotvec_mat(w) if a <= wmax else rotvec_mat(w * wmax / a)) @ self.R
        self.p, self.R = p, R
        self.env.move_hand(p, R)
        mujoco.mj_step(self.env.model, self.env.data, nstep=SUBSTEPS)
        return n <= vmax and a <= wmax

    def hold(self) -> None:
        mujoco.mj_step(self.env.model, self.env.data, nstep=SUBSTEPS)


def _frame(view, which: str):
    """World position and yaw of role A or B from a pair view."""
    p = view[getattr(VIEW_S, f"{which}_pos")]
    q = view[getattr(VIEW_S, f"{which}_quat")]
    w, x, y, z = q
    return np.asarray(p, float), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def holding(prim, rel) -> bool:
    """A hand relation on X holds (the hand carries X)."""
    return bool(int(rel["raw_mode"]) & prim.carry_mask)


def hand_target(prim, env, view, rel, cfg):
    """The hand pose a node configuration asks for in the current scene (see the module docstring)."""
    pA, yawA = _frame(view, "A")
    pB, yawB = _frame(view, "B")
    pE, RE = env.hand_pose()
    keep_xy = prim.kind == HC and not prim.x_is_part             # only X's height matters
    if holding(prim, rel):
        # X carried to its place relative to Y, the hand keeping its current grip on X
        if keep_xy:
            pX, yawX = np.array([pA[0], pA[1], pB[2] + cfg.pXY[2]]), yawA
        else:
            pX, yawX = pB + yaw_rot(yawB) @ np.asarray(cfg.pXY, float), yawB + cfg.yawXY
        Rx_now = quat_to_mat(view[VIEW_S.A_quat])
        Rx_new = yaw_rot(yawX)
        if cfg.tilt_x is not None:
            Rx_new = Rx_new @ cfg.tilt_x
        return pX + Rx_new @ (Rx_now.T @ (pE - pA)), Rx_new @ (Rx_now.T @ RE)
    return pA + yaw_rot(yawA) @ np.asarray(cfg.pEX, float), yaw_rot(yawA) @ np.asarray(cfg.rEX, float)


class Hop:
    """Go to a node: its hand pose and its gripper command.  Secure what the hand holds before moving: when the
    hand holds X and the node's command closes the fingers further, the command is applied first (and given
    HOG['grip_s'] to act), so that X is gripped before it is carried; otherwise (approaching, letting go) the
    command is applied on arrival, so that the fingers close around a body once they are there and a carried
    body is let go once it is in place.  The pose target is fixed in the world when the hop starts (where
    the node puts the hand in the scene as it is then): a target that followed X would chase a body the hand
    pushes instead of holding.

    ``check_grip`` (execution): fingers that close on a body not yet held close only where the node puts them
    relative to X.  On arrival, if X moved during the approach (the hand pushed it) and the hand is off the
    node's place relative to X by more than EXPLORE['eps'] (X's half extents), the hop re-aims from the scene
    as it is (at most HOG['regrip_aims'] times), then gives up ("unreachable") without closing: fingers closed
    on a corner hold nothing and leave the hand unable to grasp again."""

    def __init__(self, env, motor, prim, cfg, rel, check_grip: bool = False):
        self.env, self.motor, self.prim, self.cfg = env, motor, prim, cfg
        self.target, self.t_grip = None, 0.0
        cmd, now = cfg.grip_ctrl(), float(env.data.ctrl[env.gi])
        tighter = cmd < now - 1.0                                 # gripper command: 0 closed ... 255 open
        self.phase = "tighten" if tighter and holding(prim, rel) else "move"
        self.check = check_grip and tighter and not holding(prim, rel) and not prim.x_is_part
        self.aims = 0

    def _off_place(self, view) -> bool:
        """The hand is farther than EXPLORE['eps'] (in X's half extents) from the node's place relative to X."""
        pA, yawA = _frame(view, "A")
        pE, _ = self.env.hand_pose()
        d = yaw_rot(yawA).T @ (pE - pA) - np.asarray(self.cfg.pEX, float)
        return float(np.max(np.abs(d) / view[VIEW_S.A_ext])) > EXPLORE["eps"]

    def step(self, view, rel) -> str:
        """One control step: "moving", "gripping", "done" or "unreachable"."""
        if self.phase == "tighten":
            self.env.data.ctrl[self.env.gi] = self.cfg.grip_ctrl()
            self.motor.hold()
            self.t_grip += self.motor.dt
            if self.t_grip >= HOG["grip_s"]:
                self.phase, self.t_grip = "move", HOG["grip_s"]      # the grip has acted: nothing left after the move
            return "gripping"
        if self.phase == "move":
            if self.target is None:
                self.target = hand_target(self.prim, self.env, view, rel, self.cfg)
                if not bool(reach_map().reachable(*self.target)):
                    return "unreachable"
            if self.motor.step_toward(*self.target):
                if self.check and self._off_place(view):
                    if self.aims >= HOG["regrip_aims"]:
                        return "unreachable"
                    self.aims += 1
                    self.target = None                            # re-aim from the scene as it is now
                    return "moving"
                self.env.data.ctrl[self.env.gi] = self.cfg.grip_ctrl()
                self.phase = "grip"
            return "moving"
        self.motor.hold()
        self.t_grip += self.motor.dt
        return "done" if self.t_grip >= HOG["grip_s"] else "gripping"


def make_hop(env, motor, prim, cfg, rel, learning: bool = False):
    """The hop toward a node: the hand to the node's configuration (execution, not ``learning``: fingers close
    only at the node's place relative to X, see Hop)."""
    return Hop(env, motor, prim, cfg, rel, check_grip=not learning)


# ------------------------------------------------------------------ exploration: forward validation
def hog_validate(prim, env, rel0, task, out, view_fn):
    """One hop toward the task's target node (configuration ``task['cfg_target']``) from the realised state;
    success = the goal held, or a solved node of cost-to-go <= task['D_max'] reached within eps (in the
    current mode), as with Spike-MPPI (bep.explore.explore._validate)."""
    from ..explore.explore import TRANSIENT
    Zs, Ms, Ds, ids = task["solved"]
    eligible = Ds <= task["D_max"] + 1e-9
    by_mode = {}
    m0 = int(rel0["mode"])
    motor = HandMotor(env)
    hop = make_hop(env, motor, prim, task["cfg_target"], rel0, learning=True)
    traj_z, traj_m = [rel0["z"]], [m0]
    goal_hold, dt = 0, motor.dt
    best, since = np.inf, 0
    stall_steps = int(round(EXPLORE["stall_s"] / dt))
    rel, view = rel0, view_fn()
    for step in range(task["max_steps"]):
        st = hop.step(view, rel)
        if st == "unreachable":
            out["status"] = "fail:unreachable"
            return out
        out["sim_steps"] = step + 1
        view = view_fn()
        rel = prim.rel(env, view, VIEW_S)
        m = int(rel["mode"])
        traj_z.append(rel["z"])
        traj_m.append(TRANSIENT if prim.transient(rel) else m)
        why = prim.failed(rel)
        if why:
            out["status"] = "fail:" + why
            return out
        goal_hold = goal_hold + 1 if bool(prim.is_goal(rel)) else 0
        reached, Dj = -1, np.inf
        if goal_hold * dt >= EXPLORE["goal_hold_s"]:
            reached, Dj = -2, 0.0
        elif m != prim.goal_mode:
            if m not in by_mode:
                sel = np.nonzero((Ms == m) & eligible)[0]
                by_mode[m] = (Zs[sel], Ds[sel], ids[sel])
            candidates, costs, indices = by_mode[m]
            if len(candidates):
                dd = prim.distance(rel["z"], candidates, m)
                hit = np.flatnonzero(dd < prim.eps[m])
                if len(hit):
                    b = hit[np.argmin(costs[hit])]
                    reached, Dj = int(indices[b]), float(costs[b])
        if reached != -1:
            out.update(status="success", z0=rel0["z"], m0=m0, reached=reached, D_reached=Dj, cost=(step + 1) * dt,
                       traj_z=np.array(traj_z), traj_m=np.array(traj_m), dt=dt)
            return out
        if st == "done":                                     # the hop is over and nothing was reached
            out["status"] = "fail:missed"
            return out
        d = float(prim.goal_residual(rel)) if task["m_target"] == prim.goal_mode and not prim.goal_requires_lift else \
            float(prim.distance(rel["z"], task["z_target"], task["m_target"]))
        d += 1.0 * (m != task["m_target"])
        if d < best * (1.0 - EXPLORE["stall_rel"]):
            best, since = d, 0
        elif st == "moving":
            since += 1
            if since >= stall_steps:
                out["status"] = "fail:stalled"
                return out
    out["status"] = "fail:timeout"
    return out


# ------------------------------------------------------------------ execution of a call
def _hand_ctl(env):
    """The floating hand's Spike-MPPI (one per world; one thread inside pool workers)."""
    ctl = getattr(env, "_hand_ctl", None)
    if ctl is None:
        import multiprocessing as mp
        import os
        from .hand_mppi import HandMPPI
        from .mppi import MPPIConfig
        threads = 1 if mp.current_process().name != "MainProcess" else min(8, os.cpu_count() or 1)
        ctl = env._hand_ctl = HandMPPI(env, MPPIConfig(threads=threads, seed=0, horizon=HOG["horizon"]))
    return ctl


def run_primitive_hog(env, R, prim, field, x: str, target, *, timeout: float = EXEC["timeout_s"],
                      on_step=None, stop=None, log: list | None = None, ctl=None) -> dict:
    """A call (primitive, X, target) on the floating hand, planned by Spike-MPPI (bep.motor.hand_mppi) with the
    primitive's cost (bep.motor.costs.PrimitiveCost): Phi, the barrier, the keep and leave-alone terms
    and the discrete goal reward; no waypoint (the graph shaped Phi during exploration).
    Ends when the goal holds for EXEC['hold_s'] (done), Phi stops improving for EXEC['stall_s'] (stalled),
    a watched relation breaks (broken), or at the timeout (same results as bep.motor.execute.run_primitive).

    Recovery (up to EXEC['recoveries'] per call): when Phi stalls, the call returns to the primitive's validated
    graph, as Go-Explore returns to a cell: it enters at the nearest node of low cost-to-go (Roadmap.entry) and
    follows the graph's edges, each hop aimed as during validation (Roadmap.aim, PrimitiveCost.set_waypoint),
    until a node of lower cost-to-go is reached; a hop that does not reach one within EXEC['hop_stall'] typical
    edge times ends the recovery, and Phi takes over again."""
    from .costs import PrimitiveCost
    graph = relation_graph(env, R)
    watch = [i for i in graph if i.rel in [r.name for r in R.cc] and i.x != x]
    cost = PrimitiveCost(env, prim, field, x, target, keep=sorted(graph, key=str))
    pair = prim.role_pair(env, x, target)
    g_cmd = float(env.data.ctrl[env.gi] * GRIP_MAX / 255.0)
    initial = cost.rel_now(env.sensors())
    transports = prim.kind == CC
    carry = int(initial["mode"]) & prim.carry_mask if transports else 0
    if carry and initial["pinching"]:
        cost.set_transport(carry)
    if ctl is None:                                   # the floating hand's planner
        ctl = _hand_ctl(env)
        ctl.reset(g_cmd=g_cmd)
    else:                                             # the arm's (bep.motor.mppi.SpikeMPPI)
        ctl.reset(q_cmd=env.arm_q(), g_cmd=g_cmd)
    dt = ctl.cfg.control_dt
    every = max(1, int(round(EXEC["watch_s"] / dt)))
    stall = int(round(EXEC["stall_s"] / dt))
    contact_steps = max(1, int(round(EXEC["contact_grace_s"] / dt)))
    best, since, hold, recoveries = np.inf, 0, 0, 0
    aim, aim_entry, recovery_entry, failed_entries = None, -1, -1, set()
    supported, detached, released = 0, 0, False
    nav = field.roadmap
    hop_s = EXEC["hop_stall"] * (nav.edge_s + 0.5)
    for k in range(int(round(timeout / dt))):
        sensors = env.sensors()
        rel = cost.rel_now(sensors)
        if transports and not cost.carry_mode and not released and rel["pinching"]:
            carry = int(rel["mode"]) & prim.carry_mask
            if carry:
                cost.set_transport(carry)
                supported, detached = 0, 0
                cost.clear_waypoint()
                aim, best, since = None, np.inf, 0
        if cost.carry_mode:
            supported = supported + 1 if int(rel["mode"]) & prim.goal_mode else 0
            detached = 0 if rel["pinching"] else detached + 1
            if supported >= contact_steps or detached >= contact_steps:
                released = supported >= contact_steps
                if log is not None:
                    reason = "supported" if released else "detached"
                    log.append((round(k * dt, 2), "transport", reason))
                cost.set_transport(0)
                cost.clear_waypoint()
                aim, best, since = None, np.inf, 0
        mode = int(cost.transport_mode(rel))
        if cost.carry_mode and not mode & prim.carry_mask:
            mode |= cost.carry_mode
        carrying = bool(cost.carry_mode)
        if aim is not None:                                   # recovering: follow the graph's edges
            cur, k_hop = aim
            limit = nav.D[cur] if cur == aim_entry else nav.D[cur] - 1e-8
            j = nav.reached(rel["z"], mode, limit, pair=pair, carrying=carrying) if k > k_hop else None
            if j is not None and (j != cur or cur == aim_entry):
                aim_entry = -1
                if nav.goal[j]:
                    aim = None
                    cost.clear_waypoint()
                    best, since = np.inf, 0
                else:
                    aim = (j, k)
                    cost.set_waypoint(*nav.aim(j))
                if log is not None:
                    log.append((round(k * dt, 2), "node", int(j), round(float(nav.D[j]), 2)))
            elif (k - k_hop) * dt > hop_s:
                if log is not None:
                    log.append((round(k * dt, 2), "hop stalled", int(cur)))
                aim = None
                failed_entries.update((recovery_entry, cur))
                cost.clear_waypoint()
                best, since = np.inf, 0
        if cost.done(sensors):
            hold += 1
            if hold * dt >= EXEC["hold_s"]:
                return {"result": "done", "time": k * dt}
        else:
            hold = 0
        progress_mode = prim.goal_mode if cost.carry_mode and mode & prim.goal_mode else mode
        phi = float(cost.progress_value(rel["z"], progress_mode))
        if hold or phi < best * (1.0 - EXEC["stall_rel"]):
            best, since = min(best, phi), 0
        else:
            since += 1
            if since >= stall and aim is None and recoveries < EXEC["recoveries"]:
                recoveries += 1
                aim_entry = nav.entry(rel["z"], mode, exclude=failed_entries, pair=pair, carrying=carrying)
                recovery_entry, aim = aim_entry, (aim_entry, k)
                cost.set_waypoint(nav.Z[aim_entry], int(nav.M[aim_entry]))
                if log is not None:
                    log.append((round(k * dt, 2), "recover", int(rel["mode"]), round(phi, 2), "-> entry", aim_entry,
                                "D", round(float(nav.D[aim_entry]), 2)))
            elif since >= stall and aim is None:
                if log is not None:
                    log.append((round(k * dt, 2), "stalled", int(rel["mode"]), round(phi, 2)))
                return {"result": "stalled", "time": (k + 1) * dt}
        a, info = ctl.step(env.data, cost)
        ctl.apply(env.data, a)
        if log is not None and k % 20 == 0:
            log.append((round(k * dt, 2), int(rel["mode"]), round(phi, 2), round(info["best_cost"], 1)))
        if on_step is not None:
            on_step(k)
        if stop is not None and stop():
            return {"result": "interrupted", "time": (k + 1) * dt}
        if watch and k % every == every - 1:
            g = relation_graph(env, R)
            if any(w not in g for w in watch):
                return {"result": "broken", "time": (k + 1) * dt}
    return {"result": "timeout", "time": timeout}


def run_primitive_hop(env, R, prim, field, x: str, target, *, timeout: float = EXEC["timeout_s"],
                      on_step=None, stop=None) -> dict:
    """A call (primitive, X, target) by the Hand-of-God motor along the primitive's roadmap, as exploration
    validated it (the learning's motor: fast, no sampling planner): enter the graph at a nearby node of low
    cost-to-go, then straight hops from node to node of lower cost-to-go, until the goal holds for
    EXEC['hold_s'], a watched relation breaks, entries keep failing, or the timeout (same results as
    bep.motor.execute.run_primitive)."""
    nav = field.roadmap
    ref = prim.reference(env, x, target)
    pair = prim.role_pair(env, x, target)
    view_fn = lambda: env.pair_view(env.sensors(), x, ref)         # noqa: E731
    graph = relation_graph(env, R)
    watch = [i for i in graph if i.rel in [r.name for r in R.cc] and i.x != x]
    motor = HandMotor(env)
    dt = motor.dt
    every = max(1, int(round(EXEC["watch_s"] / dt)))
    failed: set = set()                                     # nodes not to enter again
    n_failed = 0

    def fail(i) -> bool:
        """Node i failed: it and the nodes sharing its successor (the same last step, retried the same way) are
        not entered again.  Returns whether the call has failed too often."""
        nonlocal n_failed
        n_failed += 1
        failed.add(int(i))
        if nav.succ[i] >= 0:
            failed.update(np.flatnonzero(nav.succ == nav.succ[i]).tolist())
        return n_failed > HOG["max_reentries"]
    cur = hop = None
    joining = False
    t_hop, hold = 0.0, 0
    hop_limit = EXEC["hop_stall"] * (nav.edge_s + 0.5) + HOG["grip_s"]
    transports = prim.kind == CC
    for k in range(int(round(timeout / dt))):
        view = view_fn()
        rel = prim.rel(env, view, VIEW_S)
        z, m = rel["z"], int(rel["mode"])
        # carrying X to its place (as run_primitive_hog): a carry hop keeps the grasp, so the hand's coordinates
        # relative to X are not compared with the nodes' (they cannot change until X is supported and let go)
        carrying = bool(transports and m & prim.carry_mask and rel.get("pinching", True) and not m & prim.goal_mode)
        if prim.is_goal(rel):
            hold += 1
            if hold * dt >= EXEC["hold_s"]:
                return {"result": "done", "time": k * dt}
        else:
            hold = 0
        if cur is not None:                                  # progress: a node of lower cost-to-go within eps
            limit = nav.D[cur] + 1e-9 if joining else nav.D[cur] - 1e-6
            r = nav.reached(z, m, limit, pair=pair, carrying=carrying)
            if r is not None and (joining or r != cur):
                cur, hop, t_hop = r, None, 0.0
                joining = False
        if cur is None or (hop is not None and t_hop > hop_limit):
            if cur is not None and fail(cur):
                return {"result": "stalled", "time": k * dt}
            cur, hop, t_hop = nav.entry(z, m, exclude=failed, pair=pair, carrying=carrying), None, 0.0
            joining = True
            hop = make_hop(env, motor, prim, prim.retarget(nav.cfg(cur), pair), rel)
        if hop is None:
            hop = make_hop(env, motor, prim, prim.retarget(nav.aim_cfg(cur), pair), rel)
        st = hop.step(view, rel)
        t_hop += dt
        if st == "unreachable":
            if fail(cur):
                return {"result": "stalled", "time": (k + 1) * dt}
            cur = None
        elif st == "done":
            hop = None                                       # next hop from wherever this one ended
            if not nav.goal[cur] and nav.reached(z, m, nav.D[cur] - 1e-6, pair=pair, carrying=carrying) is None:
                if fail(cur):                                # the hop did not get anywhere new
                    return {"result": "stalled", "time": (k + 1) * dt}
                cur = None
        if on_step is not None:
            on_step(k)
        if stop is not None and stop():
            return {"result": "interrupted", "time": (k + 1) * dt}
        if watch and k % every == every - 1:
            g = relation_graph(env, R)
            if any(w not in g for w in watch):
                return {"result": "broken", "time": (k + 1) * dt}
    return {"result": "timeout", "time": timeout}
