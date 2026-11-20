"""Closed-loop execution of one primitive call by Spike-MPPI.

``run_primitive`` runs until the primitive's goal holds (for ``EXEC['hold_s']``), a timeout, the
field value stalls, a validated hop keeps failing, or a watched relation breaks.  Near the
primitive's validated graph it executes by hops along it (``_Navigator``: the "return" of
Go-Explore), far from it it descends the cost-to-go field.
"""
from __future__ import annotations

import numpy as np

from ..config import EXEC
from ..world.graph import relation_graph


def execute_call(env, R, prim, field, ctl, x: str, target, *, timeout: float = EXEC["timeout_s"], on_step=None,
                 stop=None, learning: bool = False) -> dict:
    """One primitive call in a world with N bodies, the same in play and in tasks: the primitive's cost on
    these roles (X, target), keeping every relation that holds and does not involve X as subject, and
    stopping as "broken" as soon as one of them breaks (e.g. grasping a body that supports another).  On the
    floating hand in learning (play), the Hand-of-God motor's straight hops along the validated graph (fast);
    in tasks, Spike-MPPI on either body (the floating hand's planner, or ``ctl`` for the arm) with the same
    cost and loop (bep.motor.hog_motor.run_primitive_hog)."""
    if env.floating and learning:
        from .hog_motor import run_primitive_hop
        return run_primitive_hop(env, R, prim, field, x, target, timeout=timeout, on_step=on_step, stop=stop)
    from .hog_motor import run_primitive_hog       # the same executor for both bodies: Phi + barrier + goal reward
    return run_primitive_hog(env, R, prim, field, x, target, timeout=timeout, on_step=on_step, stop=stop,
                             ctl=None if env.floating else ctl)


def run_primitive(env, ctl, cost, *, timeout: float = EXEC["timeout_s"], watch: list | None = None, R=None,
                  on_step=None, stop=None) -> dict:
    """Run one call; returns {"result": done | stalled | timeout | broken | interrupted, "time": seconds}."""
    dt = ctl.cfg.control_dt
    steps = int(round(timeout / dt))
    stall = int(round(EXEC["stall_s"] / dt))
    return _run(env, ctl, cost, cost, dt, steps, stall, watch, R, on_step, stop)


class _Navigator:
    """Execution along the roadmap: far from the graph, descend the field toward it; near it, enter at a
    nearby low-cost node, track its successor, and move on whenever a node of lower cost-to-go is reached
    (as in the validation of every edge)."""

    def __init__(self, cost, roll, nav):
        self.cost, self.roll, self.nav, self.prim = cost, roll, nav, cost.prim
        self.failed: set = set()
        self.cur = self.wp = None
        self.t, self.joining = 0.0, False

    def _rel(self, s):
        return self.cost.rel_now(s)

    def _aim(self, k: int, join: bool = False) -> None:
        """Track node k's successor; with ``join``, first node k itself (entering the graph)."""
        nav = self.nav
        j = k if join or nav.goal[k] or nav.succ[k] < 0 else int(nav.succ[k])
        mt = self.prim.goal_mode if nav.goal[j] else int(nav.M[j])
        self.cur, self.wp, self.t, self.joining = k, j, 0.0, join
        for c in (self.cost, self.roll):
            c.set_waypoint(nav.Z[j], mt)

    def start(self, s) -> None:
        self.update(s, 0.0)

    def update(self, s, dt) -> bool:
        """Advance along the graph; False once hops keep failing (the call stalls)."""
        nav = self.nav
        rel = self._rel(s)
        z, m = rel["z"], int(rel["mode"])
        if nav.near(z, m) > EXEC["hop_join"] and m != self.prim.goal_mode:     # join the graph first
            if self.cur is not None:
                self.cur = self.wp = None
                for c in (self.cost, self.roll):
                    c.clear_waypoint()
            return True
        if self.cur is None:
            self._aim(nav.entry(z, m, exclude=self.failed), join=True)
            return True
        # progress: a node of lower cost-to-go (or, while joining, the entry node itself) within eps
        r = nav.reached(z, m, nav.D[self.cur] + (1e-6 if self.joining else -1e-6))
        if r is not None:
            self._aim(r)
            return True
        self.t += dt
        if self.t > EXEC["hop_stall"] * (nav.edge_s + 0.5):      # this hop failed: re-enter elsewhere
            self.failed.update({self.wp, self.cur})      # neither the target nor this entry again
            if len(self.failed) > 4:
                return False
            self._aim(nav.entry(z, m, exclude=self.failed), join=True)
        return True


def _run(env, ctl, cost, roll, dt, steps, stall, watch, R, on_step, stop):
    hold, best, since = 0, np.inf, 0
    nav = getattr(getattr(cost, "field", None), "roadmap", None)
    nav = _Navigator(cost, roll, nav) if EXEC["roadmap"] and nav is not None and len(nav.D) \
        and hasattr(cost, "set_waypoint") else None
    if nav is not None:
        nav.start(env.sensors())
    for k in range(steps):
        a, _ = ctl.step(env.data, roll)
        ctl.apply(env.data, a)
        if on_step is not None:
            on_step(k)
        if stop is not None and stop():
            return {"result": "interrupted", "time": (k + 1) * dt}
        s = env.sensors()
        done = cost.done(s)
        hold = hold + 1 if done else 0
        if done and hold * dt >= EXEC["hold_s"]:
            return {"result": "done", "time": (k + 1) * dt}
        if nav is not None and not done and not nav.update(s, dt):
            return {"result": "stalled", "time": (k + 1) * dt}
        if nav is None or nav.cur is None:
            # stop a call whose field value no longer improves (it would only time out)
            phi = cost.progress(s)
            if done or phi < best * (1.0 - EXEC["stall_rel"]):
                best, since = min(best, phi), 0
            else:
                since += 1
                if since >= stall:
                    return {"result": "stalled", "time": (k + 1) * dt}
        else:
            best, since = np.inf, 0                  # on the graph: hop timeouts decide (_Navigator)
        every = max(1, int(round(EXEC["watch_s"] / dt)))
        if watch and R is not None and k % every == every - 1:
            g = relation_graph(env, R, s)
            broken = [w for w in watch if w not in g]
            if broken:
                return {"result": "broken", "time": (k + 1) * dt, "broken": [str(b) for b in broken]}
    return {"result": "timeout", "time": steps * dt}
