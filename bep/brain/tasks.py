"""Tasks as physical goal examples: the bodies arranged as they should end up.

A builder arranges the example on a copy of the current state (Hand of God), reads the goal through
the discovered relations (bep.brain.goal.goal_from_example) and the yardstick (which body rests on
which, ObjWorld.facts), then restores the state.  The tasks never occur during learning.

    tower    order bottom -> top, e.g. "DCBA" (D on the table, C on D, ...)
    restack  "start>goal", two towers bottom -> top, e.g. "DCBA>ABCD": the bodies start stacked as the first
             (built for real where the bottom body lies) and must end as the second
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..world.objects import realize_world
from .goal import goal_from_example


def _put_on(env, x, ys):
    """Place x resting on top of the supports ys (centred over them)."""
    s = env.sensors()
    p = np.mean([s[env.sidx[f"{y}_pos"]] for y in ys], axis=0)
    p[2] = max(s[env.sidx[f"{y}_pos"]][2] + env.shape(y).half[2] for y in ys) + env.shape(x).half[2] + 0.001
    realize_world(env, {x: (p, 0.0)}, None, settle_s=0.2)


def _put_table(env, x, p):
    realize_world(env, {x: (np.array([p[0], p[1], env.shape(x).half[2]]), 0.0)}, None, settle_s=0.05)


def _anchor(env, bodies):
    """A free spot on the table, within reach, for building a structure."""
    s = env.sensors()
    occupied = [s[env.sidx[f"{n}_pos"]][:2] for n in env.free if n not in bodies]
    for r in np.linspace(0.42, 0.6, 7):
        for a in np.linspace(-0.6, 0.6, 9):
            q = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.linalg.norm(q - o) > 0.12 for o in occupied):
                return q
    return np.array([0.5, 0.0])


def _with_example(env, R, build, bodies):
    """Run ``build`` on a copy of the current state, read the goal and the yardstick, restore the state."""
    st, ctrl = env.get_state(), env.data.ctrl.copy()
    build()
    goal = goal_from_example(env, R, bodies)
    support = {n: env.facts()["support"][n] for n in bodies}
    env.set_state(st, ctrl)
    mujoco.mj_forward(env.model, env.data)
    return goal, support


def task_tower(env, R, order: str):
    """order: bottom -> top, e.g. 'CBA'."""
    order = [c for c in order if c in env.free]

    def build():
        _put_table(env, order[0], _anchor(env, order))
        for lo, hi in zip(order, order[1:]):
            _put_on(env, hi, [lo])
    return _with_example(env, R, build, order)


def task_restack(env, R, param: str):
    """param: 'start>goal', both towers bottom -> top.  Builds the start tower in the scene, then reads the goal."""
    start, end = ([c for c in part if c in env.free] for part in param.split(">"))
    s = env.sensors()
    _put_table(env, start[0], s[env.sidx[f"{start[0]}_pos"]][:2])
    for lo, hi in zip(start, start[1:]):
        _put_on(env, hi, [lo])
    return task_tower(env, R, "".join(end))


BUILDERS = {"tower": task_tower, "restack": task_restack}
