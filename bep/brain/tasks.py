"""Tasks as physical goal examples: the bodies arranged as they should end up.

A builder arranges the example on a copy of the current state (Hand of God), reads the goal through
the discovered relations (bep.brain.goal.goal_from_example) and the yardstick (which body rests on
which, ObjWorld.facts), then restores the state.  The tasks never occur during learning.

    tower    order bottom -> top, e.g. "DCBA" (D on the table, C on D, ...)
    restack  "start>goal", two towers bottom -> top, e.g. "DCBA>ABCD": the bodies start stacked as the first
             (built for real where the bottom body lies) and must end as the second
"""
from __future__ import annotations

import json
from pathlib import Path

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
    try:
        build()
        goal = goal_from_example(env, R, bodies)
        facts = env.facts()
        support = {n: facts["support"][n] for n in bodies if n in env.free}
    finally:
        env.set_state(st, ctrl)
        mujoco.mj_forward(env.model, env.data)
    return goal, support


def task_arrange(env, R, param: str):
    """One or more groups, each written bottom to top, separated by slashes."""
    groups = [s.strip() for s in param.split("/")]
    bodies = list("".join(groups))
    if any(not g for g in groups) or len(set(bodies)) != len(bodies) or not set(bodies) <= set(env.free):
        raise ValueError("an arrangement needs nonempty groups of distinct scene body names")

    def build():
        for group in groups:
            _put_table(env, group[0], _anchor(env, group))
            for lo, hi in zip(group, group[1:]):
                _put_on(env, hi, [lo])

    return _with_example(env, R, build, bodies)


def task_tower(env, R, order: str):
    """One group named bottom to top."""
    return task_arrange(env, R, order)


def task_example(env, R, path: str):
    """Read a physical pose example. JSON maps each body to [x, y, z, yaw] in metres and radians."""
    poses = json.loads(Path(path).read_text())
    if not isinstance(poses, dict) or not poses or not set(poses) <= set(env.free):
        raise ValueError("an example needs a nonempty mapping from free body names to poses")
    place = {}
    for n, pose in poses.items():
        pose = np.asarray(pose, float)
        if pose.shape != (4,) or not np.isfinite(pose).all():
            raise ValueError(f"invalid pose for body {n}; expected four finite numbers")
        place[n] = (pose[:3], float(pose[3]))

    def build():
        if realize_world(env, place) is None:
            raise ValueError("the example could not be realized without penetration")

    return _with_example(env, R, build, tuple(poses))


def task_restack(env, R, param: str):
    """param: 'start>goal', both towers bottom -> top.  Builds the start tower in the scene, then reads the goal."""
    start, end = ([c for c in part if c in env.free] for part in param.split(">"))
    s = env.sensors()
    _put_table(env, start[0], s[env.sidx[f"{start[0]}_pos"]][:2])
    for lo, hi in zip(start, start[1:]):
        _put_on(env, hi, [lo])
    return task_tower(env, R, "".join(end))


BUILDERS = {"tower": task_tower, "restack": task_restack, "arrange": task_arrange, "example": task_example}
