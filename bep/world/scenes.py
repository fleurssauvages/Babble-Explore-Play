"""The scenes, as data: bodies, where they start, and the tasks that can be asked in them.

Nothing here is used by learning except the list of bodies.  Tasks are physical goal *examples* (the
bodies arranged as they should end up), built by bep.brain.tasks and read through the discovered
relations at run time.

    pair_cubes  two cubes              (minimal scene: everything is learned here)
    cubes4      four cubes             tower (any order)       learned from pair_cubes
    cubes6      six cubes              tower (any order)       learned from pair_cubes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import TABLE_ANGLE, TABLE_RHO
from .objects import FIXTURES, LIBRARY, ObjWorld, apply_layout, table_layout

FIXTURE_NAMES = "PQSTUV"


@dataclass
class SceneSpec:
    name: str
    objects: str                              # free bodies
    shapes: list                              # their shapes (cycled)
    fixtures: list = field(default_factory=list)          # fixture kinds, named P, Q, ...
    fixture_poses: list = field(default_factory=list)     # (x, y, yaw[, z]) of each fixture's base
    learned_from: list = field(default_factory=list)      # minimal scenes whose learning the scene reuses
    start_rho: tuple = TABLE_RHO              # where free bodies start (radial band on the table)
    tools: dict = field(default_factory=dict)  # name -> shape of free bodies placed within reach
    tasks: dict = field(default_factory=dict)  # task name -> (default parameter, help); builders: bep.brain.tasks


MINIMAL = {"pair_cubes": SceneSpec("pair_cubes", "AB", ["cube"])}

SCENES = {
    **MINIMAL,
    "cubes4": SceneSpec("cubes4", "ABCD", ["cube"], learned_from=["pair_cubes"],
                        tasks={"tower": ("DCBA", "bottom -> top"),
                               "restack": ("DCBA>ABCD", "start tower > goal tower, bottom -> top")}),
    "cubes6": SceneSpec("cubes6", "ABCDEF", ["cube"], learned_from=["pair_cubes"],
                        tasks={"tower": ("FEDCBA", "bottom -> top")}),
}


# ------------------------------------------------------------------ worlds
def build_scene(spec: SceneSpec, tag: str = "app", body: str | None = None) -> ObjWorld:
    shapes = {n: spec.tools.get(n, spec.shapes[i % len(spec.shapes)]) for i, n in enumerate(spec.objects)}
    fix = {FIXTURE_NAMES[i]: k for i, k in enumerate(spec.fixtures)}
    assert not set(shapes) & set(fix), "body names clash"
    env = ObjWorld({n: LIBRARY[k] for n, k in shapes.items()}, {n: FIXTURES[k] for n, k in fix.items()}, tag=tag,
                   body=body)
    env.kinds = {**shapes, **fix}
    env.fixture_poses = {FIXTURE_NAMES[i]: tuple(p) for i, p in enumerate(spec.fixture_poses)}
    return env


def scene_pairs(spec: SceneSpec) -> list[tuple[str, str]]:
    """The (X, Y) kind pairs that exist in a scene: X a free body or a fixture's moving part, Y another
    body (what the probe samples)."""
    kinds = {n: spec.tools.get(n, spec.shapes[i % len(spec.shapes)]) for i, n in enumerate(spec.objects)}
    kinds.update({FIXTURE_NAMES[i]: k for i, k in enumerate(spec.fixtures)})
    if len(kinds) == 1:                              # a single body: probed against the virtual reference
        return [(next(iter(kinds.values())), "spot")]
    return sorted({(kinds[x], kinds[y]) for x in kinds for y in kinds if x != y})


def scene_kinds(spec: SceneSpec) -> tuple[list, list]:
    """Kinds of free bodies and of fixtures in a scene (what the probe needs)."""
    free = sorted({spec.tools.get(n, spec.shapes[i % len(spec.shapes)]) for i, n in enumerate(spec.objects)})
    return free, sorted(set(spec.fixtures))


# ------------------------------------------------------------------ layouts
def _spot_fn(env, fix_poses):
    def fn(rng):
        for _ in range(1000):
            r, a = rng.uniform(*TABLE_RHO), rng.uniform(-TABLE_ANGLE, TABLE_ANGLE)
            p = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.linalg.norm(p - np.array(fp[:2])) > float(np.hypot(*env.fixtures[n].part.half[:2])) + 0.12
                   for n, fp in fix_poses.items()):
                return float(p[0]), float(p[1])
        raise RuntimeError("no free table spot")
    return fn


def table_spot_fn(env):
    """Table spots clear of the scene's fixtures (at their current poses)."""
    poses = {n: env.model.body_pos[env.fix_bid[n]][:2] for n in env.fixtures}
    return _spot_fn(env, poses)


def random_layout(env, rng) -> tuple[dict, dict]:
    """Fixtures at the scene's poses (or anywhere in their regions), joints at random, free bodies apart."""
    layout, joints = {}, {}
    poses = getattr(env, "fixture_poses", {})
    for n, fx in env.fixtures.items():
        reg = fx.region
        layout[n] = poses.get(n) or (rng.uniform(*reg[0]), rng.uniform(*reg[1]), rng.uniform(*reg[2]))
        joints[n] = float(rng.integers(2)) if rng.random() < 0.5 else float(rng.random())
    layout.update(table_layout(rng, env.free, env.shapes, _spot_fn(env, layout)))
    return layout, joints


def _spots(rng, n, rho, avoid, gap=0.07, angle=TABLE_ANGLE):
    out = []
    for _ in range(4000):
        if len(out) == n:
            break
        r, a = rng.uniform(*rho), rng.uniform(-angle, angle)
        p = np.array([r * np.cos(a), r * np.sin(a)])
        if abs(p[1]) > 0.55 or p[0] > 0.97:
            continue
        if all(np.linalg.norm(p - q) > gap for q in out) and all(np.linalg.norm(p - q) > 0.2 for q in avoid):
            out.append(p)
    if len(out) < n:
        raise RuntimeError("could not lay the scene out")
    return out


def start_layout(env, spec: SceneSpec, rng):
    """Fixtures at their poses (joints closed), the free bodies apart in the scene's start band."""
    layout = {FIXTURE_NAMES[i]: p for i, p in enumerate(spec.fixture_poses)}
    avoid = [np.array(p[:2]) for p in spec.fixture_poses]
    names = [n for n in env.free if n not in spec.tools]
    for n, p in zip(names, _spots(rng, len(names), spec.start_rho, avoid)):
        layout[n] = (float(p[0]), float(p[1]), float(rng.uniform(-np.pi / 4, np.pi / 4)))
    for n in spec.tools:
        p = _spots(rng, 1, (0.38, 0.5), avoid + [np.array(layout[m][:2]) for m in names], gap=0.15, angle=0.5)[0]
        layout[n] = (float(p[0]), float(p[1]), float(rng.uniform(-0.4, 0.4)))
    apply_layout(env, layout, joints={n: 0.0 for n in env.fixtures})
    return layout


def body_frame(env, n):
    """World position and yaw of a body's frame (after kinematics)."""
    bid = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"cube{n}"))
    R = env.data.xmat[bid].reshape(3, 3)
    return env.data.xpos[bid].copy(), float(np.arctan2(R[1, 0], R[0, 0]))
