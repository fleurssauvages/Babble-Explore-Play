"""The robot's reachability map: which hand poses (pinch point and orientation) the arm can take.

    python -m bep.world.reach            # builds bep/assets/reach_map.npz once (a few minutes)

Learning runs on a floating hand (the Hand of God moves the gripper directly, no arm); the map keeps it
where the real arm could follow.  A pose is reachable when the arm's IK converges within its joint limits
and no part of the arm or hand then touches the table.

The arm is symmetric about its vertical base axis (its first joint), so a pose is described in the frame
turned by its azimuth: radius r and height z of the pinch point, and the hand's rotation relative to that
frame.  The map is a grid over (r, z) times a fixed set of hand rotations (approach directions on a
sphere x turns about the approach); a pose is reachable when its grid cell is, for the nearest rotation of
the set or the same rotation turned by half a turn about the approach (the symmetric gripper makes the
same contacts).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .env import ASSET_DIR, HOME_Q

MAP_PATH = ASSET_DIR / "reach_map.npz"
R_GRID = np.arange(0.15, 0.95 + 1e-9, 0.025)
Z_GRID = np.arange(-0.02, 0.60 + 1e-9, 0.025)
N_DIRS, N_ROLLS = 64, 8


def _rotations() -> np.ndarray:
    """The rotation set: approach directions (the hand's z axis) on a Fibonacci sphere x rolls about it."""
    out = []
    k = np.arange(N_DIRS) + 0.5
    phi = np.arccos(1 - 2 * k / N_DIRS)
    th = np.pi * (1 + 5 ** 0.5) * k
    for p, t in zip(phi, th):
        z = np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)])
        a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = a - (a @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        for j in range(N_ROLLS):
            c, s = np.cos(2 * np.pi * j / N_ROLLS), np.sin(2 * np.pi * j / N_ROLLS)
            out.append(np.stack([c * x + s * y, -s * x + c * y, z], axis=1))
    return np.array(out)


ROTS = _rotations()
# the same rotation turned by half a turn about the approach: roll + N_ROLLS / 2
PARTNER = np.array([(i // N_ROLLS) * N_ROLLS + (i % N_ROLLS + N_ROLLS // 2) % N_ROLLS for i in range(len(ROTS))])


def _quat(R) -> np.ndarray:
    """Unit quaternions (w, x, y, z) of rotation matrices (..., 3, 3)."""
    R = np.asarray(R, float)
    w = np.sqrt(np.maximum(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 1e-12)) / 2
    x = np.copysign(np.sqrt(np.maximum(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0)) / 2, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 0)) / 2, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], 0)) / 2, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


QROTS = _quat(ROTS)


# ------------------------------------------------------------------ building
_B: dict = {}


def _init():
    from .objects import ObjWorld
    env = ObjWorld({}, tag="reach", body="robot")
    import mujoco
    _B.update(env=env, mujoco=mujoco,
              robot=[b for b in env.robot_bids], table=env.table_gid)


def _touches_table(env, q) -> bool:
    mj = _B["mujoco"]
    d = env._ik_data
    d.qpos[env.arm_qadr] = q
    d.qpos[env.finger_qadr] = 0.04
    mj.mj_forward(env.model, d)
    m = env.model
    for b in _B["robot"]:                                 # every link of the arm and hand above the table top
        if d.xpos[b][2] < -0.005 and b != 0:
            return True
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if _B["table"] in (g1, g2):
            other = g2 if g1 == _B["table"] else g1
            if int(m.geom_bodyid[other]) in _B["robot"] and c.dist < 0:
                return True
    return False


def _cell(args):
    """Reachability of every rotation of the set at one (r, z) cell (on the x axis: azimuth 0)."""
    i, j = args
    env = _B["env"]
    p = np.array([R_GRID[i], 0.0, Z_GRID[j]])
    out = np.zeros(len(ROTS), bool)
    q_prev = HOME_Q
    for k, R in enumerate(ROTS):
        q, ok = env.solve_ik(p, R, q_prev)
        if not ok:
            q, ok = env.solve_ik(p, R, HOME_Q)
        if ok and not _touches_table(env, q):
            out[k] = True
            q_prev = q
    return i, j, out


def build(workers: int = 16, path: Path = MAP_PATH) -> None:
    t0 = time.time()
    grid = np.zeros((len(R_GRID), len(Z_GRID), len(ROTS)), bool)
    cells = [(i, j) for i in range(len(R_GRID)) for j in range(len(Z_GRID))]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for n, (i, j, out) in enumerate(pool.imap_unordered(_cell, cells, chunksize=4)):
            grid[i, j] = out
            if n % 100 == 0:
                print(f"{n}/{len(cells)} cells, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(path, grid=grid, r=R_GRID, z=Z_GRID, rots=ROTS)
    print(f"reachability map: {grid.mean():.1%} of poses reachable, {time.time() - t0:.0f} s -> {path}")


# ------------------------------------------------------------------ queries
class ReachMap:
    def __init__(self, path: Path = MAP_PATH):
        f = np.load(path)
        g = f["grid"]
        self.grid = g | g[:, :, PARTNER]                  # the symmetric gripper: either half turn
        self.r, self.z = f["r"], f["z"]

    def reachable(self, pos, R) -> np.ndarray:
        """Is the hand pose (pinch point pos, rotation R) reachable?  Vectorised over leading axes."""
        pos, R = np.asarray(pos, float), np.asarray(R, float)
        az = np.arctan2(pos[..., 1], pos[..., 0])
        r = np.hypot(pos[..., 0], pos[..., 1])
        c, s = np.cos(-az), np.sin(-az)                   # the hand's rotation in the azimuth's frame
        Rz = np.zeros(np.shape(az) + (3, 3))
        Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1], Rz[..., 2, 2] = c, -s, s, c, 1.0
        q = _quat(Rz @ R)
        k = np.argmax(np.abs(q @ QROTS.T), axis=-1)       # nearest rotation of the set
        i = np.clip(np.rint((r - self.r[0]) / (self.r[1] - self.r[0])).astype(int), 0, len(self.r) - 1)
        j = np.clip(np.rint((pos[..., 2] - self.z[0]) / (self.z[1] - self.z[0])).astype(int), 0, len(self.z) - 1)
        inside = (r >= self.r[0] - 0.0125) & (r <= self.r[-1] + 0.0125) & (pos[..., 2] >= self.z[0] - 0.0125) \
            & (pos[..., 2] <= self.z[-1] + 0.0125)
        return inside & self.grid[i, j, k]


_MAP = None


def reach_map() -> ReachMap:
    global _MAP
    if _MAP is None:
        if not MAP_PATH.exists():
            build()
        _MAP = ReachMap()
    return _MAP


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    build(args.workers)


if __name__ == "__main__":
    main()
