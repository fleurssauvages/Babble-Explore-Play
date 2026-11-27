"""Small geometric helpers shared by every phase (rotations, sensor layouts, table spots)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def characteristic_length(half_extents):
    """Geometric mean diameter of a body's bounding box."""
    return 2.0 * np.cbrt(np.prod(half_extents, axis=-1))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    if isinstance(q, np.ndarray) and q.shape == (4,) and q.dtype == np.float64:
        # one quaternion (every control step): the same expressions on floats, without array overhead
        w, x, y, z = q.tolist()
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    w, x, y, z = (q[..., i] for i in range(4))
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotate_into_yaw_frame(v: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Express world vectors v (...,3) in a frame rotated by ``yaw`` about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(v)
    out[..., 0] = c * v[..., 0] + s * v[..., 1]
    out[..., 1] = -s * v[..., 0] + c * v[..., 1]
    out[..., 2] = v[..., 2]
    return out


def yaw_q(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def layout_of(order) -> tuple[dict, dict, int]:
    """(slice or index per field, index array per field, total size) of a flat sensor layout."""
    kw, sl, a = {}, {}, 0
    for name, dim in order:
        kw[name] = slice(a, a + dim) if dim > 1 else a
        sl[name] = np.arange(a, a + dim)
        a += dim
    return kw, sl, a


@dataclass
class Spot:
    """A virtual support on the table: placing X 'on' it puts X on the table there."""
    x: float
    y: float
    yaw: float = 0.0

    @property
    def name(self) -> str:
        return f"spot({self.x:.2f},{self.y:.2f})"


# ------------------------------------------------------------------ rotations (6-DoF hand)
def rotvec_mat(v) -> np.ndarray:
    """Rotation matrix of a rotation vector (axis * angle)."""
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_rotvec(R) -> np.ndarray:
    """Rotation vector of a rotation matrix (inverse of rotvec_mat)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:                                   # half turn: R = 2 k k^T - I
        i = int(np.argmax(np.diag(R)))
        k = (R[:, i] + np.eye(3)[i]) / (2 * np.sqrt(max((R[i, i] + 1) / 2, 1e-12)))
        return th * k / np.linalg.norm(k)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return th * w


def random_rotation(rng) -> np.ndarray:
    """A rotation drawn uniformly (Shoemake's method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(u1) * np.cos(2 * np.pi * u3), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u3)])
    return quat_to_mat(q)                                 # (w, x, y, z)


def hand_from_axis(axis, rng=None) -> np.ndarray:
    """A hand rotation whose closing axis (column 1) is ``axis``: the approach (column 2, the fingers'
    direction) is the direction perpendicular to it that points most downward, turned by a random angle about
    the closing axis when ``rng`` is given (the approach is not part of the state)."""
    y = np.asarray(axis, float)
    y = y / max(np.linalg.norm(y), 1e-12)
    down = np.array([0.0, 0.0, -1.0])
    z = down - (down @ y) * y
    if np.linalg.norm(z) < 1e-6:                          # vertical closing axis: approach horizontally
        z = np.array([1.0, 0.0, 0.0]) - y[0] * y
    z /= np.linalg.norm(z)
    R = np.stack([np.cross(y, z), y, z], axis=1)
    if rng is not None:
        R = rotvec_mat(y * rng.uniform(-np.pi, np.pi)) @ R
    return R
