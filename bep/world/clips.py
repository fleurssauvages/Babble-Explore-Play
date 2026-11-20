"""Short clips of each learning stage (for the README).

    Recorder(env, path).frame()          one rendered frame (call it while simulating)
    Recorder(...).hold(seconds)          repeat the last frame
Clips are written with imageio at 25 fps; rendering is offscreen (MUJOCO_GL=egl).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


class Recorder:
    def __init__(self, env, path: str | Path, *, lookat=(0.55, 0.0, 0.08), distance=1.25, azimuth=150.0,
                 elevation=-28.0, size=(360, 480), every: int = 2, max_s: float = 12.0):
        import mujoco
        self.env, self.path, self.every, self.k = env, Path(path), every, 0
        self.renderer = mujoco.Renderer(env.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.frames: list = []
        self.max_frames = int(max_s * 25)

    def frame(self, force: bool = False):
        self.k += 1
        if (force or self.k % self.every == 0) and len(self.frames) < self.max_frames:
            self.renderer.update_scene(self.env.data, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def hold(self, seconds: float):
        self.frame(force=True)
        self.frames += [self.frames[-1]] * int(seconds * 25)

    def save(self):
        import imageio
        if self.frames:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self.path, np.array(self.frames), fps=25, macro_block_size=1)
        self.renderer.close()
