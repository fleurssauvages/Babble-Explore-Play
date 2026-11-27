"""Recordings of task episodes, and their replay or export.

    python -m bep.brain.solve ... --save runs/last_run.npz         # (default) the simulation state at every control step
    python -m bep.replay                                           # replay runs/last_run.npz in the MuJoCo viewer
    python -m bep.replay --file runs/x.npz --speed 0.5 --loop      # slower, looping
    python -m bep.replay --mp4 runs/x.mp4 [--width 960 --height 540 --fps 50 --start 2 --end 10]
    python -m bep.replay --gif runs/x.gif

A recording holds, per control step, the time, qpos, qvel, act, ctrl and the mocap targets (the floating hand), the
model-level body poses (fixtures placed by the layout), and what rebuilds the scene (scene name, body).  Replaying
sets each recorded state and renders it: no simulation, so it shows exactly what happened.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
CAMERA = dict(lookat=(0.55, 0.0, 0.08), distance=1.25, azimuth=150.0, elevation=-28.0)   # as bep.world.clips


class StateLog:
    """Records the simulation state at every ``frame()`` (called each control step during execution), with the
    interface of bep.world.clips.Recorder (frame, hold, save)."""

    def __init__(self, env, path: str | Path, meta: dict | None = None):
        self.env, self.path, self.meta = env, Path(path), dict(meta or {})
        m = env.model
        self.body_pos, self.body_quat = m.body_pos.copy(), m.body_quat.copy()     # fixtures placed at model level
        self.rows: dict[str, list] = {k: [] for k in ("time", "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")}

    def frame(self, force: bool = False) -> None:
        d = self.env.data
        if self.rows["time"] and not force and d.time <= self.rows["time"][-1]:
            return
        for k, v in (("time", float(d.time)), ("qpos", d.qpos.copy()), ("qvel", d.qvel.copy()), ("act", d.act.copy()),
                     ("ctrl", d.ctrl.copy()), ("mocap_pos", d.mocap_pos.copy()), ("mocap_quat", d.mocap_quat.copy())):
            self.rows[k].append(v)

    def hold(self, seconds: float) -> None:
        self.frame(force=True)                         # (replays pause at the end on their own)

    def save(self) -> Path:
        from .config import CONTROL_DT
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v) for k, v in self.meta.items()}
        tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, replay_format_version=np.asarray(FORMAT_VERSION), control_dt=np.asarray(CONTROL_DT),
                            state_time=np.asarray(self.rows["time"], float),
                            **{k: np.asarray(self.rows[k]) for k in self.rows if k != "time"},
                            body_pos=self.body_pos, body_quat=self.body_quat,
                            **{f"meta_{k}": np.asarray(v) for k, v in meta.items() if v is not None})
        tmp.replace(self.path)
        return self.path


class Tee:
    """Several recorders driven as one (a clip and a state log)."""

    def __init__(self, *recs):
        self.recs = [r for r in recs if r is not None]

    def frame(self, force: bool = False) -> None:
        for r in self.recs:
            r.frame(force)

    def hold(self, seconds: float) -> None:
        for r in self.recs:
            r.hold(seconds)

    def save(self) -> None:
        for r in self.recs:
            r.save()


# ------------------------------------------------------------------ loading
def load_recording(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        missing = sorted({"state_time", "qpos", "qvel", "meta_scene", "meta_body"} - set(z.files))
        if missing:
            raise ValueError(f"{path}: not a recording (missing {', '.join(missing)})")
        out = {k: np.asarray(z[k]).copy() for k in z.files}
    for k in [k for k in out if k.startswith("meta_")]:
        out[k] = out[k].item() if out[k].shape == () else out[k]
    return out


def _world(rec: dict):
    """The recorded scene, rebuilt (same model layout), with the recorded model-level body poses."""
    from .world.objects import body_kind, use_body
    from .world.scenes import SCENES, build_scene
    body, kind = str(rec["meta_body"]), body_kind()
    use_body(body)
    try:
        env = build_scene(SCENES[str(rec["meta_scene"])], f"replay{os.getpid()}", body=body)
    finally:
        use_body(kind)
    m = env.model
    if m.nq != rec["qpos"].shape[1] or m.nv != rec["qvel"].shape[1]:
        raise ValueError("the scene rebuilt differs from the recorded one (the scene definition changed since)")
    if rec.get("body_pos") is not None and rec["body_pos"].shape == m.body_pos.shape:
        m.body_pos[:], m.body_quat[:] = rec["body_pos"], rec["body_quat"]
    return env


def _set_state(env, rec: dict, i: int) -> None:
    import mujoco
    d = env.data
    d.time = float(rec["state_time"][i])
    d.qpos[:], d.qvel[:] = rec["qpos"][i], rec["qvel"][i]
    if env.model.na and len(rec.get("act", [])):
        d.act[:] = rec["act"][i]
    if env.model.nu and len(rec.get("ctrl", [])):
        d.ctrl[:] = rec["ctrl"][i]
    if env.model.nmocap and len(rec.get("mocap_pos", [])):
        d.mocap_pos[:], d.mocap_quat[:] = rec["mocap_pos"][i], rec["mocap_quat"][i]
    mujoco.mj_forward(env.model, d)


def _frames(t: np.ndarray, start_s: float, end_s: float | None) -> tuple[np.ndarray, int, int]:
    rel = t - t[0]
    first = int(np.searchsorted(rel, max(0.0, float(start_s)), side="left"))
    last = len(rel) - 1 if end_s is None else min(len(rel) - 1, int(np.searchsorted(rel, float(end_s), side="right")) - 1)
    if first >= len(rel) or last < first:
        raise ValueError("the requested interval holds no recorded frame")
    return rel, first, last


def _camera(cam) -> None:
    cam.lookat[:] = CAMERA["lookat"]
    cam.distance, cam.azimuth, cam.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]


def _describe(path, rec: dict, n: int) -> str:
    keys = ("task", "param", "motor", "episode", "success")
    info = ", ".join(f"{k}={rec[f'meta_{k}']}" for k in keys if f"meta_{k}" in rec)
    return f"{Path(path).name}: {rec['meta_scene']} ({info}), {n} frames, {rec['state_time'][-1] - rec['state_time'][0]:.1f} s"


# ------------------------------------------------------------------ replay and export
def _alive(viewer) -> bool:
    try:
        return bool(viewer.is_running())
    except Exception:                                  # the window is gone (its thread has ended)
        return False


def _sync(viewer) -> bool:
    """Show the current state; False once the window has been closed (a sync can race the close)."""
    try:
        viewer.sync()
        return _alive(viewer)
    except Exception:
        return False


def _wait(viewer, seconds: float) -> bool:
    """Sleep while the window stays open; False as soon as it is closed."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if not _alive(viewer):
            return False
        time.sleep(min(0.05, max(0.0, end - time.perf_counter())))
    return _alive(viewer)


def replay(path, *, speed: float = 1.0, loop: bool = False, show_ui: bool = False, start_s: float = 0.0,
           end_s: float | None = None, pause_at_end: float = 1.0) -> None:
    """The recording in MuJoCo's passive viewer, in (scaled) real time.  Without ``loop`` the last frame stays on
    screen until the window is closed; with it, the replay restarts after ``pause_at_end`` seconds."""
    import mujoco.viewer
    rec = load_recording(path)
    env = _world(rec)
    rel, first, last = _frames(rec["state_time"], start_s, end_s)
    print("replay", _describe(path, rec, last - first + 1), f"at {speed:g}x", "(close the window to quit)",
          flush=True)
    _set_state(env, rec, first)
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=show_ui, show_right_ui=show_ui)
    _camera(viewer.cam)
    try:
        open_ = True
        while open_:
            wall0, sim0 = time.perf_counter(), float(rel[first])
            for i in range(first, last + 1):
                wait = wall0 + (float(rel[i]) - sim0) / speed - time.perf_counter()
                if wait > 0 and not _wait(viewer, wait):
                    open_ = False
                    break
                _set_state(env, rec, i)
                if not _sync(viewer):
                    open_ = False
                    break
            if not open_:
                break
            if not loop:                               # the last frame stays until the window is closed
                while _wait(viewer, 0.2):
                    pass
                break
            open_ = _wait(viewer, pause_at_end)
    except KeyboardInterrupt:
        pass
    finally:
        if _alive(viewer):                             # only a window still open is closed from here
            try:
                viewer.close()
            except Exception:
                pass
        time.sleep(0.2)                                # the viewer's thread finishes before the interpreter exits


def export_video(path, out, *, speed: float = 1.0, fps: float | None = None, width: int = 960, height: int = 540,
                 start_s: float = 0.0, end_s: float | None = None) -> Path:
    """The recording rendered offscreen to an MP4 (H.264) or a GIF, by ``out``'s suffix."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    out = Path(out).expanduser()
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in ("mp4", "gif"):
        raise ValueError("export to a .mp4 or a .gif")
    if fmt == "mp4" and (width % 2 or height % 2):
        raise ValueError("MP4 width and height must be even")
    rec = load_recording(path)
    env = _world(rec)
    rel, first, last = _frames(rec["state_time"], start_s, end_s)
    dt = float(rec.get("control_dt", np.nan))
    if not np.isfinite(dt) or dt <= 0:
        dt = float(np.median(np.diff(rel[first:last + 1]))) if last > first else 0.01
    fps = (1.0 / dt) if fps is None else float(fps)
    times = np.arange(rel[first], rel[last] + 0.5 * speed / fps, speed / fps)
    idx = np.clip(np.searchsorted(rel, times, side="left"), first, last)
    m = env.model
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, width), max(m.vis.global_.offheight, height)
    renderer = mujoco.Renderer(m, height=height, width=width, max_geom=max(10000, m.ngeom + 512))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    _camera(cam)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = (imageio.get_writer(out, format="FFMPEG", mode="I", fps=fps, codec="libx264", pixelformat="yuv420p",
                                 macro_block_size=1, output_params=["-crf", "17", "-preset", "medium"])
              if fmt == "mp4" else imageio.get_writer(out, format="GIF-PIL", mode="I", duration=1.0 / fps, loop=0))
    try:
        with writer:
            for i in idx:
                _set_state(env, rec, int(i))
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        renderer.close()
    print(f"wrote {out}: {len(idx)} frames at {fps:g} fps, {width}x{height} ({_describe(path, rec, last - first + 1)})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="runs/last_run.npz", help="a recording (bep.brain.solve --save)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--viewer-ui", action="store_true", help="show the viewer's side panels")
    ap.add_argument("--start", type=float, default=0.0, help="seconds from the recording's start")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--pause-at-end", type=float, default=1.0, help="with --loop: seconds before restarting")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--mp4", default=None, help="export to this MP4 instead of replaying")
    out.add_argument("--gif", default=None, help="export to this GIF instead of replaying")
    ap.add_argument("--fps", type=float, default=None, help="export frame rate (default: one frame per control step)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()
    if args.speed <= 0:
        ap.error("--speed must be positive")
    target = args.mp4 or args.gif
    if target:
        export_video(args.file, target, speed=args.speed, fps=args.fps, width=args.width, height=args.height,
                     start_s=args.start, end_s=args.end)
    else:
        replay(args.file, speed=args.speed, loop=args.loop, show_ui=args.viewer_ui, start_s=args.start,
               end_s=args.end, pause_at_end=args.pause_at_end)


if __name__ == "__main__":
    from .terminal import run_main
    run_main(main)
