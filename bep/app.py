"""Interactive app for one scene: learn (or reuse the learning), then pick and run high-level tasks.

    python -m bep.app --scene cubes6 [--relearn] [--motor robot|hog]

Window 1: the MuJoCo viewer.  Window 2: a small panel with the task (e.g. tower) and its parameter
(e.g. the order, bottom -> top), and buttons Execute, Stop, Reset scene.  Pressing Execute while a
task runs replaces the task on the spot (a perturbation): the current primitive is interrupted and
the brain replans from the current state with the new goal.  Spike-MPPI executes each call.
"""
from __future__ import annotations

import argparse
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

from .brain.goal import success_test
from .brain.model import name
from .brain.solve import covered_calls, execute_brain_call, make_brain, models_for, rest
from .brain.tasks import BUILDERS
from .config import EXEC
from .explore.library import learning_dirs, load_library, load_play
from .world.graph import as_tuples, object_poses, relation_graph
from .world.scenes import SCENES, build_scene, start_layout


class App:
    def __init__(self, scene: str, learning: list[Path], threads: int, motor: str = "robot"):
        from .motor.mppi import MPPIConfig, SpikeMPPI
        from .world.objects import use_body
        use_body("float" if motor == "hog" else "robot")
        self.spec = SCENES[scene]
        self.env = build_scene(self.spec, f"app{os.getpid()}")
        self.R, self.lib = load_library(learning)
        self.play = load_play(learning, self.R, robot=motor == "robot")
        self.ctl = None if motor == "hog" else SpikeMPPI(self.env, MPPIConfig(threads=threads, seed=0))
        self.rng = np.random.default_rng()
        self.state, self.brain, self.goal, self.support, self.interrupt = "idle", None, None, None, False
        self.reset()
        self._panel()

    # ------------------------------------------------------------------ panel
    def _panel(self):
        self.root = tk.Tk()
        self.root.title(f"{self.spec.name}: discovered primitives")
        frm = ttk.Frame(self.root, padding=10)
        frm.grid()
        ttk.Label(frm, text="Task").grid(column=0, row=0, sticky="w")
        self.task = tk.StringVar(value=next(iter(self.spec.tasks)))
        box = ttk.Combobox(frm, textvariable=self.task, values=list(self.spec.tasks), state="readonly", width=18)
        box.grid(column=1, row=0, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda e: self._defaults())
        ttk.Label(frm, text="Parameter").grid(column=0, row=1, sticky="w")
        self.param = tk.StringVar()
        ttk.Entry(frm, textvariable=self.param, width=22).grid(column=1, row=1, sticky="w")
        self.hint = ttk.Label(frm, text="", foreground="#666")
        self.hint.grid(column=1, row=2, sticky="w")
        bar = ttk.Frame(frm)
        bar.grid(column=0, row=3, columnspan=2, pady=6)
        ttk.Button(bar, text="Execute", command=self.execute).grid(column=0, row=0, padx=3)
        ttk.Button(bar, text="Stop", command=self.stop).grid(column=1, row=0, padx=3)
        ttk.Button(bar, text="Reset scene", command=self.reset_clicked).grid(column=2, row=0, padx=3)
        self.status = tk.StringVar(value="ready")
        ttk.Label(frm, textvariable=self.status, wraplength=380, justify="left").grid(column=0, row=4, columnspan=2,
                                                                                       sticky="w")
        self._defaults()

    def _defaults(self):
        default, help_ = self.spec.tasks[self.task.get()]
        self.param.set(default)
        self.hint.config(text=help_)

    # ------------------------------------------------------------------ actions
    def reset(self):
        start_layout(self.env, self.spec, self.rng)
        if self.ctl is not None:
            self.ctl.reset(q_cmd=self.env.arm_q(), g_cmd=self.env.gripper_width())
        self.state, self.brain = "idle", None

    def reset_clicked(self):
        self.interrupt = True
        self.reset()
        self.status.set("scene reset")

    def stop(self):
        self.interrupt = True
        self.state = "idle"
        self.status.set("stopped")

    def execute(self):
        self.goal, self.support = BUILDERS[self.task.get()](self.env, self.R, self.param.get())
        if self.brain is None:                        # one brain per scene layout; a new task only changes its goal
            models = models_for(self.R, self.lib, self.play, self.env)
            self.brain = make_brain(self.env, self.lib, models, self.goal, int(self.rng.integers(1 << 30)))
        else:
            self.brain.set_goal(self.goal)
        self.interrupt = self.state == "run"          # a task change mid-run interrupts the current primitive
        self.state = "run"
        self.status.set(f"{self.task.get()} {self.param.get()}: goal {sorted(self.goal.require)}")

    # ------------------------------------------------------------------ loop
    def _ui(self):
        self.viewer.sync()
        self.root.update()

    def step(self):
        env, R = self.env, self.R
        graph = relation_graph(env, R)
        rels = as_tuples(graph)
        if self.goal.unmet(rels) == 0:
            if not success_test(env, self.support):
                rest(env, self.ctl)                     # the relations are right: let go and settle
            if success_test(env, self.support):
                self.status.set("done: goal reached")
                self.state = "idle"
                return
        poses = object_poses(env)
        key, spot, info = self.brain.decide(rels, poses, covered_calls(env, self.lib, self.brain.model, poses))
        self.status.set(f"running {name(key)}   ({' | '.join(info['plan'][:4])})")
        self.root.update()
        self.interrupt = False
        ui = type("UI", (), {"frame": lambda _s: self._ui()})()
        out = execute_brain_call(env, R, self.lib, self.ctl, key, spot, timeout=EXEC["timeout_s"], rec=ui,
                           stop=lambda: self.interrupt)
        if out["result"] == "interrupted":
            return
        rels2 = as_tuples(relation_graph(env, R))
        d = self.brain.observe(rels, key, out["time"], rels2, poses, out["result"])
        self.status.set(f"{name(key)}: {out['result']} in {out['time']:.1f} s   (unmet goal literals "
                        f"{self.goal.unmet(rels2)}, success {d['ok']})")

    def run(self):
        with mujoco.viewer.launch_passive(self.env.model, self.env.data, show_left_ui=False,
                                         show_right_ui=False) as self.viewer:
            while self.viewer.is_running():
                try:
                    if self.state == "run" and self.brain is not None:
                        self.step()
                    else:
                        self._ui()
                        time.sleep(0.02)
                except tk.TclError:              # panel closed
                    break


def main(scene: str | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if scene is None:
        ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--relearn", action="store_true",
                    help="learn from scratch (default: reuse the learning in --learning, learn only if missing)")
    ap.add_argument("--learning", default=None,
                    help="learning directories, comma-separated (default: runs/<minimal scene> of each minimal scene)")
    ap.add_argument("--probes", type=int, default=30000, help="relearning: number of probes")
    ap.add_argument("--max-rounds", type=int, default=300,
                    help="relearning: cap on exploration rounds per primitive (each stops by itself earlier)")
    ap.add_argument("--cores", type=int, default=None, help="relearning: worker processes (default all cores)")
    ap.add_argument("--no-clips", action="store_true", help="relearning: skip rendering the README clips")
    ap.add_argument("--learn-only", action="store_true", help="learn (if needed) and exit without the interface")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="Spike-MPPI rollout threads")
    ap.add_argument("--motor", choices=["robot", "hog"], default="robot",
                    help="robot: Franka arm + Spike-MPPI; hog: the floating hand and the Hand-of-God motor")
    args = ap.parse_args()
    scene = scene or args.scene
    learning = learning_dirs(SCENES[scene], args.learning)
    for d in learning:                     # each (minimal-scene) learning, learned if missing
        if args.relearn or not (d / "play.jsonl").exists():
            from .learn import learn
            learn(d.name if d.name in SCENES else scene, d, probes=args.probes, max_rounds=args.max_rounds,
                  cores=args.cores, clips=not args.no_clips)
    if not args.learn_only:
        App(scene, learning, args.threads, args.motor).run()


if __name__ == "__main__":
    main()
