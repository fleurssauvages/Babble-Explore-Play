# Babble, Explore, Play: Self-Discovered Manipulation for Zero-Shot Transfer

A Franka Panda learns manipulation primitives from two-body physics probes, then composes them in scenes
with more bodies. Learning runs on a **floating hand** (the gripper without the arm)
moved by the Hand of God, within the arm's reachability map. **Spike-MPPI**
([paper](https://arxiv.org/abs/2609.28325) ·
[code](https://github.com/fleurssauvages/Spike-MPPI-Manipulation/tree/master)) is used only to execute
tasks, on the floating hand or on the full arm.

## Phases

**1. Babble** (`bep/babble`). It discovers relations by clustering what holds at rest and what follows
the hand when doing random probes.

<p align="center"><img src="assets/readme/babble.gif" alt="babble"></p>

**2. Explore** (`bep/explore`). For each relation, a reverse Go-Explore graph grows back from its goal
states. The archive with
the best measured executor success supplies the cost-to-go field Φ.

<p align="center"><img src="assets/readme/explore.gif" alt="explore"></p>

**3. Play** (`bep/play`). Chained calls of the primitives fit effect models: success probability,
duration, relations made and broken.

<p align="center"><img src="assets/readme/play.gif" alt="play"></p>

**4. Brain** (`bep/brain`). The goal is an example arrangement. The snapshot brain searches forward from
the scene to its relation graph with Dijkstra, each call costing its expected time under the effects learned
in play, and stops at the first (cheapest) solution. Spike-MPPI executes each call.
<p align="center"><img src="assets/readme/tower_robot.gif" alt="tower on the arm"></p>

<p align="center"><em>4-cube tower, solved zero-shot on the Franka arm (brain learned on the floating hand only), real time; Phase clips at 2×.</em></p>

## Learning time

| Phase | `pair_cubes` |
|---|---:|
| Babble | 14.1 s |
| Explore | 18.8 s |
| Play | 9.8 s |
| **Total wall time** | **43.7 s** |

The run started from scratch. Totals include setup.

## Run

```bash
python -m bep.learn --scene pair_cubes                                          # learning -> runs/pair_cubes
python -m bep.brain.solve --scene cubes4 --task tower --brain snapshot --plan-once --motor robot
python -m bep.brain.solve --scene cubes4 --task restack --brain snapshot --plan-once --motor hog
python -m bep.render --scene pair_cubes                                         # clips -> runs/pair_cubes/clips
```

`solve` options: `--motor robot|hog`, `--brain tree|snapshot`, `--plan-once`, `--param` (e.g. `DCBA`, or
`DCBA>ABCD` for restack), `--episodes`, `--episode`, `--clip FILE.mp4`, `--verbose`.

The `arrange` task accepts slash-separated groups in bottom-to-top order. `--task example --param
FILE.json` accepts a map from free body names to `[x, y, z, yaw]` poses in metres and radians. The solver
pursues the learned relations in the example.

## Layout

```
bep/
  config.py     constants
  world/        robot / floating hand, bodies, sensors, relations, Hand of God, reachability map, scenes
  motor/        Spike-MPPI (arm and floating hand), native C++ rollouts, costs, executors
  babble/       probing, relation discovery
  explore/      primitives, archive, field, reverse Go-Explore
  play/         effect models, play
  brain/        goals, tasks, tree and snapshot brains, solve
  learn.py      babble -> explore -> play
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Python ≥ 3.10, MuJoCo 3.3, NumPy, SciPy, pybind11 with a C++17 compiler (native rollouts, built on first
use), imageio + ffmpeg and Pillow for clips, an EGL or OSMesa GL for offscreen rendering.
