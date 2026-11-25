# Babble, Explore, Play: Self-Discovered Manipulation for Zero-Shot Transfer

A Franka Panda learns manipulation primitives from two-body physics probes, then composes them in scenes
with more bodies. Learning runs on a **floating hand** (the gripper without the arm)
moved by the Hand of God, within the arm's reachability map. **Spike-MPPI**
([paper](https://arxiv.org/abs/2609.28325) ·
[code](https://github.com/fleurssauvages/Spike-MPPI-Manipulation/tree/master)) is used only to execute
tasks, on the floating hand or on the full arm.

## Learning: one curiosity-driven loop

<p align="center"><img src="assets/readme/loop_architecture.png" alt="the developmental loop" width="900"></p>

Learning is a sequence of **epochs** (`python -m bep.learn`). Within an epoch, curiosity hands about 60 s of
wall time to babble, explore, play and imagine, block by block; between epochs, sleep reorganises what was
learned. Each block is the counterpart of a mechanism of development or of the brain:

| Block | Biological analogue | Implementation |
|---|---|---|
| Curiosity | intrinsic motivation by learning progress | softmax bandit over activities (`bep/arbiter.py`) |
| Babble | motor babbling, event segmentation, contingency detection | unscripted moves, contact-transition event keys (`bep/babble`) |
| Explore | goal babbling | resumable reverse Go-Explore per schema (`bep/explore/explore.py`) |
| Play | play as competence testing | chained executed calls, effect models (`bep/play/play.py`) |
| Sleep | replay, systems consolidation | relations re-read, map refit, schemas revised (`bep/learn.py`) |
| Imagine | preplay, cerebellar internal model | event predictor, imagined goals, rehearsals (`bep/brain/imagine.py`) |
| Episodic memory | hippocampal index | validated graphs, play records, event store |
| Schemas | Piaget's schemas, accommodation | relation + graph + precondition network (`bep/play/schemas.py`) |
| Cognitive map | place and grid cells, predictive map | quasimetric network of the time to a goal (`bep/explore/qmap.py`) |
| Prefrontal cortex (basal ganglia) | goal-directed vs habitual control | search over predicted relation graphs, habits (`bep/brain/brain.py`) |
| Motor control | value, dopamine burst, withdrawal reflex | Spike-MPPI on Φ + discrete reward + contact barrier (`bep/motor`) |

**Curiosity** (`bep/arbiter.py`). Every activity reports a competence after each of its blocks (babble: how well
its events were predicted; explore: the schema's executor success; play: how well its calls' outcomes were
predicted; imagine: how close imagined and real outcomes were). The next block goes, by a softmax with a floor,
to the activity whose competence changes fastest: learning progress, not error. Babbling dominates while events
are new; then the time moves to exploring, playing and imagining:

<p align="center"><img src="assets/readme/epoch_budget.png" alt="wall time of three epochs, by activity" width="560"></p>

**Babble** (`bep/babble`). Unscripted motor babbling: random hand moves and finger commands from random
configurations, or from rare states and states the schemas reached (Go-Explore restarts). An *event* is the
change between two settled states, keyed by discrete indicators only: which contact flags changed, and how
the body moved. A key becomes a symbol when it is contingent on the hand's action (some context raises its
rate well above its base rate). Relations are read from the kept keys: X resting on or against Y (read at
rest after letting go), X carried by both fingers.

<p align="center"><img src="assets/readme/babble.gif" alt="babble"></p>

**Explore** (`bep/explore`). For each schema, a reverse Go-Explore graph grows back from its goal states,
in resumable blocks. Calls that failed in play become start states (first return, then explore).

<p align="center"><img src="assets/readme/explore.gif" alt="explore"></p>

**Play** (`bep/play`). Chained calls of the schemas fit effect models (success, duration, relations made and
broken) and record each call's context: contact flags, Φ, distance to the schema's graph.

<p align="center"><img src="assets/readme/play.gif" alt="play"></p>

**Imagine** (`bep/brain/imagine.py`). Goals whose imagined outcome is uncertain are planned, executed, and
compared with the imagination; an event predictor's surprise is babble's competence. Before acting on a task,
the first call of a plan is rehearsed in a floating-hand copy of the scene.

**Sleep** (between epochs). Every stored experience is replayed: relations are re-read from all events, with
stable names; the cognitive map is refit; the schemas' preconditions are refit, and schemas split or merge.

**Episodic memory**. What happened, as it happened: every babbling command's settled states and event keys,
each schema's validated graph (states, motions, their times to the goal), play's calls with their contexts, and
the start states of failed calls. Exploration revisits the failures; sleep replays the rest.

**Schemas** (`bep/play/schemas.py`). A schema is a relation the hand can make, its validated graph, a
precondition network (P(success | the call's context)) and its competence. When its outcomes vary within one
context, the discrete indicator that best separates successes from failures splits it (e.g. "placing on Y
needs X held"); schemas whose goals coincide merge.

**Cognitive map** (`bep/explore/qmap.py`). One small quasimetric network learns how long it takes to go from
any state to any other (asymmetric: lifting is not dropping), from every schema's graph. A schema's Φ is the
soft minimum of that time over its goal states, used where it predicts better than the schema's own
graph-based field on held-out samples.

**Prefrontal cortex and basal ganglia** (`bep/brain`). The goal is an example arrangement, read as required and
forbidden relation instances. The brain searches the relation graphs the effect models predict, each call
priced by its expected time with retries; it chooses table spots by Φ, blends the first call's success with the
schema's precondition, rehearses that call before acting, and takes decisions that keep succeeding as habits.

**Motor control** (`bep/motor`). Spike-MPPI executes each call, on the floating hand or on the arm, with the cost
Φ (value) + discrete success reward (dopamine burst) + contact barrier (withdrawal reflex).

<p align="center"><img src="assets/readme/cost_maps.png" alt="cost maps for grasping and placing, with and without an obstacle" width="820"></p>

Vertical slices of the executor's cost (dark: low, white: high), each panel over the range of its continuous part
Φ. Grasp: the open hand's pinch point around the cube; Φ from the cognitive map is a basin over the cube, raised
with it when it stands on another one. Place: the held cube around the target; Φ funnels down to the goal (the cube
resting on the target, outlined, where the discrete reward applies). The barrier is shown as a soft field around the
cube that must not be touched (it decays with the distance from the pinch point or the held cube); in the
controller it is a penalty on every rollout step with such a contact.

<p align="center"><img src="assets/readme/tower_robot.gif" alt="tower on the arm"></p>

## Run

```bash
python -m bep.learn --scene pair_cubes --epochs 4                 # -> runs/pair_cubes/epoch_<k>
python -m bep.brain.solve --scene cubes4 --task tower --plan-once --motor robot
python -m bep.brain.solve --scene cubes4 --task restack --plan-once --motor hog --learning runs/pair_cubes/epoch_3
python -m bep.assess --runs runs/pair_cubes/epoch_1,runs/pair_cubes/epoch_3 --tasks 3 --task-names tower,restack
```

`learn` options: `--epochs`, `--out` (default `runs/<scene>`; a run resumes from its saved state), `--seed`,
`--workers`. Each epoch is snapshot in `<out>/epoch_<k>`; `epochs.jsonl` logs every block and sleep.

`solve` options: `--motor robot|hog`, `--plan-once`, `--no-imagine`, `--learning DIR`, `--param` (e.g. `DCBA`,
or `DCBA>ABCD` for restack), `--episodes`, `--episode`, `--clip FILE.mp4`, `--verbose`. The `arrange` task
accepts slash-separated groups in bottom-to-top order. `--task example --param FILE.json` accepts a map from
free body names to `[x, y, z, yaw]` poses in metres and radians.

## Layout

```
bep/
  config.py     constants
  learn.py      the epoch loop and sleep
  arbiter.py    learning progress over activities
  workers.py    one process pool for every activity
  assess.py     the same yardstick for every learning directory
  world/        robot / floating hand, bodies, sensors, relations, Hand of God, reachability map, scenes
  motor/        Spike-MPPI (arm and floating hand), native C++ rollouts, costs, executors
  babble/       motor babbling, events, relations, primitive selection
  explore/      primitives, archive, field, reverse Go-Explore, the cost-to-go map
  play/         effect models, play, schemas (preconditions, splits, merges)
  brain/        goals, tasks, the brain, imagination, solve
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Python ≥ 3.10, MuJoCo 3.3, NumPy, SciPy, PyTorch (CPU), pybind11 with a C++17 compiler (native rollouts,
built on first use), imageio + ffmpeg and Pillow for clips, an EGL or OSMesa GL for offscreen rendering.
