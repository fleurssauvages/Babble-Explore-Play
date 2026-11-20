"""Zero-shot tasks in a scene: a brain chooses primitive calls, Spike-MPPI executes them.

    python -m bep.brain.solve --scene cubes4 --task tower --episodes 5
    python -m bep.brain.solve --scene cubes6 --task tower --param FEDCBA --episodes 5

A task is a physical initial state and a goal *example* (the bodies arranged as they should end up,
bep.brain.tasks).  The discovered relations that hold among the task's bodies in the example are the
goal literals.  The tasks never occur during learning, so they are solved zero-shot from what babble,
explore and play learned (in the minimal scene the task's scene is learned from).

Motors (--motor): the robot (Franka arm + Spike-MPPI, the default) or the Hand of God (the floating hand
and its motor, as in learning: the brain's plans without the robot's execution gap).

Loop: read the relation graph -> the brain picks (primitive, X, target) -> Spike-MPPI runs the primitive
until done, stalled, timed out, or a relation it should keep breaks -> the brain learns from the real
outcome -> repeat until the goal holds.  Success is scored with the shape-aware support facts of the
example (ObjWorld.facts).

The brain is a tree search over the relation graphs the effect models predict, pricing the success and
the failure of every call (bep.brain.search).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import CONTROL_DT, EXEC, HOG
from ..explore.library import bindings, learning_dirs, load_library, load_play
from ..motor.execute import execute_call
from ..play.effect import EffectModel
from ..world.geometry import Spot
from ..world.graph import as_tuples, object_poses, relation_graph
from ..world.scenes import table_spot_fn
from .goal import success_test
from .model import name
from .search import TreeBrain
from .snapshot import SnapshotBrain

BRAINS = {"tree": TreeBrain, "snapshot": SnapshotBrain}


def models_for(R, lib, play, env) -> dict:
    """One effect model per primitive, fitted on the play records, with the task world's shapes."""
    out = {}
    for t in lib:
        recs = [r for r in play if r.get("primitive") == t]
        if recs:
            out[t] = EffectModel.for_relations(R, t).fit(recs).set_shapes({n: env.shape(n) for n in env.cubes},
                                                                          sorted(env.fixtures))
    return out


def make_brain(env, lib, models, goal, seed: int, brain: str = "tree"):
    actions = [(t, x, y) for t in lib for x, y in bindings(env, lib[t][0])]
    return BRAINS[brain](models, actions, env.cubes, goal, spot_fn=table_spot_fn(env), seed=seed)


def covered_calls(env, lib, model, poses) -> set:
    """Indices (in model.keys) of the calls whose primitive has validated experience from the current state
    (bep.explore.field.covered), as play requires of its calls."""
    from ..explore.field import covered
    from ..world.objects import VIEW_S
    out = set()
    sens = env.sensors()
    for i, (t, x, tt) in enumerate(model.keys):
        prim, field, _ = lib[t]
        target = Spot(*model.spot(poses, x), 0.0) if tt == "spot" else tt
        rel = prim.rel(env, env.pair_view(sens, x, prim.reference(env, x, target)), VIEW_S)
        if covered(prim, field, rel["z"], int(rel["mode"])):
            out.add(i)
    return out


def execute_brain_call(env, R, lib, ctl, key, spot, *, timeout, rec=None, stop=None) -> dict:
    """Run the brain's call (primitive, X, target; a table spot for target "spot")."""
    prim_name, x, tt = key
    prim, field, _ = lib[prim_name]
    target = Spot(float(spot[0]), float(spot[1]), 0.0) if tt == "spot" else tt
    return execute_call(env, R, prim, field, ctl, x, target, timeout=timeout,
                        on_step=(lambda k: rec.frame()) if rec else None, stop=stop)


def rest(env, ctl, seconds: float = 0.5) -> float:
    """Hold the hand still with the fingers open for a moment (the goal's relations already hold)."""
    import mujoco
    env.data.ctrl[env.gi] = 255.0
    mujoco.mj_step(env.model, env.data, nstep=int(round(seconds / env.model.opt.timestep)))
    if ctl is not None:
        ctl.reset(q_cmd=env.arm_q(), g_cmd=env.gripper_width())
    return seconds


def run_plan_once(env, R, lib, models, ctl, goal, support, *, max_time, seed, verbose=False, rec=None,
                  brain="snapshot"):
    """Plan once from the start (the brain's whole path to the goal), then execute it call by call, without
    replanning: a call that does not make its predicted relation (e.g. a grasp the relations do not recognise)
    is followed by the next predicted call all the same."""
    agent = make_brain(env, lib, models, goal, seed, brain)
    t_wall = time.time()
    rels, poses = as_tuples(relation_graph(env, R)), object_poses(env)
    _, _, info = agent.decide(rels, poses)
    plan_wall = time.time() - t_wall
    t, log = 0.0, []
    for key in info["plan_keys"]:
        rels, poses = as_tuples(relation_graph(env, R)), object_poses(env)
        spot = agent.model.spot(poses, key[1]) if key[2] == "spot" else None
        agent.last_spot = spot
        out = execute_brain_call(env, R, lib, ctl, key, spot, timeout=min(EXEC["timeout_s"], max_time - t), rec=rec)
        t += out["time"]
        rels2 = as_tuples(relation_graph(env, R))
        ok = agent.observe(rels, key, out["time"], rels2, poses, out["result"])["ok"]
        entry = {"t0": round(t - out["time"], 2), "action": name(key), **out, "unmet_after": goal.unmet(rels2),
                 "learn": {"ok": ok}}
        log.append(entry)
        if verbose:
            print("   ", json.dumps(entry, default=str), flush=True)
        if t >= max_time:
            break
    if goal.unmet(as_tuples(relation_graph(env, R))) == 0 and not success_test(env, support):
        t += rest(env, ctl)
    return dict(success=success_test(env, support), actions=log, decisions=len(log), plan=info["plan"],
                plan_info={k: v for k, v in info.items() if k not in ("plan", "plan_keys")},
                plan_wall=round(plan_wall, 1), wall=round(time.time() - t_wall, 1), time=round(t, 2))


def run_task(env, R, lib, models, ctl, goal, support, *, max_time, seed, verbose=False, rec=None, brain="tree"):
    agent = make_brain(env, lib, models, goal, seed, brain)
    if ctl is not None:
        ctl.reset(q_cmd=env.arm_q(), g_cmd=env.gripper_width())
    dt = CONTROL_DT
    t, log, plan_wall, t_wall = 0.0, [], 0.0, time.time()

    def result(success, **kw):
        return dict(success=success, actions=log, decisions=len(log), plan_wall=round(plan_wall, 1),
                    wall=round(time.time() - t_wall, 1), **kw)

    while t < max_time - dt:
        rels = as_tuples(relation_graph(env, R))
        if goal.unmet(rels) == 0:
            if success_test(env, support):                                 # checked at once, no waiting
                return result(True, time=round(t, 2))
            t += rest(env, ctl)                       # the relations are right: let go and let the scene settle
            if success_test(env, support):
                return result(True, time=round(t, 2))
        tp = time.time()
        poses = object_poses(env)
        key, spot, info = agent.decide(rels, poses, covered_calls(env, lib, agent.model, poses))
        plan_wall += time.time() - tp
        if key is None:
            return result(False, time=None, reason="no plan", info=info)
        out = execute_brain_call(env, R, lib, ctl, key, spot, timeout=min(EXEC["timeout_s"], max_time - t), rec=rec)
        t += out["time"]
        rels2 = as_tuples(relation_graph(env, R))
        entry = {"t0": round(t - out["time"], 2), "action": name(key), **out, "plan": info["plan"],
                 "plan_s": info.get("plan_s"), "unmet_before": goal.unmet(rels), "unmet_after": goal.unmet(rels2),
                 "learn": agent.observe(rels, key, out["time"], rels2, poses, out["result"])}
        log.append(entry)
        if verbose:
            print("   ", json.dumps(entry, default=str), flush=True)
    return result(False, time=None, reason="time")


def main() -> None:
    from ..motor.mppi import MPPIConfig, SpikeMPPI
    from ..world.scenes import SCENES, build_scene, start_layout
    from .tasks import BUILDERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--task", required=True, help="a task of the scene (bep.world.scenes)")
    ap.add_argument("--param", default=None, help="task parameter (default: the scene's default)")
    ap.add_argument("--learning", default=None,
                    help="learning directory, or several comma-separated (default: the scene's minimal scenes)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-time", type=float, default=None,
                    help="simulated seconds per episode (default 15 s per body of the task)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--clip", default=None, help="render the first episode to this .mp4 (every episode if the path "
                                                 "contains {ep}, e.g. runs/x_ep{ep}.mp4)")
    ap.add_argument("--motor", choices=["robot", "hog"], default="robot",
                    help="robot: the Franka arm and Spike-MPPI; hog: the floating hand and the Hand-of-God motor")
    ap.add_argument("--models", choices=["hog", "robot"], default=None,
                    help="what the brain's effect models are fitted on: hog = the floating hand's play; robot = the "
                         "floating hand's play + the robot's validation calls (default: the motor's)")
    ap.add_argument("--brain", choices=list(BRAINS), default="tree",
                    help="tree: forward tree search over goal literals; snapshot: backward from the goal example")
    ap.add_argument("--episode", type=int, default=None,
                    help="run only this episode (the earlier layouts are drawn, not executed, so it starts the same)")
    ap.add_argument("--plan-once", action="store_true",
                    help="plan once from the start and execute the whole plan without replanning")
    ap.add_argument("--vertical-hand", action="store_true",
                    help="keep the gripper pointing down (default: 6-DoF hand motions)")
    args = ap.parse_args()

    spec = SCENES[args.scene]
    default, _ = spec.tasks[args.task]
    param = args.param or default
    learning = learning_dirs(spec, args.learning)
    from ..world.objects import use_body
    use_body("float" if args.motor == "hog" else "robot")
    env = build_scene(spec, f"solve{os.getpid()}")
    R, lib = load_library(learning)
    models_from = args.models or args.motor
    play = load_play(learning, R, robot=models_from == "robot")  # + the robot's validation calls (bep.validate)
    ctl = None if args.motor == "hog" else \
        SpikeMPPI(env, MPPIConfig(threads=args.threads, seed=args.seed, free_tilt=not args.vertical_hand,
                                  horizon=HOG["horizon"]))
    rng = np.random.default_rng(args.seed)
    max_time = args.max_time or 15.0 * len(param)
    results = []
    for ep in range(args.episodes):
        models = models_for(R, lib, play, env)           # fresh per episode: no learning carried across episodes
        start_layout(env, spec, rng)
        goal, support = BUILDERS[args.task](env, R, param)
        if args.episode is not None and ep != args.episode:
            continue
        init = env.facts()
        rec = None
        if args.clip and (ep == 0 or "{ep}" in args.clip):
            from ..world.clips import Recorder
            rec = Recorder(env, args.clip.format(ep=ep), every=4, max_s=max_time + 5.0)
            rec.hold(0.5)
        r = (run_plan_once if args.plan_once else run_task)(env, R, lib, models, ctl, goal, support, max_time=max_time,
                     seed=args.seed + ep, verbose=args.verbose, rec=rec, brain=args.brain)
        if rec:
            rec.hold(1.0)
            rec.save()
        r.update(episode=ep, goal=sorted(map(list, goal.require)), initial=init["support"])
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("actions", "goal", "initial")}), flush=True)
    succ = [r["success"] for r in results]
    summary = {"scene": args.scene, "task": args.task, "param": param, "motor": args.motor, "models": models_from, "brain": args.brain, "plan_once": args.plan_once,
               "free_tilt": not args.vertical_hand,
               "episodes": len(results),
               "success_rate": float(np.mean(succ)),
               "median_time_s": float(np.median([r["time"] for r in results if r["success"]])) if any(succ) else None,
               "mean_decisions": float(np.mean([r["decisions"] for r in results])),
               "mean_plan_wall_s": float(np.mean([r["plan_wall"] for r in results]))}
    print("SUMMARY", json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=str))


if __name__ == "__main__":
    main()
