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

The brain (bep.brain.brain) searches the relation graphs the effect models predict toward the goal's
literals, chooses table spots by the schemas' Phi, blends the first call's success with the schema's learned
precondition (bep.play.schemas, preconditions.npz); planning is symbolic (nothing is simulated before
acting), and a failed call is learned from before planning again.  The schemas' fields use the cost-to-go map where
it is trusted (bep.explore.qmap, qmap.npz of the learning directory), in Python and in the native rollouts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import BRAIN, CONTROL_DT, EXEC, HOG
from ..explore.library import bindings, learning_dirs, load_library, load_play
from ..motor.execute import execute_call
from ..play.effect import EffectModel
from ..terminal import ProgressBar
from ..world.geometry import Spot
from ..world.graph import as_tuples, object_poses, relation_graph
from ..world.scenes import table_spot_fn
from .goal import success_test
from .model import name
from .brain import LoopBrain


def models_for(R, lib, play, env) -> dict:
    """One effect model per primitive, fitted on the play records, with the task world's shapes."""
    out = {}
    for t in lib:
        recs = [r for r in play if r.get("primitive") == t]
        if recs:
            out[t] = EffectModel.for_relations(R, t).fit(recs).set_shapes({n: env.shape(n) for n in env.cubes},
                                                                          sorted(env.fixtures))
    return out


def load_learning(learning):
    """The library of the learning directories, with the map's fields where it is trusted, and the schemas'
    preconditions (one learning directory)."""
    R, lib = load_library(learning)
    pre = {}
    if len(learning) == 1:
        d = Path(learning[0])
        if (d / "qmap.npz").exists():
            from ..explore.qmap import QMap, install_native, wrap_library
            install_native()
            wrap_library(lib, QMap(d / "qmap.npz"))
        from ..play.schemas import load_preconditions
        pre = load_preconditions(d / "preconditions.npz")
    return R, lib, pre


def make_brain(env, R, lib, models, goal, seed: int, *, pre: dict, heuristic=None):
    actions = [(t, x, y) for t in lib for x, y in bindings(env, lib[t][0])]
    return LoopBrain(models, actions, env.cubes, goal, spot_fn=table_spot_fn(env), seed=seed, env=env, lib=lib, R=R,
                     pre=pre, heuristic=heuristic)


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
        if covered(prim, field, rel["z"], int(rel["mode"]), pair=prim.role_pair(env, x, target)):
            out.add(i)
    return out


def execute_brain_call(env, R, lib, ctl, key, spot, *, timeout, rec=None, stop=None, bar=None, elapsed=0.0) -> dict:
    """Run the brain's call (primitive, X, target; a table spot for target "spot")."""
    prim_name, x, tt = key
    target = Spot(float(spot[0]), float(spot[1]), 0.0) if tt == "spot" else tt
    show_progress = bar is not None and bar.enabled
    if show_progress:
        action = name(key)
        start_time = env.data.time
        bar.update(elapsed, action)

    def on_step(_):
        if rec:
            rec.frame()
        if show_progress:
            bar.update(elapsed + (env.data.time - start_time), action)

    prim, field, _ = lib[prim_name]
    out = execute_call(env, R, prim, field, ctl, x, target, timeout=timeout,
                       on_step=on_step if rec or show_progress else None, stop=stop)
    if show_progress:
        bar.update(elapsed + out["time"], action)
    return out


def rest(env, ctl, seconds: float = 0.5) -> float:
    """Hold the hand still with the fingers open for a moment (the goal's relations already hold)."""
    import mujoco
    env.data.ctrl[env.gi] = 255.0
    mujoco.mj_step(env.model, env.data, nstep=int(round(seconds / env.model.opt.timestep)))
    if ctl is not None:
        ctl.reset(q_cmd=env.arm_q(), g_cmd=env.gripper_width())
    return seconds


def planning_progress(bar, elapsed):
    if bar is None or not bar.enabled:
        return None
    bar.update(elapsed, "planning")

    def report(status, expanded, limit):
        if limit is None:
            bar.update(elapsed, status)
        else:
            bar.update_phase(expanded, limit, "states", status)

    return report


def run_plan_once(env, R, lib, models, ctl, goal, support, *, max_time, seed, verbose=False, rec=None, brain=None,
                  bar=None):
    """Plan once from the start (the brain's whole path to the goal), then execute it call by call; plan again
    from the current state only when a call fails (it does not end done, or does not make its predicted
    relation, e.g. a support knocked over while grasping)."""
    agent = brain(env, R, lib, models, goal, seed)
    t_wall = time.time()
    rels, poses = as_tuples(relation_graph(env, R)), object_poses(env)
    _, _, info = agent.decide(rels, poses, on_progress=planning_progress(bar, 0.0))
    plan_wall = time.time() - t_wall
    first = info
    plan, plans = list(info["plan_keys"]), [round(plan_wall, 2)]
    t, log = 0.0, []
    while plan:
        key = plan.pop(0)
        rels, poses = as_tuples(relation_graph(env, R)), object_poses(env)
        spot = agent.model.spot(poses, key[1]) if key[2] == "spot" else None
        agent.last_spot = spot
        out = execute_brain_call(env, R, lib, ctl, key, spot, timeout=min(EXEC["timeout_s"], max_time - t),
                                 rec=rec, bar=bar, elapsed=t)
        t += out["time"]
        rels2 = as_tuples(relation_graph(env, R))
        ok = agent.observe(rels, key, out["time"], rels2, poses, out["result"])["ok"]
        entry = {"t0": round(t - out["time"], 2), "action": name(key), **out, "unmet_after": goal.unmet(rels2),
                 "learn": {"ok": ok}}
        log.append(entry)
        finished = t >= max_time or entry["unmet_after"] == 0
        if not finished and (out["result"] != "done" or not ok):
            tp = time.time()
            _, _, info = agent.decide(rels2, object_poses(env), on_progress=planning_progress(bar, t))
            plans.append(round(time.time() - tp, 2))
            plan_wall += plans[-1]
            plan = list(info.get("plan_keys") or [])
            entry["replan"] = info.get("plan", [])
        if verbose:
            if bar:
                bar.clear()
            print("   ", json.dumps(entry, default=str), flush=True)
        if finished:
            break
    if goal.unmet(as_tuples(relation_graph(env, R))) == 0 and not success_test(env, support):
        if bar:
            bar.update(t, "settling")
        t += rest(env, ctl)
    result = dict(success=success_test(env, support), actions=log, decisions=len(log), plan=first["plan"],
                  plan_info={k: v for k, v in first.items() if k not in ("plan", "plan_keys")},
                  replans=len(plans) - 1, plans_s=plans,
                  plan_wall=round(plan_wall, 1), wall=round(time.time() - t_wall, 1), time=round(t, 2))
    if not result["success"] and not info["plan_keys"]:
        result["reason"] = "search limit" if info.get("state_limit_reached") else "no plan"
    return result


def run_task(env, R, lib, models, ctl, goal, support, *, max_time, seed, verbose=False, rec=None, brain=None,
             bar=None):
    agent = brain(env, R, lib, models, goal, seed)
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
            if bar:
                bar.update(t, "settling")
            t += rest(env, ctl)                       # the relations are right: let go and let the scene settle
            if success_test(env, support):
                return result(True, time=round(t, 2))
        tp = time.time()
        on_progress = planning_progress(bar, t)
        poses = object_poses(env)
        key, spot, info = agent.decide(rels, poses, covered_calls(env, lib, agent.model, poses), on_progress=on_progress)
        plan_wall += time.time() - tp
        if key is None:
            reason = "search limit" if info.get("state_limit_reached") else "no plan"
            return result(False, time=None, reason=reason, info=info)
        out = execute_brain_call(env, R, lib, ctl, key, spot, timeout=min(EXEC["timeout_s"], max_time - t),
                                 rec=rec, bar=bar, elapsed=t)
        t += out["time"]
        rels2 = as_tuples(relation_graph(env, R))
        entry = {"t0": round(t - out["time"], 2), "action": name(key), **out, "plan": info["plan"],
                 "plan_s": info.get("plan_s"), "unmet_before": goal.unmet(rels), "unmet_after": goal.unmet(rels2),
                 "learn": agent.observe(rels, key, out["time"], rels2, poses, out["result"])}
        log.append(entry)
        if verbose:
            if bar:
                bar.clear()
            print("   ", json.dumps(entry, default=str), flush=True)
    return result(False, time=None, reason="time")


def main() -> None:
    from ..motor.mppi import MPPIConfig, SpikeMPPI
    from ..world.scenes import SCENES, build_scene, start_layout
    from .tasks import BUILDERS

    tower_max_time = 40.0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), required=True)
    ap.add_argument("--task", choices=list(BUILDERS), required=True)
    ap.add_argument("--param", default=None, help="task parameter (default: the scene's default)")
    ap.add_argument("--learning", default=None,
                    help="learning directory, or several comma-separated (default: latest saved learning under runs/ "
                         "for each source scene)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=None,
                    help=f"simulated seconds per episode (default {tower_max_time:g} s for tower, "
                         f"{EXEC['timeout_s']:g} s per body for other tasks)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", action="store_true", help="print machine-readable episode results and summary")
    ap.add_argument("--out", default=None, help="save detailed results to a JSON file")
    ap.add_argument("--clip", default=None, help="render the first episode to this .mp4 (every episode if the path "
                                                 "contains {ep}, e.g. runs/x_ep{ep}.mp4)")
    ap.add_argument("--save", default="runs/last_run.npz",
                    help="record each episode's simulation state for bep.replay ('' to skip; with several episodes and "
                         "no {ep} in the path, episode k is saved as <name>_ep<k>.npz)")
    ap.add_argument("--motor", choices=["robot", "hog"], default="robot",
                    help="robot: the Franka arm and Spike-MPPI; hog: the floating hand and the Hand-of-God motor")
    ap.add_argument("--episode", type=int, default=None,
                    help="run only this episode (the earlier layouts are drawn, not executed, so it starts the same)")
    ap.add_argument("--heuristic", choices=("learned", "count"), default=BRAIN["heuristic"],
                    help="the search's estimate of the time to the goal: the literal costs learned from play's records "
                         "(bep.brain.heuristic), or BRAIN['goal_weight_s'] per unmet literal")
    ap.add_argument("--plan-once", action="store_true",
                    help="plan once from the start and execute the plan, replanning only after a failed call")
    ap.add_argument("--vertical-hand", action="store_true",
                    help="keep the gripper pointing down (default: 6-DoF hand motions)")
    ap.add_argument("--substeps", type=int, default=2,
                    help="physics steps per 10 ms control step in Spike-MPPI's rollouts (the real simulation keeps 2): "
                         "2 (default) matches the real physics, 1 is about 1.6x faster but plans worse")
    ap.add_argument("--horizon", type=int, default=None,
                    help=f"Spike-MPPI horizon in control steps (default: {EXEC['arm_horizon']} on the arm, "
                         f"{HOG['horizon']} on the floating hand)")
    args = ap.parse_args()
    if args.horizon is not None:
        if args.horizon < 1:
            ap.error("--horizon must be at least 1")
        EXEC["arm_horizon"] = HOG["horizon"] = args.horizon    # this process only: the arm's and the hand's planner

    spec = SCENES[args.scene]
    param = args.param if args.param is not None else spec.tasks.get(args.task, (None, ""))[0]
    if param is None:
        ap.error(f"{args.task} requires --param for {args.scene}")
    try:
        learning = learning_dirs(spec, args.learning)
    except FileNotFoundError as exc:
        ap.error(str(exc))
    from ..world.objects import use_body
    use_body("float" if args.motor == "hog" else "robot")
    env = build_scene(spec, f"solve{os.getpid()}")
    R, lib, pre = load_learning(learning)
    play = load_play(learning, R)
    heuristic = None
    if args.heuristic == "learned":
        from .heuristic import LiteralCost
        heuristic = LiteralCost().fit(play, R, seed=args.seed)
    ctl = None if args.motor == "hog" else \
        SpikeMPPI(env, MPPIConfig(threads=args.threads, seed=args.seed, free_tilt=not args.vertical_hand,
                                  horizon=EXEC["arm_horizon"], rollout_substeps=args.substeps))
    rng = np.random.default_rng(args.seed)
    results = []
    for ep in range(args.episodes):
        models = models_for(R, lib, play, env)           # fresh per episode: no learning carried across episodes
        start_layout(env, spec, rng)
        goal, support = BUILDERS[args.task](env, R, param)
        default_time = tower_max_time if args.task == "tower" else EXEC["timeout_s"] * max(1, len(goal.poses))
        max_time = args.max_time or default_time
        if args.episode is not None and ep != args.episode:
            continue
        init = env.facts()
        start_time = env.data.time
        clip_path = None
        with ProgressBar(f"{args.task} ep {ep}", max_time, "sim s") as bar:
            bar.update(0.0, "preparing")
            clip = log = None
            if args.clip and (ep == 0 or "{ep}" in args.clip):
                from ..world.clips import Recorder
                clip_path = args.clip.format(ep=ep)
                clip = Recorder(env, clip_path, every=4, max_s=max_time + 5.0)
                clip.hold(0.5)
            if args.save:
                from ..replay import StateLog
                save_path = Path(args.save.format(ep=ep)) if "{ep}" in args.save else Path(args.save)
                if "{ep}" not in args.save and args.episodes > 1 and args.episode is None:
                    save_path = save_path.with_name(f"{save_path.stem}_ep{ep}{save_path.suffix}")
                log = StateLog(env, save_path, {"scene": args.scene, "body": "float" if args.motor == "hog" else "robot",
                                                "task": args.task, "param": param, "motor": args.motor,
                                                "seed": args.seed, "episode": ep, "learning": ",".join(map(str, learning))})
                log.frame(force=True)
            from ..replay import Tee
            rec = Tee(clip, log) if (clip or log) else None
            r = (run_plan_once if args.plan_once else run_task)(env, R, lib, models, ctl, goal, support,
                         max_time=max_time, seed=args.seed + ep, verbose=args.verbose, rec=rec, bar=bar,
                         brain=lambda *a: make_brain(*a, pre=pre, heuristic=heuristic))
            elapsed = r["time"] if r["time"] is not None else env.data.time - start_time
            if log:
                log.meta.update(success=bool(r["success"]), time=elapsed, actions=list(r.get("actions", [])))
            if rec:
                bar.update(elapsed, "saving clip" if clip else "saving recording")
                rec.hold(1.0)
                rec.save()
            status = "solved" if r["success"] else r.get("reason", "unsolved")
            if status == "time":
                status = "time limit"
            if args.json:
                bar.update(elapsed, status)
            else:
                bar.clear()
        r.update(episode=ep, goal=sorted(map(list, goal.require)), goal_forbid=sorted(map(list, goal.forbid)),
                 initial=init["support"])
        results.append(r)
        if args.json:
            print(json.dumps({k: v for k, v in r.items() if k not in ("actions", "goal", "goal_forbid", "initial")}),
                  flush=True)
        else:
            outcome = "Solved" if r["success"] else "Unsolved"
            if status not in ("solved", "unsolved"):
                outcome += f" ({status})"
            actions = f"{r['decisions']} {'action' if r['decisions'] == 1 else 'actions'}"
            print(f"Episode {ep}  {outcome}  ·  {actions}  ·  {elapsed:.1f} s sim  ·  {r['wall']:.1f} s wall",
                  flush=True)
            if clip_path is not None:
                print(f"  Clip  {clip_path}", flush=True)
            if log is not None:
                print(f"  Recording  {log.path}  (python -m bep.replay --file {log.path})", flush=True)
    succ = [r["success"] for r in results]
    summary = {"scene": args.scene, "task": args.task, "param": param, "motor": args.motor, "plan_once": args.plan_once,
               "heuristic": args.heuristic if heuristic is None else {"learned": heuristic.report()},
               "free_tilt": not args.vertical_hand,
               "horizon": EXEC["arm_horizon"] if args.motor == "robot" else HOG["horizon"],
               "rollout_substeps": args.substeps,
               "episodes": len(results),
               "success_rate": float(np.mean(succ)) if results else None,
               "median_time_s": float(np.median([r["time"] for r in results if r["success"]])) if any(succ) else None,
               "mean_decisions": float(np.mean([r["decisions"] for r in results])) if results else None,
               "mean_plan_wall_s": float(np.mean([r["plan_wall"] for r in results])) if results else None}
    if args.json:
        print("SUMMARY", json.dumps(summary))
    elif len(results) > 1:
        totals = f"Solved {sum(succ)}/{len(results)} episodes ({summary['success_rate']:.0%})"
        if summary["median_time_s"] is not None:
            totals += f"  ·  {summary['median_time_s']:.1f} s median sim time"
        print(f"\n{totals}")
    elif not results:
        print("No episodes ran.")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1, default=str))


if __name__ == "__main__":
    from ..terminal import run_main
    run_main(main)
