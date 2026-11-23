"""Explore: reverse Go-Explore toward one discovered relation.

    python -m bep.explore.explore --relations runs/pair_cubes/relations --probes runs/pair_cubes/probes.npz \
        --target r0 --out runs/pair_cubes/prim_r0

Loop: choose an arm -> propose an earlier configuration from a solved node -> Hand of God -> forward
validation into any solved epsilon-ball -> archive -> Dijkstra.  The forward validation moves the
floating hand with the Hand-of-God motor (bep.motor.hog_motor); no sampling planner in learning.

* goal = the target relation under its grasp or release conditions; seeds are reproduced probe states; a validated
  candidate already in the goal mode extends the goal set,
* contact modes = bitmasks of discovered relations; states in object units, so nodes of every pair of
  shapes share one graph,
* arms: the backward operators (move X, move the hand, turn the hand, change the grip, switch a relation
  on, move a joint, swap shapes) and "dir" (step the solved node closest to a start configuration toward it).
  A bandit picks them by yield, new solved nodes per simulated second, with a floor,
* proposals are resampled until the Hand of God's geometric pre-check passes and their relational state
  is not predicted to land on a solved node (checked before simulating, for every operator),
* step lengths adapt so that validation succeeds in a target band,
* edges are re-validated now and then (``revalidate_p``): Dijkstra prices them by expected time c / p,
* candidates run in an asynchronous queue; a validation whose distance to its target stalls is aborted,
* the distance is learned: per mode, the fixed weights rescaled by how fast each coordinate moves there
  (bep.explore.field.learned_metric), refreshed with the field,
* stop: once the compacted field has converged and covers the start states (or its coverage has stopped
  improving), the real executor is tested from fresh starts; exploring goes on only while its success
  rate improves significantly,
* trajectory states that are physical transients (X moving with no relation holding) are stored with
  mode -1 and ignored by the field.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..config import EXPLORE, progress, task_seed
from ..motor.hog_motor import hog_validate, run_primitive_hop
from ..world.objects import pair_world, use_body
from ..world.relations import HC, RelationSet
from .archive import Archive, Node
from .field import ModeField, learned_metric
from .primitive import DiscoveredPrimitive, ctx_pair, load_end_states, restore

TRANSIENT = -1
_W: dict = {}


# ------------------------------------------------------------------ workers
def _with_chain(prim, chain: str):
    """Earlier primitives seeding the start states (``target=dir,...``, their exploration directories): a
    chained primitive starts where their executed calls actually ended (dir/ends.npz)."""
    if chain:
        ends = []
        for item in chain.split(","):
            target, path = item.split("=", 1)
            path = Path(path)
            specs = path.parent / "primitives.json"
            fallback = next((p["kinds"] for p in json.loads(specs.read_text()) if p["target"] == target),
                            prim.kinds) if specs.exists() else prim.kinds
            ends.append(load_end_states(path / "ends.npz", fallback))
        prim.set_chain(ends)
    return prim


def _init_worker(rel_path: str, probes: str, target: str, chain: str = "") -> None:
    use_body("float")
    prim = _with_chain(DiscoveredPrimitive(RelationSet.load(rel_path), target, probes=probes), chain)
    _W.update(prim=prim)


def _test(task: dict) -> dict:
    """Realise a backward candidate and validate it forward with Spike-MPPI, or (``edge`` set) re-run a
    validated edge from its source node's state."""
    prim = _W["prim"]
    prim.steps.update(task["steps"])
    prim.set_metric(task["metric"])
    out = {"parent": task["parent"], "op": task["op"], "arm": task["arm"], "B_pose": task["ctx"], "sim_steps": 0,
           "edge": task.get("edge")}
    pair = ctx_pair(task["ctx"])
    if task.get("edge") is not None:                     # re-validation of an edge
        env = pair_world(*pair)
        restore(env, task["state"], task["ctrl"], task["ctx"])
        rel0 = prim.rel(env, env.sensors())
        return hog_validate(prim, env, rel0, task, out, lambda: env.pair_view(env.sensors(), "A", prim.reference(env)))
    r = prim.realize(None, task["cfg"], task["ctx"], q_hint=task.get("q_hint"))
    if isinstance(r, str):
        out["status"] = "invalid:" + r
        return out
    state, ctrl, rel0 = r
    Zs, Ms, _, _ = task["solved"]
    m0 = int(rel0["mode"])
    same = Ms == m0
    if np.any(same) and prim.distance(rel0["z"], Zs[same], m0).min() < EXPLORE["novelty"] * prim.eps[m0]:
        out["status"] = "redundant"
        return out
    out.update(state=state, ctrl=ctrl)
    if prim.is_goal(rel0):
        out.update(status="goal", z0=rel0["z"], m0=m0)
        return out
    env = pair_world(*pair)
    env.set_state(state, ctrl)
    return hog_validate(prim, env, rel0, task, out, lambda: env.pair_view(env.sensors(), "A", prim.reference(env)))


# ------------------------------------------------------------------ proposals
def select(archive: Archive, rng, goal_mode: int) -> int:
    """Mode-balanced selection, favouring nodes far from the goal and rarely tried."""
    D = np.array([nd.D for nd in archive.nodes])
    modes = np.array([nd.mode for nd in archive.nodes])
    tries = np.array([nd.tries for nd in archive.nodes])
    ok = np.isfinite(D)
    present = sorted(set(modes[ok].tolist()))
    pm = np.array([EXPLORE["goal_select_weight"] if m == goal_mode else 1.0 for m in present])
    m = present[rng.choice(len(present), p=pm / pm.sum())]
    idx = np.nonzero(ok & (modes == m))[0]
    dmax = max(D[idx].max(), 1e-6)
    w = (0.3 + D[idx] / dmax) / np.sqrt(1.0 + tries[idx])
    return int(idx[rng.choice(len(idx), p=w / w.sum())])


class NoGoalSeed(ValueError):
    """No observed configuration could be reproduced in the exploration world."""


def seed_goals(archive: Archive, prim, rng, *, limit: int = EXPLORE["n_goal_seeds"]) -> None:
    """Reproduce observed goal configurations in the exploration world."""
    from ..world.hog import GRIP_CLOSE, GRIP_HOLD
    from ..world.geometry import quat_to_mat
    from ..world.objects import SPOT, VIEW_S
    if prim.kind == HC:                    # hand-object targets: the probe events in which they held
        s2 = prim.probes["ev_s"]
        pairs = [prim.probe_pairs[i] for i in prim.probes["ev_probe"]]
    else:
        s2, pairs = prim.oriented()
    holds = prim.R.holds_all(s2, VIEW_S)[prim.target]
    if prim.goal_requires_lift:
        holds &= (prim.probes["ev_kind"] == 2) & prim.probes["ev_follow"]
    if prim.kind == "object-state":
        holds = holds & (s2[:, VIEW_S.A_art] > 0)
    cand = np.nonzero(holds)[0]
    rng.shuffle(cand)
    ctrl0 = np.zeros(pair_world(*prim.kinds).model.nu)
    ctrl0[-1] = 255.0                                  # the gripper open
    per_pair = Counter()
    share = max(1, int(np.ceil(limit / len(prim.training_pairs))))
    for i in cand:
        if len(archive.nodes) >= limit:
            break
        pair = pairs[i]
        if pair not in prim.training_pairs and (prim.kind == "object-object" or pair[0] != prim.kinds[0]):
            continue
        if pair not in prim.training_pairs:
            pair = prim.kinds
        if per_pair[pair] >= share:
            continue
        cfg = prim.cfg_from_view(s2[i], ctrl0, pair)
        if prim.kind == HC and not prim.x_is_part:         # X held where it was, above the virtual reference
            cfg.pXY, cfg.yawXY = np.array([0.0, 0.0, float(s2[i][VIEW_S.A_pos][2]) + SPOT.half[2]]), 0.0
        cfg.grip = GRIP_CLOSE if prim.goal_requires_lift else GRIP_HOLD
        ctx = prim.sample_ctx(rng, pair)
        r = prim.realize(None, cfg, ctx)
        if prim.goal_requires_lift and (isinstance(r, str) or not prim.is_goal(r[2])):
            # A grasp whose approach is only reachable at its observed workspace pose
            # can still seed the graph there.
            rotation = quat_to_mat(s2[i][VIEW_S.A_quat])
            ctx[:2] = s2[i][VIEW_S.A_pos][:2]
            ctx[2] = np.arctan2(rotation[1, 0], rotation[0, 0])
            r = prim.realize(None, cfg, ctx)
        if isinstance(r, str) or not bool(prim.is_goal(r[2])):
            continue
        state, ctrl, rel = r
        archive.add_node(Node(rel["z"], int(rel["mode"]), state, ctrl, ctx, "goal", -1, goal=True, D=0.0))
        per_pair[pair] += 1
    if not any(nd.goal for nd in archive.nodes):
        raise NoGoalSeed(f"{prim.target} has no reproducible goal state in the probe data")
    archive.dijkstra()


def directed(archive: Archive, prim, rng):
    """Step the solved (non-goal) node closest to a start configuration toward it, re-expressed
    for the start's shapes."""
    start, _, zs, _ = prim.sample_start(rng)
    D = np.array([nd.D for nd in archive.nodes])
    ok = np.nonzero(np.isfinite(D) & np.array([not nd.goal for nd in archive.nodes]))[0]
    if not len(ok):
        ok = np.nonzero(np.isfinite(D))[0]
    if len(prim.training_pairs) > 1:
        matching = np.array([i for i in ok if ctx_pair(archive.nodes[i].B_pose) == start.pair], int)
        if len(matching):
            ok = matching
    Z = np.array([archive.nodes[i].z for i in ok])
    W = prim.metric_of(0)
    gap = (np.abs(Z - zs) * W).sum(1)                  # every coordinate (joints and opening included)
    i = int(rng.choice(ok[np.argsort(gap)[:EXPLORE["directed_top"]]]))
    nd = archive.nodes[i]
    cfg = prim.node_cfg(nd)
    if cfg.pair != start.pair:
        cfg = prim.cfg_from_z(nd.z, start.pair, cfg.grip)
    op, cfg = prim.toward(cfg, start, rng)
    return i, op, cfg


def adapt_steps(steps: dict, window: dict) -> None:
    lo, hi = EXPLORE["target_rate"]
    g = EXPLORE["step_gain"]
    for op in ("XY", "EX"):
        w = window[op][-EXPLORE["window"]:]
        if len(w) < EXPLORE["min_window"]:
            continue
        rate = float(np.mean(w))
        if rate > hi:
            steps[op] = min(EXPLORE["step_bounds"][1], steps[op] * g)
        elif rate < lo:
            steps[op] = max(EXPLORE["step_bounds"][0], steps[op] / g)
        if rate > hi or rate < lo:
            window[op] = window[op][-6:]


def predicted_redundant(prim, op: str, cfg, parent_mode: int, solved) -> bool:
    """Skip, before simulating, a proposal whose relational state (known from the configuration) lies very
    close to a solved node.  Motion operators keep the parent's mode: compared with its nodes; the others
    usually change the mode, which is unknown until the Hand of God settles: compared with every solved
    node (a state that close to a node almost always settles in that node's mode)."""
    Zs, Ms, _, _ = solved
    z = prim.z_of_cfg(cfg)
    r = EXPLORE["prefilter"] * EXPLORE["novelty"] * prim.eps[parent_mode]
    if op.rstrip("*") in ("XY", "EX"):
        same = Ms == parent_mode
        return bool(np.any(same) and prim.distance(z, Zs[same], parent_mode).min() < r)
    for m in np.unique(Ms):
        if prim.distance(z, Zs[Ms == m], m).min() < r:
            return True
    return False


def solved_for_pair(archive, prim, solved, pair):
    """Only compare with states whose supporting bodies have the same geometry."""
    if len(prim.training_pairs) == 1:
        return solved
    compatible = np.array([ctx_pair(archive.nodes[i].B_pose) == pair for i in solved[3]], bool)
    return tuple(a[compatible] for a in solved)


class Arms:
    """Bandit over proposal arms: probability ∝ yield (new solved nodes per simulated second, smoothed by a
    prior, over the arm's last ``arm_window`` tests), mixed with a floor."""

    def __init__(self, prim):
        ok = {"XY": not prim.x_is_part, "EX": True, "rot": True, "grip": True, "rel": bool(prim.pools),
              "joint": prim.x_is_part or prim.y_is_part,
              "swap": len(prim.training_pairs) > 1, "dir": True}
        self.names = [a for a in EXPLORE["ops"] if ok.get(a)]
        self.hist = {a: deque(maxlen=EXPLORE["arm_window"]) for a in self.names}

    def probs(self) -> np.ndarray:
        s0, c0 = EXPLORE["arm_prior"]
        rate = np.array([(s0 + sum(g for g, _ in self.hist[a])) / (c0 + sum(c for _, c in self.hist[a]))
                         for a in self.names])
        f = min(EXPLORE["arm_floor"], 1.0 / len(self.names))
        return f + (1.0 - f * len(self.names)) * rate / rate.sum()

    def pick(self, rng) -> str:
        return self.names[int(rng.choice(len(self.names), p=self.probs()))]

    def update(self, arm: str, gain: float, cost: float) -> None:
        if arm in self.hist:
            self.hist[arm].append((float(gain), float(cost)))

    def summary(self) -> dict:
        return {a: round(float(p), 3) for a, p in zip(self.names, self.probs())}


def proposal(arm: str, prim, rng, archive, solved):
    """A proposal from the arm, resampled until the Hand of God's geometric pre-check passes and it is not
    predicted redundant (or give up and use the last one)."""
    for _ in range(EXPLORE["precheck_tries"]):
        if arm == "dir":
            i, op, cfg = directed(archive, prim, rng)
        else:
            i = select(archive, rng, prim.goal_mode)
            op, cfg = prim.propose(prim.node_cfg(archive.nodes[i]), rng, op=arm)
        ctx = prim.sample_ctx(rng, cfg.pair)
        if prim.precheck(cfg, ctx) and not predicted_redundant(
                prim, op, cfg, archive.nodes[i].mode, solved_for_pair(archive, prim, solved, cfg.pair)):
            break
    return i, op, cfg, ctx


# ------------------------------------------------------------------ executor test
def _exec_test(task: dict) -> dict:
    """The learning's executor (the Hand-of-God motor along the graph, bep.motor.hog_motor.run_primitive_hop, as
    play runs calls) from one fresh start state, on the field of the archive saved at ``task['archive']``;
    success = it reports done."""
    prim = _W["prim"]
    prim.set_metric(task["metric"])
    if _W.get("field_key") != task["key"]:
        _W["field"], _W["field_key"] = ModeField(Archive.load(task["archive"]), prim, learning=True), task["key"]
    rng = np.random.default_rng(task["seed"])
    _, env, _, _ = prim.sample_start(rng)
    target = None if prim.kind != "object-object" else "B"
    out = run_primitive_hop(env, prim.R, prim, _W["field"], "A", target, timeout=task["timeout"])
    return {"exec": out["result"] == "done", "time": out["time"], "state": env.get_state(),
            "ctrl": env.data.ctrl.copy()}


def executor_rate(pool, archive_path: Path, key: int, n: int, seed: int, metric: dict) -> float:
    """Success rate of the executor from n fresh start states (one per worker task)."""
    tasks = [dict(archive=str(archive_path), key=key, seed=task_seed(seed, i), timeout=EXPLORE["exec_timeout_s"],
                  metric=metric) for i in range(n)]
    return float(np.mean([r["exec"] for r in pool.map(_exec_test, tasks)]))


def significantly_better(rate: float, prev: float, n: int) -> bool:
    """One-sided two-proportion z-test (n trials each) that rate > prev."""
    p = 0.5 * (rate + prev)
    se = np.sqrt(max(p * (1 - p), 1e-9) * 2.0 / n)
    return (rate - prev) / se > EXPLORE["exec_z"]


class StopRule:
    """Stop once the compacted field has converged: its values at a fixed set of start states and at a frozen
    sample of graph states from every mode change by less than stop_tol (median relative change) for
    stop_window consecutive checks, and the start states are covered (or their coverage has plateaued)."""

    def __init__(self, prim, n: int = 64):
        rng = np.random.default_rng(123)
        starts = [prim.sample_start(rng) for _ in range(n)]
        self.Z = np.array([s[2] for s in starts])
        self.M = np.array([int(s[3]) for s in starts])
        self.n_start = n
        self.frozen = False
        self.prev, self.calm = None, 0
        self.capped_hist: list[float] = []

    def _freeze(self, archive, rng):
        """Add a fixed sample of non-goal graph states, balanced over modes (taken once, half-way to min_rounds)."""
        nodes = [nd for nd in archive.nodes if np.isfinite(nd.D) and not nd.goal]
        by_mode: dict = defaultdict(list)
        for nd in nodes:
            by_mode[nd.mode].append(nd)
        per = max(1, EXPLORE["stop_sample"] // max(1, len(by_mode)))
        pick = [nd for lst in by_mode.values() for nd in (rng.choice(lst, min(per, len(lst)), replace=False))]
        if pick:
            self.Z = np.vstack([self.Z, np.array([nd.z for nd in pick])])
            self.M = np.concatenate([self.M, np.array([nd.mode for nd in pick])])
        self.frozen = True

    def update(self, field: ModeField, archive, rounds: int) -> dict:
        if not self.frozen and rounds >= EXPLORE["min_rounds"] // 2:
            self._freeze(archive, np.random.default_rng(321))
            self.prev = None
        v = field.value(self.Z, self.M)
        s = v[:self.n_start]
        # a start is covered when its own mode has been explored and its value is below the cap
        uncovered = (s >= field.cap - 1e-6) | ~field.explored(self.M[:self.n_start])
        out = {"start_phi_median": round(float(np.median(s)), 2),
               "start_capped_frac": round(float(np.mean(uncovered)), 3)}
        if self.prev is not None:
            change = float(np.median(np.abs(v - self.prev) / np.maximum(self.prev, 0.5)))
            self.calm = self.calm + 1 if change < EXPLORE["stop_tol"] else 0
            out["field_change"] = round(change, 4)
        self.prev = v
        out["calm_checks"] = self.calm
        self.capped = out["start_capped_frac"]
        self.capped_hist.append(self.capped)
        out["coverage_plateau"] = self.plateau()
        return out

    def plateau(self) -> bool:
        k = EXPLORE["coverage_plateau"]
        h = self.capped_hist
        return len(h) > k and min(h[-k:]) >= min(h[:-k]) - 1e-9

    def converged(self, rounds: int) -> bool:
        """Calm field AND the start states covered, or their coverage no longer improving."""
        return rounds >= EXPLORE["min_rounds"] and self.calm >= EXPLORE["stop_window"] \
            and (self.capped <= EXPLORE["uncovered_tol"] or self.plateau())


# ------------------------------------------------------------------ main loop
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relations", default="runs/pair_cubes/relations")
    ap.add_argument("--probes", default="runs/pair_cubes/probes.npz")
    ap.add_argument("--target", default="r0")
    ap.add_argument("--rounds", type=int, default=300, help="maximum rounds (a round = workers * per_worker results)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-worker", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--no-stop", action="store_true", help="run all rounds (disable the convergence stop)")
    ap.add_argument("--clips", type=int, default=0, help="rendered runs of the final primitive (runs/<scene>/clips)")
    ap.add_argument("--chain", default="", help="earlier primitives seeding the start states: target=dir,...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=0, help="seed for comparing executors across exploration runs")
    ap.add_argument("--out", default="runs/pair_cubes/prim_r0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_body("float")
    rng = np.random.default_rng(args.seed)
    prim = DiscoveredPrimitive(RelationSet.load(args.relations), args.target, probes=args.probes)
    _with_chain(prim, args.chain)
    archive = Archive()
    steps = dict(prim.steps)
    seed_goals(archive, prim, rng)
    log = open(out / "log.jsonl", "w")
    stats = defaultdict(Counter)
    window = defaultdict(list)
    arms = Arms(prim)
    stop = StopRule(prim)
    n_round = args.workers * args.per_worker
    done = queue.Queue()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                         initargs=(args.relations, args.probes, args.target, args.chain))
    n_sub = [0]
    metric: dict = {}
    cfgs: dict = {}                                     # node -> configuration (the Hand-of-God motor's target)

    def node_cfg(i):
        if i not in cfgs:
            cfgs[i] = prim.node_cfg(archive.nodes[i])
        return cfgs[i]

    def submit():
        """Asynchronous queue: a new candidate as soon as a worker is free, built from the current archive."""
        solved = archive.solved_arrays()
        prim.steps.update(steps)
        base = dict(max_steps=args.max_steps, steps=dict(steps), metric=metric,
                    seed=task_seed(args.seed, n_sub[0]))
        n_sub[0] += 1
        if archive.edges and rng.random() < EXPLORE["revalidate_p"]:
            w = 1.0 / np.array(archive.e_tries, float)
            k = int(rng.choice(len(archive.edges), p=w / w.sum()))
            i, j, _ = archive.edges[k]
            ni, nj = archive.nodes[i], archive.nodes[j]
            if np.isfinite(nj.D):
                aim = ni.parent if ni.parent >= 0 else j        # the motion the edge was validated with
                task = dict(base, solved=solved_for_pair(archive, prim, solved, ctx_pair(ni.B_pose)),
                            edge=k, parent=i, op="revalidate", arm=None, state=ni.state, ctrl=ni.ctrl,
                            ctx=ni.B_pose, z_target=archive.nodes[aim].z, m_target=archive.nodes[aim].mode,
                            D_max=nj.D, cfg_target=prim.retarget(node_cfg(aim), ctx_pair(ni.B_pose)))
                n_inflight[0] += 1
                pool.apply_async(_test, (task,), callback=done.put, error_callback=lambda e: done.put({"error": repr(e)}))
                return
        arm = arms.pick(rng)
        i, op, cfg, ctx = proposal(arm, prim, rng, archive, solved)
        nd = archive.nodes[i]
        nd.tries += 1
        task = dict(base, solved=solved_for_pair(archive, prim, solved, cfg.pair),
                    cfg=cfg, ctx=ctx, parent=i, op=op, arm=arm, z_target=nd.z, m_target=nd.mode,
                    D_max=np.inf, q_hint=nd.state[1:8], cfg_target=prim.retarget(node_cfg(i), cfg.pair))
        n_inflight[0] += 1
        pool.apply_async(_test, (task,), callback=done.put, error_callback=lambda e: done.put({"error": repr(e)}))

    t0 = time.time()
    n_done, rnd, sim_steps = 0, 0, 0
    n_inflight, exec_hist = [0], []
    best_archive, best_metric, best_steps = None, None, None
    best_rate, best_round = -1.0, 0
    reval = Counter()
    try:
        for _ in range(args.workers * EXPLORE["inflight_per_worker"]):
            submit()
        while rnd < args.rounds:
            res = done.get()
            n_inflight[0] -= 1
            if "error" in res:
                raise RuntimeError(res["error"])
            st = res["status"]
            sim_steps += res.get("sim_steps", 0)
            dt = res.get("dt", 0.01)
            if res.get("edge") is not None:                          # an edge re-validation
                archive.record_validation(res["edge"], st == "success", res.get("cost"))
                reval[st == "success"] += 1
                archive.dijkstra()
            else:
                stats[res["op"]][st] += 1
                base_op = res["op"].rstrip("*").split(":")[0]
                if st == "success" or st.startswith("fail:"):
                    window[base_op].append(st == "success")
                gain = 0.0
                if st == "goal":
                    n_goal = sum(nd.goal for nd in archive.nodes)
                    if n_goal < max(EXPLORE["n_goal_seeds"], EXPLORE["goal_share"] * (len(archive.nodes) - n_goal)):
                        archive.add_node(Node(res["z0"], int(res["m0"]), res["state"], res["ctrl"], res["B_pose"],
                                              res["op"], res["parent"], goal=True, D=0.0))
                        gain = 1.0
                elif st == "success":
                    reached = res["reached"]
                    if reached == -2:
                        pair = ctx_pair(res["B_pose"])
                        reached = next((k for k, nd in enumerate(archive.nodes) if nd.goal and ctx_pair(nd.B_pose) == pair), -1)
                    if reached >= 0:
                        k = archive.add_node(Node(res["z0"], int(res["m0"]), res["state"], res["ctrl"], res["B_pose"],
                                                  res["op"], res["parent"]))
                        archive.add_transition(k, reached, res["cost"], res["traj_z"], res["traj_m"], dt=res["dt"],
                                               stride=max(1, int(round(0.04 / res["dt"]))))
                        # This is the only edge out of the new node, and no existing node points to it.
                        archive.nodes[k].D = archive.nodes[reached].D + archive.edge_cost(len(archive.edges) - 1)
                        gain = 1.0
                arms.update(res["arm"], gain, res.get("sim_steps", 0) * dt + EXPLORE["realize_cost_s"])
            n_done += 1
            if n_done % n_round:
                submit()
                continue
            adapt_steps(steps, window)
            modes = Counter(int(nd.mode) for nd in archive.nodes if np.isfinite(nd.D))
            D = np.array([nd.D for nd in archive.nodes])
            entry = {"round": rnd, "time_s": round(time.time() - t0, 1), "nodes": len(archive.nodes),
                     "edges": len(archive.edges), "samples": len(archive.sZ), "tested": n_done,
                     "sim_steps": sim_steps, "solved_modes": {str(m): c for m, c in sorted(modes.items())},
                     "maxD": round(float(D[np.isfinite(D)].max()), 2),
                     "steps": {k: round(v, 4) for k, v in steps.items()}, "arms": arms.summary(),
                     "revalidated": {"ok": reval[True], "failed": reval[False]},
                     "ops": {op: dict(c) for op, c in stats.items()}}
            stop_now = False
            if rnd % EXPLORE["stop_every"] == EXPLORE["stop_every"] - 1 or rnd == args.rounds - 1:
                field = ModeField(archive, prim, learning=True)
                metric = learned_metric(field, prim)
                prim.set_metric(metric)
                entry["coverage"] = stop.update(field, archive, rnd + 1)
                entry["kappas"] = {str(m): round(v, 2) for m, v in field.kappas.items()}
                entry["field_samples"] = field.n_kept
                archive.save(out / "archive.npz")
                (out / "metric.json").write_text(json.dumps({str(m): w.tolist() for m, w in metric.items()}))
                stop_now = not args.no_stop and stop.converged(rnd + 1)
                entry["converged"] = stop_now
                if stop_now:             # the field converged: does the real executor still improve?
                    while n_inflight[0] > 0:
                        done.get()                               # let in-flight validations finish (dropped)
                        n_inflight[0] -= 1
                    n = EXPLORE["exec_tests"]
                    rate = executor_rate(pool, out / "archive.npz", rnd, n, args.eval_seed + 1000, metric)
                    if rate > best_rate:
                        best_archive, best_metric, best_steps = deepcopy(archive), metric, dict(steps)
                        best_rate, best_round = rate, rnd + 1
                    exec_hist.append({"round": rnd + 1, "success": rate})
                    entry["executor"] = rate
                    improving = len(exec_hist) == 1 or significantly_better(rate, exec_hist[-2]["success"], n)
                    stop_now = rate >= 1.0 - 1e-9 or not improving
                    if not stop_now:
                        stop.calm = 0                        # explore on until the field settles again
                        for _ in range(args.workers * EXPLORE["inflight_per_worker"] - 1):
                            submit()
            print(json.dumps(entry), flush=True)
            cov = entry.get("coverage", {})
            progress(rnd + 1, args.rounds, f"{entry['nodes']} nodes, {entry['edges']} edges"
                     + (f", start coverage {100 * (1 - cov['start_capped_frac']):.0f}%" if cov else "")
                     + (f", executor {100 * entry['executor']:.0f}%" if "executor" in entry else ""))
            log.write(json.dumps(entry) + "\n")
            log.flush()
            rnd += 1
            if stop_now:
                break
            submit()
        # the executor from fresh starts one last time: its score, and the end states of its successful calls
        # (primitives chained after this one start there: bep.explore.primitive.DiscoveredPrimitive.set_chain)
        if best_archive is not None and best_round != rnd:
            if not exec_hist or exec_hist[-1]["round"] != rnd:
                archive.save(out / "archive.npz")
                rate = executor_rate(pool, out / "archive.npz", rnd, EXPLORE["exec_tests"], args.eval_seed + 1000, metric)
                exec_hist.append({"round": rnd, "success": rate})
                if rate > best_rate:
                    best_round = rnd
            if best_round != rnd:
                archive, metric, steps = best_archive, best_metric, best_steps
        if best_archive is None:
            best_round = rnd
        archive.save(out / "archive.npz")
        (out / "metric.json").write_text(json.dumps({str(m): np.asarray(w).tolist() for m, w in metric.items()}))
        res = pool.map(_exec_test, [dict(archive=str(out / "archive.npz"), key=-1, seed=task_seed(args.eval_seed + 777, i),
                                         timeout=EXPLORE["exec_timeout_s"], metric=metric)
                                    for i in range(EXPLORE["end_runs"])])
        ok = [r for r in res if r["exec"]]
        np.savez_compressed(out / "ends.npz", pair=np.asarray(prim.kinds), states=np.array([r["state"] for r in ok]),
                            ctrls=np.array([r["ctrl"] for r in ok]))
        exec_hist.append({"round": best_round, "success": float(np.mean([r["exec"] for r in res])), "final": True})
        progress(rnd, args.rounds, f"final executor {100 * exec_hist[-1]['success']:.0f}% "
                                   f"({len(ok)} end states kept)")
    finally:
        archive.save(out / "archive.npz")
        (out / "steps.json").write_text(json.dumps(steps))
        (out / "metric.json").write_text(json.dumps({str(m): np.asarray(w).tolist() for m, w in metric.items()}))
        pool.terminate()
    if exec_hist:                        # the primitive's measured quality (used to pick among duplicates)
        (out / "executor.json").write_text(json.dumps({"success": exec_hist[-1]["success"], "history": exec_hist}))
    if args.clips:
        record_clips(archive, prim, out.parent / "clips" / f"explore_{args.target}", args.clips)


def record_clips(archive: Archive, prim, path, n: int, seconds: float = 8.0):
    """The final primitive, run by the learning's executor (bep.motor.hog_motor.run_primitive_hop), from
    representative nodes of the graph (50th-90th percentile of cost-to-go), filmed with the wide scene camera;
    the first n successful runs are kept."""
    from ..world.clips import Recorder
    field = ModeField(archive, prim, learning=True)
    ok = sorted([k for k, nd in enumerate(archive.nodes) if np.isfinite(nd.D) and not nd.goal],
                key=lambda k: archive.nodes[k].D)
    far = [ok[int(q * (len(ok) - 1))] for q in np.linspace(0.5, 0.9, 4 * n)] if ok else []
    saved = 0
    for i, k in enumerate(far):                     # only runs that reach the goal are kept, n of them
        if saved >= n:
            break
        nd = archive.nodes[k]
        env = pair_world(*ctx_pair(nd.B_pose))
        restore(env, nd.state, nd.ctrl, nd.B_pose)
        target = None if prim.kind != "object-object" else "B"
        rec = Recorder(env, f"{path}_{saved}.mp4")
        out = run_primitive_hop(env, prim.R, prim, field, "A", target, timeout=seconds, on_step=lambda k: rec.frame())
        if out["result"] == "done":
            rec.save()
            saved += 1
        else:
            rec.renderer.close()


if __name__ == "__main__":
    main()
