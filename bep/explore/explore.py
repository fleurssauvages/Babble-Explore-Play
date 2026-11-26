"""Explore: reverse Go-Explore toward one schema's relation, run in blocks the arbiter schedules.

Loop: choose an arm -> propose an earlier configuration from a solved node -> Hand of God -> forward
validation into any solved epsilon-ball -> archive -> Dijkstra.  The forward validation moves the floating
hand with the Hand-of-God motor (bep.motor.hog_motor).

* goal = the target relation under its grasp or release conditions; seeds are reproduced probe states; a
  validated candidate already in the goal mode extends the goal set,
* contact modes = bitmasks of discovered relations; states in object units, so nodes of every pair of shapes
  share one graph,
* arms: the backward operators (move X, move the hand, turn the hand, change the grip, switch a relation on,
  move a joint, swap shapes) and "dir" (step the solved node closest to a start configuration toward it),
  picked by a bandit on their yield (new solved nodes per simulated second), with a floor,
* proposals are resampled until the Hand of God's geometric pre-check passes and their relational state is
  not predicted to land on a solved node; step lengths adapt so that validation succeeds in a target band;
  edges are re-validated now and then (Dijkstra prices them by expected time c / p),
* trajectory states that are physical transients (X moving with no relation holding) are stored with mode
  -1 and ignored by the field.

``Explorer`` makes the search resumable block by block and closes two loops:

* play's failed calls become start states: the directed arm aims, LOOP['fail_p'] of the time, at the
  configuration a failed call started from (first return, then explore), and half of the executor tests
  after a block start there (a schema variant, bep.play.schemas: all of them, from its own context);
* the schema's competence after a block (the executor's success on those tests) is the arbiter's measure of
  its learning progress (bep.arbiter).

Workers keep the schema's primitive between tasks (bep.workers.CACHE), rebuilt when its relations or the end
states it is chained to change.
"""
from __future__ import annotations

import json
import os
import queue
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from ..config import EXPLORE, LOOP, task_seed
from ..world.objects import pair_world
from ..world.relations import HC, RelationSet
from .archive import Archive, Node
from .field import ModeField, learned_metric
from ..motor.hog_motor import hog_validate
from .primitive import DiscoveredPrimitive, ctx_pair, load_end_states, restore

# ------------------------------------------------------------------ the search (worker side and proposals)
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


def _test(task: dict) -> dict:
    """Realise a backward candidate and validate it forward with the Hand-of-God motor, or (``edge`` set)
    re-run a validated edge from its source node's state."""
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
    holds = prim.R.holds_all(s2, VIEW_S)[prim.goal_rel]
    if prim.goal_requires_lift:
        holds &= (prim.probes["ev_kind"] == 2) & prim.probes["ev_follow"]
    if prim.kind == "object-state":
        holds = holds & (s2[:, VIEW_S.A_art] > 0)
    cand = np.nonzero(holds)[0]
    rng.shuffle(cand)
    if EXPLORE["typical_seeds"] and len(cand):       # the most typical goal states first (prototypes)
        cand = cand[np.argsort(_atypicality(prim, s2[cand]), kind="stable")]
    ctrl0 = np.zeros(pair_world(*prim.kinds).model.nu)
    ctrl0[-1] = 255.0                                  # the gripper open
    per_pair = Counter()
    share = max(1, int(np.ceil(limit / len(prim.training_pairs))))
    for i in cand:
        if len(archive.nodes) >= limit:
            break
        pair = source = pairs[i]
        if prim.y_spot:                    # a table spot as Y: the observed state re-expressed in the spot's units
            if pair[0] != prim.kinds[0]:
                continue
            pair = prim.kinds
        elif pair not in prim.training_pairs and (prim.kind == "object-object" or pair[0] != prim.kinds[0]):
            continue
        if pair not in prim.training_pairs:
            pair = source = prim.kinds
        if per_pair[pair] >= share:
            continue
        cfg = prim.retarget(prim.cfg_from_view(s2[i], ctrl0, source), pair)
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


def _atypicality(prim, views) -> np.ndarray:
    """How far each observed goal state is from the median of them, in the relation's own features (the
    grasp's contact points for a hand-object relation, X's place relative to Y otherwise), each feature in units
    of its spread."""
    from ..world.objects import VIEW_S
    from ..world.relations import cc_features, hc_features, member_vector
    if prim.kind == HC:
        f = hc_features(views, VIEW_S, rotation=False, body=True)
        X = f["body"] if prim.rel_target.body_members is not None else member_vector(HC, f)
    elif prim.kind == "object-object":
        X = cc_features(views, VIEW_S, "AB")["p"]
    else:
        return np.zeros(len(views))
    med = np.median(X, axis=0)
    spread = np.median(np.abs(X - med), axis=0) + 1e-6
    return np.abs((X - med) / spread).sum(1)


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
        hand = True
        ok = {"XY": not prim.x_is_part, "EX": hand, "rot": hand, "grip": hand, "rel": bool(prim.pools) and hand,
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
def call_target(prim, env):
    """The target of a call in the primitive's pair world: body B, its table spot (Y = "spot"), or none."""
    if prim.kind != "object-object":
        return None
    return env.ref if prim.y_spot else "B"



# ------------------------------------------------------------------ worker side
def _sig(paths) -> tuple:
    return tuple((str(p), os.path.getmtime(p) if os.path.exists(p) else 0.0) for p in paths)


def worker_prim(spec: dict):
    """The schema's primitive in a worker (cached; rebuilt when its relations or chained end states change)."""
    from ..workers import CACHE
    folder = Path(spec["folder"])
    chain_files = [Path(p) / "ends.npz" for p in spec["chain"].values()]
    key = ("prim", str(folder), _sig([folder / "relations.json", *chain_files]))
    if key not in CACHE:
        for k in [k for k in CACHE if isinstance(k, tuple) and k[:2] == ("prim", str(folder))]:
            del CACHE[k]
        prim = DiscoveredPrimitive(RelationSet.load(folder / "relations"), spec["target"], kinds=tuple(spec["kinds"]),
                                   probes=spec["probes"])
        chain = ",".join(f"{t}={p}" for t, p in spec["chain"].items() if (Path(p) / "ends.npz").exists())
        _with_chain(prim, chain)
        CACHE[key] = prim
    return CACHE[key]


def test(task: dict) -> dict:
    """``_test`` with the schema's cached primitive."""
    _W["prim"] = worker_prim(task["spec"])
    return _test(task)


def _worker_field(prim, path: str, metric):
    from ..workers import CACHE
    key = ("field", path, os.path.getmtime(path))
    if CACHE.get("field_key") != key:
        prim.set_metric(metric)
        CACHE["field"], CACHE["field_key"] = ModeField(Archive.load(path), prim, learning=True), key
    return CACHE["field"]


def exec_test(task: dict) -> dict:
    """The learning's executor (bep.motor.hog_motor.run_primitive_hop) from one start: a fresh start of the
    schema (``start`` None) or a configuration a failed play call started from."""
    from ..motor.hog_motor import run_primitive_hop
    prim = worker_prim(task["spec"])
    prim.set_metric(task["metric"])
    field = _worker_field(prim, task["archive"], task["metric"])
    rng = np.random.default_rng(task["seed"])
    if task.get("start") is None:
        _, env, _, _ = prim.sample_start(rng)
    else:
        cfg = task["start"]
        r = prim.realize(None, cfg, prim.sample_ctx(rng, cfg.pair))
        if isinstance(r, str):
            return {"exec": None}
        env = pair_world(*cfg.pair)                   # realize left the start in this (cached) world
    out = run_primitive_hop(env, prim.R, prim, field, "A", call_target(prim, env), timeout=task["timeout"])
    ok = out["result"] == "done"
    return {"exec": ok, "time": out["time"], "from_failure": task.get("start") is not None,
            "state": env.get_state() if ok else None, "ctrl": env.data.ctrl.copy() if ok else None,
            "pair": tuple(env.kinds[n] for n in ("A", "B") if n in env.kinds)}


# ------------------------------------------------------------------ main process
class Explorer:
    """One schema's exploration, resumable block by block."""

    def __init__(self, run: Path, spec: dict, chain: dict, seed: int = 0):
        self.run, self.spec_in, self.target = Path(run), dict(spec), spec["target"]
        self.folder = self.run / f"prim_{self.target}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chain = dict(chain)                         # earlier schemas: target -> folder (their ends.npz)
        self.rng = np.random.default_rng(task_seed(seed, hash(self.target) % 10007))
        self.n_sub = 0
        self.fail_starts: list = []
        self.history: list = []
        self.stats = defaultdict(Counter)
        self.window = defaultdict(list)
        self.steps = dict(EXPLORE["step0"])
        self.metric: dict = {}
        self.load()

    # --------------------------------------------------------- state
    def task_spec(self) -> dict:
        return {"folder": str(self.folder), "target": self.target, "kinds": list(self.spec_in["kinds"]),
                "probes": str(self.run / "probes.npz"), "chain": {t: str(p) for t, p in self.chain.items()}}

    def load(self) -> None:
        """The primitive (main process) and the archive (resumed from disk, or seeded from goal states)."""
        self.prim = worker_prim(self.task_spec())
        self.prim.steps.update(self.steps)
        self._wrap_starts()
        path = self.folder / "archive.npz"
        if path.exists():
            self.archive = Archive.load(path)
            mt = self.folder / "metric.json"
            if mt.exists():
                self.metric = {int(m): w for m, w in json.loads(mt.read_text()).items()}
                self.prim.set_metric(self.metric)
            st = self.folder / "steps.json"
            if st.exists():
                self.steps.update(json.loads(st.read_text()))
        else:
            self.archive = Archive()
            seed_goals(self.archive, self.prim, self.rng)
        self.arms = Arms(self.prim)
        self.cfgs: dict = {}
        ex = self.folder / "executor.json"
        if ex.exists():
            self.history = json.loads(ex.read_text()).get("history", [])

    def _wrap_starts(self) -> None:
        """Directed proposals aim at a failed call's start configuration LOOP['fail_p'] of the time."""
        prim, base = self.prim, self.prim.sample_start
        explorer = self

        def sample_start(rng, chain=None):
            pool = explorer.focus_starts()
            if pool and rng.random() < LOOP["fail_p"]:
                cfg = pool[int(rng.integers(len(pool)))]
                return cfg, None, prim.z_of_cfg(cfg), None
            return base(rng, chain)
        prim.sample_start = sample_start

    def focus_starts(self) -> list:
        """The failure starts to aim at: those of the current focus (a schema variant's context), else all."""
        focus = getattr(self, "focus", None)
        if focus is None:
            return self.fail_starts
        infos = getattr(self, "fail_infos", [None] * len(self.fail_starts))
        sel = [c for c, i in zip(self.fail_starts, infos) if focus(i)]
        return sel or self.fail_starts

    def add_failures(self, cfgs: list, infos: list | None = None) -> None:
        """Failed calls' start configurations, re-expressed for the schema's own pair of kinds (``infos``: the
        calls' contexts, kept alongside so a schema variant can practise on its own context's failures)."""
        out = []
        for cfg in cfgs:
            c = self.prim.retarget(cfg, tuple(self.prim.kinds))
            c.pair = tuple(self.prim.kinds)
            out.append(c)
        infos = list(infos) if infos is not None else [None] * len(out)
        self.fail_starts = (self.fail_starts + out)[-LOOP["fail_starts"]:]
        self.fail_infos = (getattr(self, "fail_infos", []) + infos)[-LOOP["fail_starts"]:]

    def node_cfg(self, i: int):
        if i not in self.cfgs:
            self.cfgs[i] = self.prim.node_cfg(self.archive.nodes[i])
        return self.cfgs[i]

    def save(self) -> None:
        tmp = self.folder / "archive.tmp.npz"
        self.archive.save(tmp)
        tmp.replace(self.folder / "archive.npz")
        (self.folder / "metric.json").write_text(json.dumps({str(m): np.asarray(w).tolist() for m, w in self.metric.items()}))
        (self.folder / "steps.json").write_text(json.dumps(self.steps))
        succ = self.history[-1]["success"] if self.history else 0.0
        (self.folder / "executor.json").write_text(json.dumps({"success": succ, "history": self.history[-50:]}))

    # --------------------------------------------------------- one block
    def _task(self):
        archive, prim = self.archive, self.prim
        solved = archive.solved_arrays()
        prim.steps.update(self.steps)
        base = dict(max_steps=150, steps=dict(self.steps), metric=self.metric, seed=task_seed(17, self.n_sub),
                    spec=self.task_spec())
        self.n_sub += 1
        if archive.edges and self.rng.random() < EXPLORE["revalidate_p"]:
            w = 1.0 / np.array(archive.e_tries, float)
            k = int(self.rng.choice(len(archive.edges), p=w / w.sum()))
            i, j, _ = archive.edges[k]
            ni, nj = archive.nodes[i], archive.nodes[j]
            if np.isfinite(nj.D):
                aim = ni.parent if ni.parent >= 0 else j
                return dict(base, solved=solved_for_pair(archive, prim, solved, ctx_pair(ni.B_pose)), edge=k, parent=i,
                            op="revalidate", arm=None, state=ni.state, ctrl=ni.ctrl, ctx=ni.B_pose,
                            z_target=archive.nodes[aim].z, m_target=archive.nodes[aim].mode, D_max=nj.D,
                            cfg_target=prim.retarget(self.node_cfg(aim), ctx_pair(ni.B_pose)))
        arm = self.arms.pick(self.rng)
        i, op, cfg, ctx = proposal(arm, prim, self.rng, archive, solved)
        nd = archive.nodes[i]
        nd.tries += 1
        return dict(base, solved=solved_for_pair(archive, prim, solved, cfg.pair), cfg=cfg, ctx=ctx, parent=i, op=op,
                    arm=arm, z_target=nd.z, m_target=nd.mode, D_max=np.inf, q_hint=nd.state[1:8],
                    cfg_target=prim.retarget(self.node_cfg(i), cfg.pair))

    def _result(self, res: dict) -> None:
        archive = self.archive
        st = res["status"]
        if res.get("edge") is not None:
            archive.record_validation(res["edge"], st == "success", res.get("cost"))
            archive.dijkstra()
            return
        self.stats[res["op"]][st] += 1
        base_op = res["op"].rstrip("*").split(":")[0]
        if st == "success" or st.startswith("fail:"):
            w = self.window[base_op]
            w.append(st == "success")
            del w[:-EXPLORE["window"]]                # adapt_steps reads only the last window
        gain = 0.0
        if st == "goal":
            n_goal = sum(nd.goal for nd in archive.nodes)
            if n_goal < max(EXPLORE["n_goal_seeds"], EXPLORE["goal_share"] * (len(archive.nodes) - n_goal)):
                archive.add_node(Node(res["z0"], int(res["m0"]), res["state"], res["ctrl"], res["B_pose"], res["op"],
                                      res["parent"], goal=True, D=0.0))
                gain = 1.0
        elif st == "success":
            reached = res["reached"]
            if reached == -2:
                pair = ctx_pair(res["B_pose"])
                reached = next((k for k, nd in enumerate(archive.nodes) if nd.goal and ctx_pair(nd.B_pose) == pair), -1)
            if reached >= 0 and self._crowded(res["z0"], int(res["m0"])):
                reached = -1                          # beyond the node cap: too close to what the graph has
            if reached >= 0:
                k = archive.add_node(Node(res["z0"], int(res["m0"]), res["state"], res["ctrl"], res["B_pose"], res["op"],
                                          res["parent"]))
                archive.add_transition(k, reached, res["cost"], res["traj_z"], res["traj_m"], dt=res["dt"],
                                       stride=max(1, int(round(0.04 / res["dt"]))))
                archive.nodes[k].D = archive.nodes[reached].D + archive.edge_cost(len(archive.edges) - 1)
                gain = 1.0
        self.arms.update(res["arm"], gain, res.get("sim_steps", 0) * res.get("dt", 0.01) + EXPLORE["realize_cost_s"])

    def _crowded(self, z, mode: int) -> bool:
        """Beyond EXPLORE['node_cap'] nodes, a new node must lie sqrt(nodes / cap) times farther than the novelty
        radius from the solved nodes of its mode (the graph grows where it is sparse, not everywhere)."""
        cap = EXPLORE["node_cap"]
        n = len(self.archive.nodes)
        if not cap or n <= cap:
            return False
        same = [nd.z for nd in self.archive.nodes if nd.mode == mode and np.isfinite(nd.D)]
        if not same:
            return False
        d = self.prim.distance(z, np.array(same), mode).min()
        return bool(d < EXPLORE["novelty"] * self.prim.eps[mode] * np.sqrt(n / cap))

    def run_block(self, pool, rounds: int | None = None, focus=None) -> dict:
        """``rounds`` rounds of the asynchronous search (a round = one result per worker), then the field, the
        learned metric, the executor tests (half from failed play calls' starts) and the end states.  ``focus``:
        a predicate on failure contexts (a schema variant): directed proposals and tests use those failures."""
        self.focus = focus
        try:
            return self._run_block(pool, rounds)
        finally:
            self.focus = None

    def _run_block(self, pool, rounds: int | None = None) -> dict:
        rounds = rounds or LOOP["explore_rounds"]
        n0 = len(self.archive.nodes)
        total = rounds * pool.n
        done: queue.Queue = queue.Queue()
        inflight = 0

        def submit():
            nonlocal inflight
            pool.apply_async("bep.explore.explore:test", self._task(), done.put,
                             lambda e: done.put({"error": repr(e)}))
            inflight += 1
        for _ in range(min(total, pool.n * EXPLORE["inflight_per_worker"])):
            submit()
        submitted, processed, sim = inflight, 0, 0
        while processed < total:
            res = done.get()
            inflight -= 1
            if "error" in res:
                raise RuntimeError(res["error"])
            self._result(res)
            sim += res.get("sim_steps", 0)
            processed += 1
            if processed % pool.n == 0:
                adapt_steps(self.steps, self.window)
            if submitted < total:
                submit()
                submitted += 1
        field = ModeField(self.archive, self.prim, learning=True)
        self.metric = learned_metric(field, self.prim)
        self.prim.set_metric(self.metric)
        self.save()
        competence, ends = self.executor_tests(pool)
        self.history.append({"success": competence, "nodes": len(self.archive.nodes)})
        del self.history[:-50]                        # only the latest is read (executor.json keeps 50)
        self.save()
        self._save_ends(ends)
        return {"target": self.target, "nodes": len(self.archive.nodes), "new_nodes": len(self.archive.nodes) - n0,
                "competence": round(competence, 3), "sim_steps": sim, "failure_starts": len(self.fail_starts)}

    def executor_tests(self, pool) -> tuple[float, list]:
        n = LOOP["exec_tests"]
        starts = [None] * n
        pool_starts = self.focus_starts()
        if pool_starts:
            share = n if getattr(self, "focus", None) is not None else n // 2      # a variant: its own failures only
            for k in range(share):
                starts[k] = pool_starts[int(self.rng.integers(len(pool_starts)))]
        tasks = [dict(spec=self.task_spec(), archive=str(self.folder / "archive.npz"), metric=self.metric,
                      seed=task_seed(29, self.n_sub * 101 + k), timeout=EXPLORE["exec_timeout_s"], start=s)
                 for k, s in enumerate(starts)]
        self.n_sub += 1
        res = [r for r in pool.map("bep.explore.explore:exec_test", tasks) if r["exec"] is not None]
        rate = float(np.mean([r["exec"] for r in res])) if res else 0.0
        return rate, [r for r in res if r["exec"]]

    def _save_ends(self, ends: list) -> None:
        """Successful end states (chained schemas start there), the latest LOOP['ends_per_block'] x 4 kept."""
        path = self.folder / "ends.npz"
        old_s, old_c = [], []
        if path.exists():
            with np.load(path) as f:
                old_s, old_c = list(f["states"]), list(f["ctrls"])
        S = (old_s + [e["state"] for e in ends])[-4 * LOOP["ends_per_block"]:]
        C = (old_c + [e["ctrl"] for e in ends])[-4 * LOOP["ends_per_block"]:]
        if S:
            tmp = self.folder / "ends.tmp.npz"
            np.savez_compressed(tmp, pair=np.asarray(self.prim.kinds), states=np.array(S), ctrls=np.array(C))
            tmp.replace(path)
