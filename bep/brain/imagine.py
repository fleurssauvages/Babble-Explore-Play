"""Imagine: an internal model of events, and goals chosen by imagining their outcome.

* The event predictor (a cerebellum-like internal model, Wolpert, Miall & Kawato 1998): from the settled
  state before a babbling command (contact flags, X relative to Y, the pinch point relative to X) and the
  command's class, the probability of each kept event key (bep.babble.events).  It is fitted at sleep.  During
  an epoch, a babble block's competence is how well it predicted the block's events (exp of minus the mean
  cross-entropy per event): babbling where effects are still surprising has learning progress.
* The imagine activity: in a random layout of the scene, the relation graphs reachable in up to three calls
  are imagined with the world model of the effect models; the goal is the one whose imagined plan has a
  success probability nearest 1/2 (intermediate difficulty: what is neither known nor out of reach); the
  loop's brain (bep.brain.brain) plans it and the calls are executed and recorded like play's.  The
  activity's competence is 1 - |imagined success - real success|: how well imagination predicts reality.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ..config import PLAY, task_seed
from ..babble.events import key_table

CLASSES = ("move-up", "move-down", "move-flat", "move-none", "close", "open", "retract")


# ------------------------------------------------------------------ the event predictor
def predictor_inputs(rows) -> np.ndarray:
    from ..world.objects import VIEW_S
    from ..world.relations import cc_features, hc_features
    f0 = np.asarray(rows["f0"])
    onehot = [np.eye(3)[np.clip(f0[:, 0], 0, 2)], f0[:, 1:]]
    cmd = np.array([[c == k for k in CLASSES] for c in rows["cmdc"]], float)
    v = rows["v_pre"]
    p = np.clip(cc_features(v, VIEW_S, "AB")["p"], -3, 3)
    h = np.clip(hc_features(v, VIEW_S, rotation=False)["p"], -3, 3)
    return np.concatenate([*onehot, cmd, p, h], axis=1).astype(np.float32)


class EventPredictor:
    def __init__(self):
        self.keys: list[str] = []
        self.st = None

    def fit(self, ev, steps: int = 300) -> dict:
        import torch
        torch.set_num_threads(8)
        R = ev.rows
        self.keys = [t["key"] for t in key_table(ev) if t["kept"]]
        if not self.keys:
            return {"fitted": False}
        X = predictor_inputs(R)
        Y = np.array([[k in fk for k in self.keys] for fk in R["factors"]], np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3
        Xt, Yt = torch.tensor((X - self.mu) / self.sd), torch.tensor(Y)
        d, h, K = X.shape[1], 64, len(self.keys)
        net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, K))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        rng = np.random.default_rng(0)
        for _ in range(steps):
            i = torch.tensor(rng.integers(len(Y), size=min(4096, len(Y))))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[i]), Yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.st = [p.detach().numpy() for p in net.parameters()]
        return {"fitted": True, "keys": len(self.keys), "surprise": round(self.surprise(R), 4)}

    def logits(self, X):
        W1, b1, W2, b2 = self.st
        h = np.maximum(((X - self.mu) / self.sd) @ W1.T + b1, 0)
        return h @ W2.T + b2

    def surprise(self, rows, start: int = 0) -> float:
        """Mean cross-entropy per event (summed over the keys) of rows[start:]."""
        sub = {k: v[start:] for k, v in rows.items()}
        if not len(sub["f0"]):
            return 0.0
        z = self.logits(predictor_inputs(sub))
        Y = np.array([[k in fk for k in self.keys] for fk in sub["factors"]], float)
        bce = np.logaddexp(0, z) - Y * z
        return float(bce.sum(1).mean())


# ------------------------------------------------------------------ imagined goals (worker side)
def _reachable(M, s0, poses0, depth: int = 3) -> dict:
    """Relation graphs reachable in up to ``depth`` predicted successful calls: state -> (success probability
    of the path, path length)."""
    out, frontier = {}, [(s0, poses0, 1.0, 0)]
    seen = {s0}
    while frontier:
        s, poses, p, n = frontier.pop(0)
        if n >= depth:
            continue
        for k in M.keys:
            o = M.outcomes(s, poses, k)
            if not o:
                continue
            s2, q = o[0][1], p * o[0][0]
            if s2 not in seen:
                seen.add(s2)
                out[s2] = (q, n + 1)
                frontier.append((s2, o[0][3], q, n + 1))
    return out


def _setup(task: dict):
    """The scene in the task's random layout, the effect models and a brain."""
    from ..brain.goal import Goal
    from ..explore.library import bindings
    from ..play.play import _library, _models
    from ..world.graph import as_tuples, object_poses, relation_graph
    from ..world.objects import apply_layout
    from ..world.scenes import random_layout, table_spot_fn
    from .brain import LoopBrain
    env, R, lib, _ = _library(task["run"], task["version"], task["scene"])
    rng = np.random.default_rng(task["seed"])
    history = task["history"]
    models = _models(R, lib, history, env)
    if not all(t in models for t in lib):
        return None
    layout, joints = random_layout(env, rng)
    apply_layout(env, layout, joints=joints)
    actions = [(t, x, y) for t in lib for x, y in bindings(env, lib[t][0])]
    hc = [r.name for r in R.hc]
    brain = LoopBrain(models, actions, env.cubes, Goal(set(), hc), spot_fn=table_spot_fn(env), seed=task["seed"],
                      env=env, lib=lib, R=R, imagine=False)
    s0 = brain.model.abstract(as_tuples(relation_graph(env, R)))
    return env, R, lib, brain, s0, rng, object_poses(env)


def _choose_goal(brain, s0, poses, R, rng):
    """The resting arrangement reachable in up to three imagined calls whose imagined success is nearest 1/2."""
    reach = _reachable(brain.model, s0, poses)
    cc, hc = {r.name for r in R.cc}, {r.name for r in R.hc}
    cands = [(abs(p - 0.5), n, s, p) for s, (p, n) in reach.items()
             if any(i[0] in cc for i in s) and not any(i[0] in hc for i in s)]          # resting arrangements
    if not cands:
        return None
    _, n, goal_s, pred = min(cands, key=lambda c: (c[0], c[1], rng.random()))
    return sorted([list(i) for i in goal_s if i[0] in cc]), n, float(pred)


def imagine_episode(task: dict) -> dict:
    """Pursue an imagined goal with the brain, executing and recording every call like play."""
    from ..brain.goal import Goal
    from ..play.play import call_and_record
    from ..world.geometry import Spot
    from ..world.graph import as_tuples, object_poses, relation_graph
    got = _setup(task)
    empty = {"recs": [], "starts": [], "pred": None, "real": None}
    if got is None:
        return empty
    env, R, lib, brain, s0, rng, poses = got
    g = _choose_goal(brain, s0, poses, R, rng)
    if g is None:
        return empty
    req_l, n, pred = g
    req = set(map(tuple, req_l))
    cc, hc = {r.name for r in R.cc}, [r.name for r in R.hc]
    forbid = {(q, u, v) for q in cc for u in env.cubes for v in env.cubes if u != v and (q, u, v) not in req}
    goal = Goal(req, hc, True, forbid)
    brain.set_goal(goal)
    recs, starts = [], []
    for step in range(n + 2):
        rels = as_tuples(relation_graph(env, R))
        if goal.unmet(rels) == 0:
            break
        key, spot, info = brain.decide(rels, object_poses(env))
        if key is None:
            break
        t, x, tt = key
        target = Spot(float(spot[0]), float(spot[1]), 0.0) if tt == "spot" else tt
        rec, start = call_and_record(env, R, lib, t, x, target)
        rec.update(imagined_goal=sorted(map(list, req)), episode=task["seed"], step=step)
        recs.append(rec)
        starts.append(start)
        brain.observe(rels, key, rec["time"], as_tuples(relation_graph(env, R)), object_poses(env), rec["result"])
    real = float(goal.unmet(as_tuples(relation_graph(env, R))) == 0)
    return {"recs": recs, "starts": starts, "pred": pred, "real": real, "goal": sorted(map(list, req))}


# ------------------------------------------------------------------ the loop extension
class ImagineExtension:
    def __init__(self):
        self.predictor = EventPredictor()
        self.n = 0

    def sleep(self, loop) -> dict:
        first = self.predictor.st is None
        info = self.predictor.fit(loop.events) if len(loop.events) else {"fitted": False}
        if first and info.get("fitted"):
            loop.arbiter.hist["babble"] = []          # babble's competence changes measure: start its history again
        return {"event_predictor": info}

    def babble_competence(self, ev, start: int):
        if self.predictor.st is None:
            return None
        return float(np.exp(-self.predictor.surprise(ev.rows, start)))

    def _ready(self, loop) -> bool:
        """Imagination needs a model of every schema: each has an archive and enough play calls to be fitted."""
        calls = Counter(r["primitive"] for r in loop.player.history)
        return bool(loop.prims) and all((loop.out / f"prim_{p['target']}" / "archive.npz").exists()
                                        and calls[p["target"]] >= PLAY["min_fit"] for p in loop.prims)

    def activities(self, loop) -> list:
        return ["imagine"] if self._ready(loop) else []

    def _record(self, loop, outs) -> list:
        recs = [r for o in outs for r in o["recs"]]
        with open(loop.player.path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        loop.player.history += recs
        fails: dict = {}
        for o in outs:
            for s in o["starts"]:
                if not s["ok"]:
                    fails.setdefault(s["primitive"], []).append(s)
        for t, lst in fails.items():
            ex = loop.explorers.get(t)
            if ex is not None:
                ex.add_failures([ex.prim.cfg_from_view(s["view"], s["ctrl"], s["pair"]) for s in lst], lst)
        return recs

    def run(self, loop, name: str):
        loop._write_library()
        tasks = [{"run": str(loop.out), "version": loop.n_blocks, "scene": loop.scene_name,
                  "history": list(loop.player.history[-PLAY["history_cap"]:]),
                  "seed": task_seed(loop.seed + 11, self.n * 1000 + k)}
                 for k in range(loop.pool.n)]
        self.n += 1
        outs = loop.pool.map("bep.brain.imagine:imagine_episode", tasks)
        recs = self._record(loop, outs)
        scored = [o for o in outs if o["pred"] is not None]
        err = float(np.mean([abs(o["pred"] - o["real"]) for o in scored])) if scored else 1.0
        st = {"episodes": len(scored), "calls": len(recs),
              "imagined_success": round(float(np.mean([o["pred"] for o in scored])), 3) if scored else None,
              "real_success": round(float(np.mean([o["real"] for o in scored])), 3) if scored else None,
              "goals": dict(Counter(len(o["goal"]) for o in scored))}
        return st, (1.0 - err) if scored else None              # nothing imagined: not applicable for now
