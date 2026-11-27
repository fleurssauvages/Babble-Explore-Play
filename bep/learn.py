"""Learning: epochs of curiosity-driven activities (babble, explore, play, imagine), with sleep between them.

    python -m bep.learn --scene pair_cubes --epochs 4          # -> runs/pair_cubes/

Epoch k:
  * activities, until LOOP['epoch_s'] of wall time: the arbiter (bep.arbiter) picks babble, explore of a
    schema, play or imagine by learning progress; the first epoch babbles until the event store
    yields schemas;
  * sleep (between epochs only): the relations are re-read from every stored event (bep.babble.events), schemas
    are created, split or merged (bep.play.schemas), the cost-to-go map is refit (bep.explore.qmap), the archives are remapped
    to the new vocabulary, and the learning directory is snapshot in <out>/epoch_<k> (the format of the
    tasks: bep.brain.solve --learning <out>/epoch_<k> runs them on it).

Loops closed inside an epoch: play's failed calls become exploration start states; babble restarts from
states the schemas reached (their goal states) as well as from rare events.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .arbiter import Arbiter
from .babble.discovery import babble_block, read_vocabulary, scene_info
from .babble.events import Events
from .config import BABBLE, LOOP, task_seed
from .explore.explore import Explorer
from .explore.primitive import ctx_pair
from .play.play import Player
from .terminal import ProgressBar, ascii_bar
from .world.relations import RelationSet

SNAPSHOT = ("relations.json", "relations.npz", "relations_all.json", "relations_all.npz", "probes.npz",
            "primitives.json", "learning.json", "play.jsonl")


def epoch_summary(info: dict, elapsed: float) -> str:
    """One row per activity, using wall-time share and final mean arbiter progress."""
    seconds = defaultdict(float)
    blocks = Counter()
    for block in info["blocks"]:
        activity = block["activity"].partition(":")[0]
        seconds[activity] += block["wall_s"]
        blocks[activity] += 1

    curiosity = defaultdict(list)
    for name, stats in info["arbiter"].items():
        curiosity[name.partition(":")[0]].append(stats["progress"])

    activities = sorted(seconds.keys() | curiosity.keys())
    sleep = info["sleep"].get("sleep_s")
    if sleep is not None:
        seconds["sleep"] = sleep
        activities.append("sleep")
    overhead = elapsed - sum(seconds.values())
    if overhead >= 0.05:
        seconds["other"] = overhead
        activities.append("other")
    total = sum(seconds.values())

    width = max(map(len, ["Activity", *activities]))
    curiosity_bar_max = 0.30
    lines = [f"Epoch {info['epoch']}  {total:.1f}s",
             f"{'Activity':<{width}}  {'Blocks':>6}  {'Wall time':<27}  Curiosity"]
    for activity in activities:
        duration = seconds[activity]
        share = duration / total if total else 0.0
        time_bar = f"{ascii_bar(duration / 60.0, 12)} {duration:6.1f}s {share:4.0%}"
        score = "-"
        if activity in curiosity:
            progress = float(np.mean(curiosity[activity]))
            score = f"{ascii_bar(progress / curiosity_bar_max, 12)} {progress:.3f}"
        count = blocks[activity] if activity in blocks or activity in curiosity else "-"
        lines.append(f"{activity:<{width}}  {count:>6}  {time_bar}  {score}")
    rates = []
    for primitive, success in sorted(info["competence"].items()):
        rate = f"{success:.1%}" if success is not None else "untested"
        rates.append(f"{primitive} {rate}")
    # the schema's last own explore block (half of its tests start from failed play calls): see the executor
    # score (--patience) for the success from fixed fresh starts
    lines.append("Last explore tests  " + ("  |  ".join(rates) or "none yet"))
    return "\n".join(lines)


def remap_modes(archive, old_bits: list, new_bits: list) -> None:
    """Rewrite an archive's mode bitmasks from one vocabulary's bit order to another's (by relation name;
    bits of relations that no longer exist are dropped)."""
    if old_bits == new_bits:
        return
    where = {n: i for i, n in enumerate(new_bits)}

    def remap(m: int) -> int:
        if m < 0:
            return m
        out = 0
        for b, n in enumerate(old_bits):
            if m >> b & 1 and n in where:
                out |= 1 << where[n]
        return out
    for nd in archive.nodes:
        nd.mode = remap(int(nd.mode))
    archive.sM = [remap(int(m)) for m in archive.sM]


class Loop:
    def __init__(self, out: Path, scene: str, seed: int = 0, workers: int | None = None):
        from .workers import Pool
        self.out, self.scene_name, self.seed = Path(out), scene, seed
        self.out.mkdir(parents=True, exist_ok=True)
        self.scene = scene_info(scene)
        self.pool = Pool(self.scene, workers or LOOP["workers"])
        self.rng = np.random.default_rng(seed)
        self.arbiter = Arbiter(np.random.default_rng(seed + 1))
        self.events = Events()
        self.explorers: dict[str, Explorer] = {}
        self.prims: list[dict] = []
        self.R_all: RelationSet | None = None
        self.vocab_version = 0
        self.player = Player(self.out, scene, seed)
        self.n_blocks = 0
        self.extensions: list = []          # the map, schemas, imagination: sleep(loop) / activities(loop) / run(...)
        state = self.out / "loop_state.pkl"
        if state.exists():                  # resume a run (next epochs after the saved ones)
            with open(state, "rb") as f:
                saved = pickle.load(f)
            self.events, self.prims, self.epoch = saved["events"], saved["prims"], saved["epoch"]
            self.arbiter.hist = saved["arbiter"]
            self.R_all = RelationSet.load(self.out / "relations_all") if self.prims else None
            self._explorers()
        else:
            self.epoch = 0

    # --------------------------------------------------------- vocabulary and schemas
    def _chain(self, p: dict) -> dict:
        return {q["target"]: self.out / f"prim_{q['target']}" for q in self.prims if q["stage"] < p["stage"]}

    def _explorers(self) -> None:
        """One explorer per schema, its relations written in its folder; existing archives are kept (remapped
        to the current vocabulary's bits)."""
        from .babble.select import primitive_relations
        vocab = RelationSet.load(self.out / "relations")
        keep = {}
        for p in self.prims:
            folder = self.out / f"prim_{p['target']}"
            folder.mkdir(parents=True, exist_ok=True)
            rels = primitive_relations(vocab, p["target"])
            old_bits = None
            if (folder / "relations.json").exists():
                old_bits = RelationSet.load(folder / "relations").bit_names()
            rels.save(folder / "relations")
            if old_bits is not None and old_bits != rels.bit_names() and (folder / "archive.npz").exists():
                from .explore.archive import Archive          # (unchanged bits: the archive and its cached fields stay)
                a = Archive.load(folder / "archive.npz")
                remap_modes(a, old_bits, rels.bit_names())
                a.save(folder / "archive.npz")
            ex = self.explorers.get(p["target"])
            keep[p["target"]] = Explorer(self.out, p, self._chain(p), self.seed)
            if ex is not None:
                keep[p["target"]].fail_starts = ex.fail_starts
                keep[p["target"]].fail_infos = getattr(ex, "fail_infos", [None] * len(ex.fail_starts))
        self.explorers = keep

    def sleep(self) -> dict:
        """Between epochs: relations re-read from every event, schemas updated, archives remapped, then the
        extensions' sleep (the map, schema splits and merges)."""
        t0 = time.time()
        old = self.R_all
        self.events.trim(BABBLE["max_probes"])
        R, vocab, prims, report = read_vocabulary(self.events, self.out, self.scene, old)
        self.R_all, self.prims = R, prims
        self.vocab_version += 1
        self._explorers()
        info = {"relations": [r.name for r in R], "schemas": [p["target"] for p in prims]}
        for ext in self.extensions:
            info.update(ext.sleep(self) or {})
        self.pool.restart()                  # memory: fresh workers each epoch (their caches are rebuilt anyway)
        info["sleep_s"] = round(time.time() - t0, 1)
        self.arbiter.wake()
        return info

    # --------------------------------------------------------- activities
    def activities(self) -> list[str]:
        acts = ["babble"] + [f"explore:{t}" for t in self.explorers]
        if any((self.out / f"prim_{t}" / "archive.npz").exists() for t in self.explorers):
            acts.append("play")
        for ext in self.extensions:
            acts += ext.activities(self)
        return acts

    def schema_bases(self, n: int) -> list:
        """Babble restarts from states the schemas reached (their goal states): practising from new skills."""
        nodes = [(ctx_pair(nd.B_pose), nd.state, nd.ctrl) for ex in self.explorers.values()
                 for nd in ex.archive.nodes if nd.goal and ex.prim.kind == "object-object"]
        nodes += [(ctx_pair(nd.B_pose), nd.state, nd.ctrl) for ex in self.explorers.values()
                  for nd in ex.archive.nodes if nd.goal and ex.prim.kind != "object-object"
                  and ctx_pair(nd.B_pose)[1] != "spot"]
        if not nodes:
            return []
        pick = self.rng.integers(len(nodes), size=n)
        return [{"pair": nodes[i][0], "state": nodes[i][1], "ctrl": nodes[i][2]} for i in pick]

    def run_block(self, name: str) -> dict:
        t0 = time.time()
        self.n_blocks += 1
        if name == "babble":
            start = len(self.events)
            st = babble_block(self.pool, self.events, self.rng, task_seed(self.seed, 5000 + self.n_blocks),
                              schema_bases=self.schema_bases)
            competence = 1.0 - st["novelty"]
            for ext in self.extensions:                      # how well the event predictor foresaw them
                c = ext.babble_competence(self.events, start) if hasattr(ext, "babble_competence") else None
                if c is not None:
                    competence = st["predicted"] = c
        elif name.startswith("explore:") and name.split(":", 1)[1] in self.explorers:
            t = name.split(":", 1)[1]
            st = self.explorers[t].run_block(self.pool)
            competence = st["competence"]
        elif name == "play":
            self._write_library()
            st, fails = self.player.run_block(self.pool)
            for t, lst in fails.items():
                ex = self.explorers.get(t)
                if ex is not None:
                    ex.add_failures([ex.prim.cfg_from_view(v, c, pair) for v, c, pair, _ in lst],
                                    [info for *_, info in lst])
            competence = st["competence"]
        else:
            st, competence = next(e for e in self.extensions if name in e.activities(self)).run(self, name)
        self.arbiter.update(name, competence)
        st.update(activity=name, wall_s=round(time.time() - t0, 2),
                  competence=None if competence is None else round(float(competence), 4))
        return st

    def _write_library(self) -> None:
        """The files load_library reads are current (archives and metrics are saved by the explorers); written
        only when changed (the workers rebuild their library when its files change, bep.play.play.library_version)."""
        path = self.out / "primitives.json"
        text = json.dumps([p for p in self.prims if (self.out / f"prim_{p['target']}" / "archive.npz").exists()], indent=1)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)

    # --------------------------------------------------------- epochs
    def run_epoch(self) -> dict:
        t0 = time.time()
        self.player.epoch = self.epoch
        blocks = []
        budget = LOOP["epoch_s"]
        with ProgressBar(f"Epoch {self.epoch}", budget, "s") as bar:
            def run_block(name):
                bar.update(time.time() - t0, name)
                blocks.append(self.run_block(name))

            while not self.prims and time.time() - t0 < budget:     # first epoch: babble until schemas
                run_block("babble")
                if len(self.events.recs) >= BABBLE["min_probes"]:
                    bar.update(time.time() - t0, "discovering schemas")
                    R, vocab, prims, _ = read_vocabulary(self.events, self.out, self.scene, self.R_all)
                    if prims:
                        self.R_all, self.prims = R, prims
                        self._explorers()
            while self.prims and time.time() - t0 < budget:
                self.arbiter.set_activities(self.activities())
                run_block(self.arbiter.pick())
            wall = time.time() - t0
            bar.update(wall, "updating library")
            self._write_library()
            info = {"epoch": self.epoch, "wall_s": round(wall, 1), "blocks": blocks, "arbiter": self.arbiter.summary(),
                    "competence": {t: (ex.history[-1]["success"] if ex.history else None) for t, ex in self.explorers.items()}}
            if self.prims:
                bar.update(wall, "updating models")
            info["sleep"] = self.sleep() if self.prims else {}
            bar.update(wall, "saving snapshot")
            self._snapshot()
            with open(self.out / "epochs.jsonl", "a") as f:
                f.write(json.dumps(info, default=float) + "\n")
            self.epoch += 1
            with open(self.out / "loop_state.pkl", "wb") as f:
                pickle.dump({"events": self.events, "prims": self.prims, "epoch": self.epoch, "arbiter": self.arbiter.hist}, f)
            bar.clear()
        return info

    def _snapshot(self) -> None:
        snap = self.out / f"epoch_{self.epoch}"
        if snap.exists():
            shutil.rmtree(snap)
        snap.mkdir()
        for name in SNAPSHOT:
            if (self.out / name).exists():
                shutil.copy2(self.out / name, snap / name)
        (snap / "primitives.json").write_text(json.dumps(
            [p for p in self.prims if (self.out / f"prim_{p['target']}" / "archive.npz").exists()], indent=1))
        for p in self.prims:
            src = self.out / f"prim_{p['target']}"
            if (src / "archive.npz").exists():
                shutil.copytree(src, snap / src.name)
        for ext in self.extensions:
            if hasattr(ext, "snapshot"):
                ext.snapshot(self, snap)

    def close(self) -> None:
        self.pool.close()


def make_loop(args) -> Loop:
    """The loop with its sleep-time and activity extensions: the cost-to-go map, the schemas' preconditions
    (splits and merges), imagination."""
    from .brain.imagine import ImagineExtension
    from .explore.qmap import MapExtension
    from .play.schemas import SchemaExtension
    loop = Loop(Path(args.out), args.scene, args.seed, args.workers)
    loop.extensions += [MapExtension(), SchemaExtension(), ImagineExtension()]
    return loop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="pair_cubes")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop once the fixed executor score has not improved for this many epochs (0: run every epoch)")
    ap.add_argument("--exec-tests", type=int, default=32, help="fixed executor tests per schema after each epoch")
    args = ap.parse_args()
    args.out = args.out or f"runs/{args.scene}"
    from .world.objects import use_body
    use_body("float")
    loop = make_loop(args)
    curve = Path(args.out) / "curve.jsonl"
    best, since = -1.0, 0
    if curve.exists():                                  # resuming: the convergence state so far
        for ln in curve.read_text().splitlines():
            c = json.loads(ln)
            if c["score"] > best + 0.02:
                best, since = c["score"], 0
            else:
                since += 1
    try:
        while loop.epoch < args.epochs:
            started = time.monotonic()
            info = loop.run_epoch()
            print(epoch_summary(info, time.monotonic() - started), flush=True)
            if args.patience:
                c = epoch_score(loop, args.exec_tests)
                with open(curve, "a") as f:
                    f.write(json.dumps(c) + "\n")
                per = "  |  ".join(f"{t} {v:.1%}" for t, v in sorted(c["exec_fixed"].items()) if v is not None)
                print(f"Executor score {c['score']:g}  (fixed starts: {per})", flush=True)
                if c["score"] > best + 0.02:
                    best, since = c["score"], 0
                else:
                    since += 1
                if since >= args.patience:
                    print(f"CONVERGED after epoch {loop.epoch - 1}: no improvement for {since} epochs", flush=True)
                    break
    finally:
        loop.close()


def epoch_score(loop, n: int) -> dict:
    """The epoch's comparable measure: each schema's executor success from n fixed starts (bep.assess), their sum,
    and the epoch's play success per schema."""
    from collections import Counter
    from .assess import executor_fixed
    snap = loop.out / f"epoch_{loop.epoch - 1}"
    ex = executor_fixed(snap, n, loop.pool)
    last = json.loads((loop.out / "epochs.jsonl").read_text().splitlines()[-1])
    n_play, ok_play = Counter(), Counter()
    for r in loop.player.history[-2000:]:
        if r.get("epoch", loop.epoch - 1) == loop.epoch - 1:
            n_play[r["primitive"]] += 1
            ok_play[r["primitive"]] += bool(r["success"])
    vals = [v for v in ex.values() if v is not None]
    # the sum over schemas: a newly discovered schema adds to the score instead of lowering a mean
    return {"epoch": loop.epoch - 1, "exec_fixed": ex, "score": round(float(np.sum(vals)), 4) if vals else 0.0,
            "schemas": len(ex), "wall_s": last["wall_s"], "sleep_s": last["sleep"].get("sleep_s"),
            "play": {t: round(ok_play[t] / n_play[t], 3) for t in n_play}}


if __name__ == "__main__":
    from .terminal import run_main
    run_main(main)
