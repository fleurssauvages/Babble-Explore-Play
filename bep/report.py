"""README material from the learning runs: per-phase results and one GIF per phase.

    python -m bep.report                    # every runs/<scene> present -> assets/

assets/results.json   per learning: time and amount of work of every phase; zero-shot task results
assets/results.md     the same as Markdown tables, also written into README.md between the RESULTS markers
assets/<scene>_<phase>.gif   the phase's clips stitched together (probe, explore, play, solve)
Task results are read from runs/<scene>/solve_<task>.json (bep.brain.solve --out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from .config import CONTROL_DT, SUBSTEPS

PHASES = ("probe", "explore", "play", "solve")
PHASE_TITLES = {"probe": "1 · Probe", "explore": "2 · Explore", "play": "3 · Play", "solve": "4 · Zero-shot task"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def scene_results(run: Path) -> dict:
    timing = json.loads((run / "timing.json").read_text())
    probe = json.loads((run / "probe_report.json").read_text())
    phys_dt = CONTROL_DT / SUBSTEPS
    out = {"total_s": timing["total_s"],
           "probe": {"time_s": timing["probe_s"], "probes": probe["n_probes"], "events": probe["events"],
                     "relations": len(probe["relations"]),
                     "physics_steps": int(round(probe.get("sim_s", 0.0) / phys_dt)),
                     "simulated_s": probe.get("sim_s")},
           "explore": {"time_s": round(sum(timing["explore_s"].values()), 1), "primitives": {}}}
    tot = {"rounds": 0, "tested": 0, "transitions": 0, "control_steps": 0}
    for p in json.loads((run / "primitives.json").read_text()):
        t = p["target"]
        log = _jsonl(run / f"prim_{t}" / "log.jsonl")
        last = log[-1] if log else {}
        cov = next((e["coverage"] for e in reversed(log) if "coverage" in e), {})
        d = {"kind": p["kind"], "time_s": timing["explore_s"].get(t), "rounds": len(log),
             "tested": last.get("tested"), "transitions": last.get("edges"), "nodes": last.get("nodes"),
             "control_steps": last.get("sim_steps"),
             "start_coverage": None if not cov else round(1 - cov["start_capped_frac"], 2)}
        out["explore"]["primitives"][t] = d
        for k in tot:
            tot[k] += d.get(k) or 0
    out["explore"].update(tot)
    ex = _jsonl(run / "play.jsonl")
    per = {}
    for r in ex:
        d = per.setdefault(r["primitive"], [0, 0])
        d[0] += r["success"]
        d[1] += 1
    out["play"] = {"time_s": timing["play_s"], "executed": len(ex),
                   "success": {t: f"{a}/{n}" for t, (a, n) in sorted(per.items())},
                   "control_steps": int(round(sum(r["time"] for r in ex) / CONTROL_DT))}
    val = run / "validation.json"
    if val.exists():
        v = json.loads(val.read_text())
        out["validate"] = {"time_s": timing.get("validate_s"),
                           "primitives": {t: (d["hand_of_god"], d["robot"]) for t, d in v["primitives"].items()},
                           "robot_play": v.get("robot_play", {})}
    return out


def task_results(runs: Path) -> list[dict]:
    out = []
    for f in sorted(runs.glob("*/solve_*.json")):
        out.append(json.loads(f.read_text())["summary"])
    return out


def markdown(res: dict) -> str:
    scenes = list(res)
    fmt = lambda v: "–" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:g}")  # noqa: E731
    rows = [("**Probe** time", lambda r: f"{r['probe']['time_s']:.0f} s"),
            ("probes / events", lambda r: f"{r['probe']['probes']:,} / {r['probe']['events']:,}"),
            ("physics steps", lambda r: fmt(r["probe"]["physics_steps"])),
            ("relations found", lambda r: fmt(r["probe"]["relations"])),
            ("**Explore** time", lambda r: f"{r['explore']['time_s']:.0f} s"),
            ("primitives", lambda r: ", ".join(f"{t} ({d['kind'].split('-')[0][0].upper()}{d['kind'].split('-')[1][0].upper()}"
                                               f", {d['time_s']:.0f} s)" for t, d in r["explore"]["primitives"].items())),
            ("rounds / candidates tested", lambda r: f"{r['explore']['rounds']} / {fmt(r['explore']['tested'])}"),
            ("validated transitions", lambda r: fmt(r["explore"]["transitions"])),
            ("control steps (100 Hz)", lambda r: fmt(r["explore"]["control_steps"])),
            ("**Play** time", lambda r: f"{r['play']['time_s']:.0f} s"),
            ("executed calls (successes per primitive)",
             lambda r: f"{r['play']['executed']} ({', '.join(f'{t}: {v}' for t, v in r['play']['success'].items())})"),
            ("control steps (100 Hz)", lambda r: fmt(r["play"]["control_steps"])),
            ("**Validate** time", lambda r: f"{r['validate']['time_s']:.0f} s" if r.get("validate") else "–"),
            ("primitive success: floating hand / robot", lambda r: ", ".join(
                f"{t}: {fmt(h)} / {fmt(b)}" for t, (h, b) in r["validate"]["primitives"].items()) if r.get("validate") else "–"),
            ("**Total learning**", lambda r: f"**{r['total_s'] / 60:.1f} min**")]
    lines = ["| | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
    for name, f in rows:
        lines.append(f"| {name} | " + " | ".join(f(res[s]) for s in scenes) + " |")
    return "\n".join(lines) + "\n"


def task_markdown(tasks: list[dict]) -> str:
    fmt = lambda v: "–" if v is None else f"{v:.1f} s"  # noqa: E731
    lines = ["| Zero-shot task | motor | success | median time (simulated) | decisions | planning (wall) |",
             "|---|---|---|---|---|---|"]
    for d in tasks:
        lines.append(f"| {d['scene']} · {d['task']} (`{d['param']}`) | {d.get('motor', 'robot')} | "
                     f"{d['success_rate'] * 100:.0f}% of {d['episodes']} | {fmt(d['median_time_s'])} | "
                     f"{d['mean_decisions']:.1f} | {d.get('mean_plan_wall_s', 0):.1f} s |")
    return "\n".join(lines) + "\n"


def stitch(clips: list[Path], out: Path, fps: int = 10, width: int = 320, max_s: float = 24.0) -> bool:
    """Concatenate clips (subsampled to ``fps``, resized to ``width``) into one looping GIF."""
    import imageio.v2 as imageio
    from PIL import Image
    frames = []
    per = max_s / max(1, len(clips))
    for c in clips:
        r = imageio.get_reader(c)
        step = max(1, int(round(r.get_meta_data().get("fps", 25) / fps)))
        for i, f in enumerate(r):
            if i % step:
                continue
            if len(frames) and i / (step * fps) > per:
                break
            im = Image.fromarray(f)
            frames.append(im.resize((width, int(im.height * width / im.width)), Image.LANCZOS))
        r.close()
    if not frames:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return True


def phase_clips(run: Path, phase: str) -> list[Path]:
    c = run / "clips"
    pats = {"probe": ["probe_*.mp4"], "explore": ["explore_*.mp4"], "play": ["play_executed_*.mp4"],
            "solve": ["solve_*.mp4"]}[phase]
    return [p for pat in pats for p in sorted(c.glob(pat))]


def main() -> None:
    from .world.scenes import SCENES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--assets", default="assets")
    args = ap.parse_args()
    runs, assets = Path(args.runs), Path(args.assets)
    res = {s: scene_results(runs / s) for s in SCENES if (runs / s / "timing.json").exists()}
    tasks = task_results(runs)
    assets.mkdir(exist_ok=True)
    (assets / "results.json").write_text(json.dumps({"learning": res, "tasks": tasks}, indent=1))
    md = markdown(res) + ("\n" + task_markdown(tasks) if tasks else "")
    (assets / "results.md").write_text(md)
    for s in res:
        for ph in PHASES:
            ok = stitch(phase_clips(runs / s, ph), assets / f"{s}_{ph}.gif")
            print(s, ph, "gif" if ok else "no clips")
    readme = Path("README.md")
    if readme.exists() and "<!-- RESULTS -->" in readme.read_text():        # refresh the README tables
        txt = readme.read_text()
        i, j = txt.index("<!-- RESULTS -->"), txt.index("<!-- /RESULTS -->")
        readme.write_text(txt[:i] + "<!-- RESULTS -->\n" + md + txt[j:])
    print(md)


if __name__ == "__main__":
    main()
