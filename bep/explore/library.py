"""The library of a scene's learning: its relations and explored primitives, and how to call them.

* ``load_library``: the explored primitives of a learning directory (relations, primitive, field, goal
  states), or of several merged (minimal scenes learned separately).
* ``bindings``: every way to call a primitive in a world: subject X of the primitive's X kind, target a
  body of its Y kind or a table spot (object-object), none otherwise.
* ``load_play``: the play records of the learning(s), with names as in the merged library.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..world.geometry import Spot
from ..world.objects import is_fixture
from ..world.relations import CC, HC, SC, RelationSet
from ..world.scenes import table_spot_fn
from .archive import Archive
from .field import ModeField
from ..babble.select import primitive_relations
from .primitive import DiscoveredPrimitive, load_end_states


def learned_kinds(scene: str | Path) -> set | None:
    """The (X kind, Y kind) pairs a learning directory was probed on (its scene's pairs), None if unknown."""
    from ..world.scenes import SCENES, scene_pairs
    path = Path(scene)
    metadata = path / "learning.json"
    if metadata.exists():
        return {tuple(p) for p in json.loads(metadata.read_text())["pairs"]}
    name = path.name
    return set(scene_pairs(SCENES[name])) if name in SCENES else None


def load_library(scene) -> tuple:
    """(relations, target -> (primitive, field, goal states)) of one learning directory, or of several merged.

    Merging several (minimal-scene) learnings: every relation and primitive keeps its own learning, renamed
    ``<learning>.<name>`` so that names never clash, and remembers the body kinds it was learned on
    (``Relation.kinds``); the merged relation set reads each relation only on kind pairs its learning saw,
    or on pairs no learning saw (extrapolation to unseen bodies)."""
    if isinstance(scene, (list, tuple)):
        if len(scene) == 1:
            return load_library(scene[0])
        return _merge([Path(p) for p in scene])
    scene = Path(scene)
    R = RelationSet.load(scene / "relations")
    lib, specs = {}, json.loads((scene / "primitives.json").read_text())
    covered = learned_kinds(scene)
    for p in specs:
        if covered is not None and p.get("roles") and not p.get("channel"):   # channels: roles of the primitive only
            relation = R.get(p["target"])
            relation.kinds = {tuple(k) for k in p["roles"]}
            relation.covered = covered
        arc = scene / f"prim_{p['target']}" / "archive.npz"
        if not arc.exists():
            continue
        prim = DiscoveredPrimitive(primitive_relations(R, p["target"]), p["target"], kinds=tuple(p["kinds"]),
                                   probes=str(scene / "probes.npz"))
        prim.roles = {tuple(k) for k in p["roles"]} if "roles" in p else set(prim.observed_roles())
        ex = scene / f"prim_{p['target']}" / "executor.json"
        prim.score = json.loads(ex.read_text())["success"] if ex.exists() else None
        mt = scene / f"prim_{p['target']}" / "metric.json"
        if mt.exists():                   # the distance learned during exploration (before the roadmap is built)
            prim.set_metric({int(m): w for m, w in json.loads(mt.read_text()).items()})
        A = Archive.load(arc)
        lib[p["target"]] = (prim, ModeField(A, prim), np.array([nd.z for nd in A.nodes if nd.goal]))
    for p in specs:                       # chained primitives start where earlier stages' calls end (as in exploration)
        if p["target"] in lib:
            ends = [load_end_states(end, q["kinds"]) for q in specs if q["stage"] < p["stage"]
                    if (end := scene / f"prim_{q['target']}" / "ends.npz").exists()]
            if ends:
                lib[p["target"]][0].set_chain(ends)
    _cover(lib)
    return R, lib


def _cover(lib: dict) -> None:
    """Roles any primitive of the library was learned in, per relation kind (a primitive extrapolates only to
    roles no primitive of its kind covers)."""
    covered: dict = {}
    for prim, _, _ in lib.values():
        covered.setdefault(prim.kind, set()).update(prim.roles)
    for prim, _, _ in lib.values():
        prim.covered = covered[prim.kind]


def _probe_views(d: Path) -> dict:
    """A learning's probe states, as features per relation kind (to compare relations across learnings)."""
    from ..world.objects import VIEW_S
    from ..world.relations import cc_features, hc_features, sc_features
    f = np.load(d / "probes.npz")
    s2 = f["s2"]
    return {CC: [cc_features(s2, VIEW_S, o) for o in ("AB", "BA")],
            HC: [hc_features(f["ev_s"], VIEW_S, rotation=False, body=True)],
            SC: [sc_features(s2, VIEW_S, o) for o in ("A", "B")]}


def _same(a, b, views: list[dict], overlap: float = 0.5) -> bool:
    """Two relations of different learnings are one if, on the probe states of both learnings, they fire on
    mostly the same states (more than ``overlap`` of the smaller firing set)."""
    if a.kind != b.kind or a.members.shape[1] != b.members.shape[1]:
        return False
    fa = np.concatenate([a.holds_f(f).ravel() for v in views for f in v[a.kind]])
    fb = np.concatenate([b.holds_f(f).ravel() for v in views for f in v[b.kind]])
    small = min(fa.sum(), fb.sum())
    return bool(small) and (fa & fb).sum() > overlap * small


def _merge(dirs: list[Path]) -> tuple:
    rels, lib, rest, owner, views = [], {}, [], {}, []
    for d in dirs:
        R, sub = load_library(d)
        kinds, tag = learned_kinds(d), d.name
        for r in R.relations:
            r.name = f"{tag}.{r.name}"
            r.kinds = r.kinds or kinds
            owner[r.name] = len(views)
        for t, (prim, field, goals) in sub.items():
            prim.target, prim.name = f"{tag}.{t}", f"prim_{tag}.{t}"
            prim.origin = prim.target              # its name in its own learning (play records)
            lib[prim.target] = (prim, field, goals)
        rels += R.relations
        views.append(_probe_views(d))
        if R.cc:                                   # resting tilt is learned where bodies rest on each other
            rest.append(R.rest_tilt)
    # one relation learned in several minimal scenes (e.g. "cube held in the fingers"): merged under the first
    # name; of their primitives, the one whose executor scored best is kept (it serves the union of roles)
    aliases, keep = {}, []
    for r in rels:
        twin = next((k for k in keep if owner[k.name] != owner[r.name] and _same(k, r, views)), None)
        if twin is None:
            keep.append(r)
            continue
        old = r.name
        aliases[old] = twin.name
        twin.kinds = (twin.kinds or set()) | (r.kinds or set())
        r.name = twin.name                         # its own primitive (own relation set) now speaks of the twin
        if old in lib:
            pr = lib.pop(old)
            pr[0].target, pr[0].name = twin.name, f"prim_{twin.name}"
            if twin.name not in lib or _better(pr[0], lib[twin.name][0]):
                if twin.name in lib:
                    pr[0].roles = pr[0].roles | lib[twin.name][0].roles
                    aliases[f"prim:{lib[twin.name][0].origin}"] = None
                lib[twin.name] = pr
                aliases[f"prim:{old}"] = twin.name
            else:
                lib[twin.name][0].roles |= pr[0].roles
                aliases[f"prim:{old}"] = None      # its play records describe a primitive no longer used
    rels = keep
    covered = set().union(*(learned_kinds(d) or set() for d in dirs))
    for r in rels:
        r.covered = covered
    Rall = RelationSet(rels, max(rest) if rest else None)
    Rall.aliases = aliases
    for prim, _, _ in lib.values():
        prim.Rall = Rall                           # relations a call must keep may come from other learnings
    _cover(lib)
    return Rall, lib


def _better(a, b) -> bool:
    """Primitive a measured better than b (executor success at the end of exploration; unknown: b stays)."""
    return a.score is not None and (b.score is None or a.score > b.score)


def learning_dirs(spec, given: str | None = None) -> list[Path]:
    """Learning directories of a scene: the given ones (comma-separated), else those of the minimal scenes it
    is learned from, else its own."""
    if given:
        return [Path(p) for p in given.split(",")]
    return [Path("runs") / n for n in (spec.learned_from or [spec.name])]


def load_play(scene, R=None, robot: bool = False) -> list[dict]:
    """Play records of one learning directory, or of several (relation and primitive names as load_library;
    ``R``: the merged relation set, whose aliases rename merged relations and drop unused primitives); with
    ``robot``, the robot's validation calls too (bep.validate, play_robot.jsonl)."""
    from ..play.effect import load as _load
    files = ("play.jsonl", "play_robot.jsonl") if robot else ("play.jsonl",)

    def load(d):
        return [r for f in files if (Path(d) / f).exists() for r in _load(Path(d) / f)]
    if not isinstance(scene, (list, tuple)) or len(scene) == 1:
        return load(Path(scene[0] if isinstance(scene, (list, tuple)) else scene))
    al = getattr(R, "aliases", {}) if R is not None else {}
    out = []
    for d in map(Path, scene):
        tag = d.name
        name = lambda q: al.get(f"{tag}.{q}", f"{tag}.{q}")                       # noqa: E731
        ren = lambda insts: [[name(i[0]), *i[1:]] for i in insts]                 # noqa: E731
        for r in load(d):
            prim = f"{tag}.{r['primitive']}"
            key = f"prim:{prim}"
            if key in al and al[key] is None:
                continue
            out.append(dict(r, primitive=al.get(key, prim) if key in al else prim,
                            before=ren(r["before"]), after=ren(r["after"])))
    return out


def bindings(env, prim) -> list[tuple]:
    """(X, target) pairs a primitive can be called with in this world; target None, a body or "spot"."""
    def fits(n, kind):
        """A fixture needs its own kind; any free body may take a free body's role (a primitive learned on
        one shape is extrapolated to others)."""
        return env.kinds[n] == kind if is_fixture(kind) or is_fixture(env.kinds[n]) else True
    r = prim.rel_target
    roles, covered = getattr(prim, "roles", None), getattr(prim, "covered", set())

    def role_ok(kx, ky=None):
        """Learned in this role, or a role no primitive of its kind was learned in (extrapolation)."""
        if not roles:
            return True
        if ky is None:                                   # X alone (hand-object, state, or a table spot)
            return kx in {k[0] for k in roles} or kx not in {k[0] for k in covered}
        return (kx, ky) in roles or (kx, ky) not in covered
    kd = env.kinds
    xs = [n for n in env.cubes if fits(n, prim.kinds[0]) and r.applies(kd[n]) and role_ok(kd[n])]
    if prim.kind != CC:
        return [(x, None) for x in xs]
    out = []
    for x in xs:
        out += [(x, y) for y in env.cubes if y != x and fits(y, prim.kinds[1]) and r.applies(kd[x], kd[y])
                and role_ok(kd[x], kd[y])]
        if not is_fixture(prim.kinds[1]) and prim.accepts_spot(env, x):
            out.append((x, "spot"))
    return out


def resolve_target(env, target, rng):
    """'spot' -> a random table spot clear of fixtures; otherwise unchanged."""
    if target == "spot":
        return Spot(*table_spot_fn(env)(rng), 0.0)
    return target
