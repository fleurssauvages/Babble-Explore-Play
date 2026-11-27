"""Bodies of any shape, articulated fixtures, and the worlds that hold them.

* ``Shape``: a box, a cylinder, or a *compound* of boxes.
  Shapes are properties of the world (the simulator knows them); the pipeline uses them
  only to express relations in each body's own units.
* ``Fixture``: a static base with one *moving part* on a slide or hinge joint (a drawer,
  a door, a lid).  The moving part is a body like any other: it has a pose, contacts and
  extents, so it takes part in pair relations ("X inside the drawer"); in addition it has
  a joint value, normalised by the joint range, for state relations ("closed").
* ``ObjWorld``: Franka + named free bodies + named fixtures.  It provides a *pair view*:
  any (X, Y) pair of movable bodies (free bodies or moving parts), or X and a table spot,
  in one fixed layout that also carries both shapes and joint values, so relations are
  evaluated identically for every pair.
* ``pair_world``: cached two-body worlds (roles X = "A", Y = "B"), each role a free shape
  or a fixture; worlds with the same kinds of roles share a state dimension.
* ``facts``: a shape-aware geometric yardstick, used only to score pipelines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import mujoco
import numpy as np

from .env import ASSET_DIR, GRIPPER_MAX_WIDTH, HOME_Q, StackEnv, ee_down_rot
from .geometry import Spot, characteristic_length, layout_of, yaw_q

# the robot and two-body part of every pair view (the original two-cube layout)
CUBE_VIEW_ORDER = [("ee_pos", 3), ("ee_quat", 4), ("A_pos", 3), ("A_quat", 4), ("B_pos", 3), ("B_quat", 4),
                   ("finger1", 1), ("finger2", 1), ("lf_A", 1), ("rf_A", 1), ("A_B", 1), ("A_table", 1),
                   ("B_table", 1), ("hand_B", 1), ("hand_table", 1), ("A_vel", 3), ("B_vel", 3)]

COLORS = ["0.20 0.45 0.90", "0.90 0.30 0.20", "0.25 0.75 0.30", "0.95 0.80 0.15", "0.65 0.35 0.85",
          "0.15 0.80 0.80", "0.85 0.55 0.25", "0.55 0.55 0.60"]


# ------------------------------------------------------------------ shapes and fixtures
@dataclass(frozen=True)
class Shape:
    kind: str                      # "box" | "cylinder" | "compound"
    half: tuple                    # half extents of the bounding box (x, y, z); cylinder: (r, r, half height)
    mass: float = 0.08
    parts: tuple = ()              # compound: ((half_x, half_y, half_z), (x, y, z) offset) boxes

    @property
    def ext(self) -> np.ndarray:
        return np.array(self.half, float)

    @property
    def length(self) -> float:
        return float(characteristic_length(self.half))

    def floor_clearance(self, rotation: np.ndarray | None = None) -> float:
        """Minimum centre height above a plane for this orientation."""
        if rotation is None:
            return self.half[2]
        vertical = rotation[2]
        if self.kind == "cylinder":
            return float(self.half[0] * np.linalg.norm(vertical[:2]) + self.half[2] * abs(vertical[2]))
        if self.kind == "compound":
            return float(max(np.abs(vertical) @ h - vertical @ p for h, p in self.parts))
        return float(np.abs(vertical) @ self.ext)

    @property
    def sym(self) -> float:
        """Period of the shape's symmetry about the vertical axis (0 = any angle)."""
        if self.kind == "cylinder":
            return 0.0
        if self.kind == "compound":
            return 2 * np.pi
        return np.pi / 2 if abs(self.half[0] - self.half[1]) < 1e-6 else np.pi

    def sample_yaw(self, rng) -> float:
        """Sample one full period of physically distinct angles about the vertical."""
        return float(rng.uniform(-self.sym / 2.0, self.sym / 2.0))

    @property
    def grasp_width(self) -> float:
        """Narrowest horizontal width of any part (what fingers can close on)."""
        if self.kind == "compound":
            return float(min(2 * min(h[0], h[1]) for h, _ in self.parts))
        return float(2 * min(self.half[0], self.half[1]))

    def geoms_xml(self, name: str, rgba: str) -> str:
        if self.kind == "compound":
            m = self.mass / len(self.parts)
            return "\n".join(f'      <geom name="{name}_geom{k}" class="cube" type="box" size="{h[0]} {h[1]} {h[2]}" '
                             f'pos="{o[0]} {o[1]} {o[2]}" mass="{m}" rgba="{rgba} 1"/>'
                             for k, (h, o) in enumerate(self.parts))
        size = f"{self.half[0]} {self.half[2]}" if self.kind == "cylinder" else f"{self.half[0]} {self.half[1]} {self.half[2]}"
        return f'      <geom name="{name}_geom" class="cube" type="{self.kind}" size="{size}" mass="{self.mass}" rgba="{rgba} 1"/>'


def compound(parts, mass: float) -> Shape:
    lo = np.min([np.array(o) - np.array(h) for h, o in parts], axis=0)
    hi = np.max([np.array(o) + np.array(h) for h, o in parts], axis=0)
    centre = (lo + hi) / 2
    parts = tuple((tuple(h), tuple(np.round(np.array(o) - centre, 5))) for h, o in parts)
    return Shape("compound", tuple(np.round((hi - lo) / 2, 5)), mass, parts)


@dataclass(frozen=True)
class Fixture:
    """Static base + one moving part on a joint.  Part geometry is given in the part frame; the part
    frame sits at ``part_pos`` in the base frame when the joint is at 0."""
    base: tuple                    # ((half), (pos)) static boxes in the base frame
    part: Shape                    # moving part (compound shape), frame at its bounding-box centre
    part_pos: tuple
    joint: str                     # "slide" | "hinge"
    axis: tuple
    range: tuple
    joint_pos: tuple = (0.0, 0.0, 0.0)     # joint anchor in the part frame (hinge)
    damping: float = 4.0
    frictionloss: float = 0.6
    region: tuple = ((0.74, 0.80), (-0.12, 0.12), (np.pi - 0.15, np.pi + 0.15))   # where the base may stand

    @property
    def ext(self) -> np.ndarray:
        return self.part.ext


LIBRARY = {
    "cube": Shape("box", (0.025, 0.025, 0.025), 0.08),
    "can": Shape("cylinder", (0.022, 0.022, 0.03), 0.07),
    "tall": Shape("box", (0.02, 0.02, 0.04), 0.08),
}
# articulated fixtures (static base + one moving part on a joint).  None in the stacking scenes; the
# generic support (object-state relations, moving parts in pair views) is kept for later scenes.
FIXTURES: dict = {}
SPOT = Shape("box", (0.05, 0.05, 0.025))           # virtual support under a table spot
# body kinds; "spot" is the virtual reference standing in for Y when a scene has a single body
TYPES = sorted(LIBRARY) + sorted(FIXTURES) + ["spot"]
LEGACY_TYPES = ("brick", "can", "cube", "slab", "tall", "spot")

VIEW_ORDER = CUBE_VIEW_ORDER + [("A_ext", 3), ("B_ext", 3), ("A_sym", 1), ("B_sym", 1), ("hand_A", 1),
                                ("A_q", 1), ("B_q", 1), ("A_qrange", 2), ("B_qrange", 2), ("A_art", 1),
                                ("B_art", 1), ("A_gw", 1), ("B_gw", 1)]
_kw, VIEW_SLICES, VIEW_DIM = layout_of(VIEW_ORDER)
VIEW_S = SimpleNamespace(**_kw)


# ------------------------------------------------------------------ bodies: the robot or the floating hand
# Learning (babble, explore, play) runs on the floating hand: the same gripper without an arm, moved by the
# Hand of God within the robot's reachability map (bep.world.reach); validation and robot solving run on the
# Franka arm.  The choice is process-wide (each phase is a process).
BODY = {"kind": "robot"}
HAND_HOME = (np.array([0.5295, 0.0, 0.2793]), np.diag([1.0, -1.0, -1.0]))   # the pinch at the arm's home pose
PINCH = np.array([0.0, 0.0, 0.1034])                # pinch site in the hand body's frame (same orientation)


def use_body(kind: str) -> None:
    assert kind in ("robot", "float"), kind
    BODY["kind"] = kind


def body_kind() -> str:
    return BODY["kind"]


def float_hand_xml() -> str:
    """The floating-hand model: the gripper subtree of the robot model (hand, fingers, pads, pinch site,
    finger tendon and actuator) on a free joint, welded to a mocap target the Hand of God moves.  Only the meshes
    the hand uses are kept (the arm's would be loaded in every world for nothing: ~0.4 M vertices, 100-250 MB)."""
    import re
    path = ASSET_DIR / "_generated_hand_float.xml"
    version = "<!-- float_hand_xml: hand meshes only -->"
    if path.exists() and version in path.read_text():
        return path.name
    src = (ASSET_DIR / "panda_gripper_box.xml").read_text()
    head = src[:src.index("<worldbody>")]
    i = src.index('<body name="hand"')
    depth, j = 0, i
    while True:                                          # the hand's subtree: up to its matching </body>
        a, b = src.find("<body", j), src.find("</body>", j)
        if a != -1 and a < b:
            depth, j = depth + 1, a + 5
        else:
            depth, j = depth - 1, b + 7
            if depth == 0:
                break
    hand = src[i:j]
    hand = hand[:hand.index(">") + 1].replace('pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834"',
                                              'pos="0.5295 0 0.3827" quat="0 1 0 0"') \
        + '\n      <freejoint name="hand_free"/>' + hand[hand.index(">") + 1:]
    tail = src[src.index("<tendon>"):src.index("<actuator>")]
    tail = tail.replace("</equality>", '  <weld body1="hand_target" body2="hand" relpose="0 0 0 1 0 0 0" '
                                       'solref="0.01 1" solimp="0.95 0.99 0.001"/>\n  </equality>')
    used = set(re.findall(r'mesh="([^"]+)"', hand))
    def mesh_name(ln):                                   # a mesh's name, or its file's stem when it has none
        m = re.search(r'name="([^"]+)"', ln) or re.search(r'file="(?:[^"]*/)?([^"/.]+)\.[^"]*"', ln)
        return m.group(1) if m else None
    head = "".join(ln for ln in head.splitlines(keepends=True) if "<mesh" not in ln or mesh_name(ln) in used)
    act = src[src.index("<actuator>"):src.index("</actuator>")]
    act = "<actuator>\n" + act[act.index("    <!-- Remap"):] + "</actuator>\n"
    xml = (head + '<worldbody>\n    <body name="hand_target" mocap="true" pos="0.5295 0 0.3827" quat="0 1 0 0"/>\n    '
           + hand + "\n  </worldbody>\n\n  " + tail + "  " + act + "</mujoco>\n")
    xml = xml.replace('<mujoco model="panda_gripper">', f'<mujoco model="floating_hand">\n  {version}')
    pending = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    pending.write_text(xml)
    pending.replace(path)
    return path.name


# ------------------------------------------------------------------ scene XML
DIST_CUTOFF = 0.06                  # distance sensors report this (m) beyond it: the continuous barrier has faded
COLLISION_GEOM = {"left_finger": "left_finger_box", "right_finger": "right_finger_box"}   # else "<part>_collision"
def scene_xml(shapes: dict[str, Shape], fixtures: dict[str, Fixture], body: str = "robot") -> str:
    """Franka (or the floating hand) + table + free bodies + fixtures, with pose / contact / joint sensors for
    every movable body."""
    base = (ASSET_DIR / "stacking_scene.xml").read_text()
    hand_file = float_hand_xml() if body == "float" else "panda_gripper_box.xml"   # one box collision per finger
    base = base.replace('file="panda_gripper.xml"', f'file="{hand_file}"')
    head = base[:base.index("<worldbody>")]
    world = base[base.index("<worldbody>"):base.index("</worldbody>")]
    world = world[:world.index('<body name="cubeB"')]
    # the floor under the table: a body pushed off the table comes to rest (far from any goal)
    world += '\n    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1" rgba="0.3 0.3 0.3 1"/>\n'
    movable = sorted(shapes) + sorted(fixtures)
    bodies, sensors, excl = [], [], []
    for i, n in enumerate(sorted(shapes)):
        bodies.append(f'''    <body name="cube{n}" pos="{0.3 + 0.07 * i} -0.55 {shapes[n].half[2]}">
      <freejoint name="cube{n}_free"/>
{shapes[n].geoms_xml("cube" + n, COLORS[i % len(COLORS)])}
      <site name="cube{n}_site" size="0.003" rgba="0 0 0 0"/>
    </body>''')
    for n, fx in sorted(fixtures.items()):
        bgeoms = "\n".join(f'      <geom name="fix{n}_base{k}" type="box" size="{h[0]} {h[1]} {h[2]}" '
                           f'pos="{o[0]} {o[1]} {o[2]}" rgba="0.55 0.42 0.30 1"/>' for k, (h, o) in enumerate(fx.base))
        jp = fx.joint_pos
        bodies.append(f'''    <body name="fix{n}" pos="0.8 0 0">
{bgeoms}
      <body name="cube{n}" pos="{fx.part_pos[0]} {fx.part_pos[1]} {fx.part_pos[2]}">
        <joint name="{n}_joint" type="{fx.joint}" axis="{fx.axis[0]} {fx.axis[1]} {fx.axis[2]}" pos="{jp[0]} {jp[1]} {jp[2]}"
               range="{fx.range[0]} {fx.range[1]}" damping="{fx.damping}" frictionloss="{fx.frictionloss}" armature="0.05"/>
{fx.part.geoms_xml("cube" + n, "0.62 0.48 0.33")}
        <site name="cube{n}_site" size="0.003" rgba="0 0 0 0"/>
      </body>
    </body>''')
        excl.append(f'    <exclude body1="fix{n}" body2="cube{n}"/>')
        sensors.append(f'<jointpos name="{n}_q" joint="{n}_joint"/>')
    for n in movable:
        sensors += [f'<framepos name="{n}_pos" objtype="site" objname="cube{n}_site"/>',
                    f'<framequat name="{n}_quat" objtype="site" objname="cube{n}_site"/>',
                    f'<framelinvel name="{n}_vel" objtype="site" objname="cube{n}_site"/>',
                    f'<contact name="lf_{n}" body1="left_finger" body2="cube{n}" data="found" reduce="mindist"/>',
                    f'<contact name="rf_{n}" body1="right_finger" body2="cube{n}" data="found" reduce="mindist"/>',
                    f'<contact name="{n}_table" body1="cube{n}" geom2="table" data="found" reduce="mindist"/>',
                    f'<contact name="hand_{n}" subtree1="hand" body2="cube{n}" data="found" reduce="mindist"/>',
                    # the whole moving body (the arm's links and the hand; the floating hand alone) on the cube
                    f'<contact name="body_{n}" subtree1="{"hand" if body == "float" else "link1"}" body2="cube{n}" '
                    'data="found" reduce="mindist"/>']
    for i, x in enumerate(movable):
        for y in movable[i + 1:]:
            a, b = sorted((x, y))
            sensors.append(f'<contact name="{a}_{b}" body1="cube{a}" body2="cube{b}" data="found" reduce="mindist"/>')
    # signed distances of the barrier pairs (the continuous barrier, EXEC['barrier_sigma']): every part of the moving
    # body that can collide with every movable body, and every pair of movable bodies
    parts = ("hand", "left_finger", "right_finger") if body == "float" else ("link6", "link7", "hand", "left_finger",
                                                                           "right_finger")
    # (each part's collision geom only: the visual meshes do not collide and are costly to measure)
    coll = {p: COLLISION_GEOM.get(p, f"{p}_collision") for p in parts}
    for n in movable:
        for part in parts:
            sensors.append(f'<distance name="dist_{part}_{n}" geom1="{coll[part]}" body2="cube{n}" '
                           f'cutoff="{DIST_CUTOFF}"/>')
    for i, x in enumerate(movable):
        for y in movable[i + 1:]:
            a, b = sorted((x, y))
            sensors.append(f'<distance name="dist_{a}_{b}" body1="cube{a}" body2="cube{b}" cutoff="{DIST_CUTOFF}"/>')
    for part in parts:                     # every part of the moving body above the table (its height: the table barrier)
        sensors.append(f'<distance name="dist_{part}_table" geom1="{coll[part]}" geom2="table" cutoff="{DIST_CUTOFF}"/>')
    for n in movable:                      # every movable body against the table (Phi's blending across contacts)
        sensors.append(f'<distance name="dist_{n}_table" body1="cube{n}" geom2="table" cutoff="{DIST_CUTOFF}"/>')
    sensors += ['<framepos name="ee_pos" objtype="site" objname="pinch"/>',
                '<framequat name="ee_quat" objtype="site" objname="pinch"/>',
                '<jointpos name="finger1" joint="finger_joint1"/>',
                '<jointpos name="finger2" joint="finger_joint2"/>',
                f'<contact name="hand_table" subtree1="{"hand" if body == "float" else "link6"}" geom2="table" '
                'data="found" reduce="mindist"/>']
    contact = "\n  <contact>\n" + "\n".join(excl) + "\n  </contact>\n" if excl else ""
    return (head + world + "\n".join(bodies) + "\n  </worldbody>\n" + contact
            + "\n  <sensor>\n    " + "\n    ".join(sensors) + "\n  </sensor>\n</mujoco>\n")


# ------------------------------------------------------------------ world
class ObjWorld(StackEnv):
    """Franka (or the floating hand) + named free bodies of arbitrary shapes + named fixtures."""

    def __init__(self, shapes: dict[str, Shape], fixtures: dict[str, Fixture] | None = None,
                 tag: str = "obj", body: str | None = None) -> None:
        self.body = body or body_kind()
        self.floating = self.body == "float"
        self.shapes = dict(shapes)
        self.fixtures = dict(fixtures or {})
        self.cubes = sorted(self.shapes) + sorted(self.fixtures)        # every movable body
        self.free = sorted(self.shapes)
        path = ASSET_DIR / f"_generated_{tag}_{os.getpid()}_{''.join(self.cubes)}.xml"
        path.write_text(scene_xml(self.shapes, self.fixtures, self.body))
        try:
            self._stack_init(path)
        finally:
            path.unlink(missing_ok=True)
        m = self.model
        jid = lambda n: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n))  # noqa: E731
        self.cube_qadr = {n: int(m.jnt_qposadr[jid(f"cube{n}_free")]) for n in self.free}
        self.cube_dadr = {n: int(m.jnt_dofadr[jid(f"cube{n}_free")]) for n in self.free}
        self.joint_qadr = {n: int(m.jnt_qposadr[jid(f"{n}_joint")]) for n in self.fixtures}
        self.joint_dadr = {n: int(m.jnt_dofadr[jid(f"{n}_joint")]) for n in self.fixtures}
        self.fix_bid = {n: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"fix{n}")) for n in self.fixtures}
        self.sidx = {}
        for s in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s)
            self.sidx[name] = np.arange(m.sensor_adr[s], m.sensor_adr[s] + m.sensor_dim[s])
        self._views: dict = {}
        self._pviews: dict = {}
        from ..config import CONTROL_DT, SUBSTEPS
        self.model.opt.timestep = CONTROL_DT / SUBSTEPS      # the controller's physics everywhere

    def _stack_init(self, path):
        """The model setup of bep.env.StackEnv (robot collision filtering, finger pads, IK data, ...),
        without requiring two free bodies named A and B."""
        from .env import ACTIVE_COLLISION_BODIES
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        name2id = lambda t, n: int(mujoco.mj_name2id(m, t, n))  # noqa: E731
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.gi = m.nu - 1                                  # the gripper: the last actuator of either body
        if self.floating:
            self.hand_qadr = int(m.jnt_qposadr[name2id(J, "hand_free")])
            self.hand_dadr = int(m.jnt_dofadr[name2id(J, "hand_free")])
            self.mocap_id = int(m.body_mocapid[name2id(mujoco.mjtObj.mjOBJ_BODY, "hand_target")])
            self.arm_qadr = self.arm_dadr = None
        else:
            self.arm_qadr = np.array([m.jnt_qposadr[name2id(J, f"joint{i}")] for i in range(1, 8)])
            self.arm_dadr = np.array([m.jnt_dofadr[name2id(J, f"joint{i}")] for i in range(1, 8)])
        self.finger_qadr = np.array([m.jnt_qposadr[name2id(J, f"finger_joint{i}")] for i in (1, 2)])
        self.pinch_sid = name2id(mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.table_gid = name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")

        self.finger_bids = (name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                            name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"))
        self.A_bid = name2id(mujoco.mjtObj.mjOBJ_BODY, "cubeA")
        self.B_bid = name2id(mujoco.mjtObj.mjOBJ_BODY, "cubeB")
        root = name2id(mujoco.mjtObj.mjOBJ_BODY, "hand" if self.floating else "link0")
        active = {name2id(mujoco.mjtObj.mjOBJ_BODY, n) for n in ACTIVE_COLLISION_BODIES}
        self.robot_bids = set()
        for b in range(m.nbody):
            anc = b
            while anc > 0 and anc != root:
                anc = int(m.body_parentid[anc])
            if anc == root:
                self.robot_bids.add(b)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in self.robot_bids and b not in active and m.geom_contype[g] != 0:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            if b in self.finger_bids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] != 0:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            if b in self.finger_bids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_friction[g] = (1.2, 0.02, 0.002)
                m.geom_condim[g] = 4
        if not self.floating:
            self.jnt_low = np.array([m.jnt_range[name2id(J, f"joint{i}"), 0] for i in range(1, 8)])
            self.jnt_high = np.array([m.jnt_range[name2id(J, f"joint{i}"), 1] for i in range(1, 8)])
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self.nsensordata = int(m.nsensordata)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = int(mujoco.mj_stateSize(m, self.state_spec))
        self._ik_data = mujoco.MjData(m)

    def shape(self, n) -> Shape:
        return self.shapes[n] if n in self.shapes else self.fixtures[n].part

    # ------------------------------------------------------------ placement
    def place(self, n, x, y, yaw, z=None, data=None):
        d = self.data if data is None else data
        a, v = self.cube_qadr[n], self.cube_dadr[n]
        d.qpos[a:a + 3] = (x, y, self.shapes[n].half[2] if z is None else z)
        d.qpos[a + 3:a + 7] = yaw_q(yaw)
        d.qvel[v:v + 6] = 0.0

    def pair_contact(self, x: str, y: str) -> str:
        a, b = sorted((x, y))
        return f"{a}_{b}"

    def reset_robot(self) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        if self.floating:
            self.set_hand(*HAND_HOME)
        else:
            d.qpos[self.arm_qadr] = HOME_Q
            d.ctrl[:7] = HOME_Q
        d.qpos[self.finger_qadr] = GRIPPER_MAX_WIDTH / 2
        d.ctrl[self.gi] = 255.0

    # ------------------------------------------------------------ the floating hand
    def set_hand(self, pos, R, data=None) -> None:
        """Teleport the floating hand: pinch point at pos, rotation R (its mocap target with it)."""
        d = self.data if data is None else data
        R = np.asarray(R, float)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        a, v = self.hand_qadr, self.hand_dadr
        d.qpos[a:a + 3] = np.asarray(pos, float) - R @ PINCH
        d.qpos[a + 3:a + 7] = q
        d.qvel[v:v + 6] = 0.0
        self.move_hand(pos, R, d)

    def set_state(self, state, ctrl=None, data=None) -> None:
        """Restore a physics state; on the floating hand the mocap target (not part of the physics state) is
        put back where the hand is, so the weld holds it there."""
        super().set_state(state, ctrl, data)
        if self.floating:
            d = self.data if data is None else data
            a = self.hand_qadr
            d.mocap_pos[self.mocap_id] = d.qpos[a:a + 3]
            d.mocap_quat[self.mocap_id] = d.qpos[a + 3:a + 7]

    def move_hand(self, pos, R, data=None) -> None:
        """Move the floating hand's mocap target (the weld pulls the hand there)."""
        d = self.data if data is None else data
        R = np.asarray(R, float)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        d.mocap_pos[self.mocap_id] = np.asarray(pos, float) - R @ PINCH
        d.mocap_quat[self.mocap_id] = q

    def hand_pose(self, data=None):
        """Pinch point and hand rotation (either body)."""
        d = self.data if data is None else data
        return d.site_xpos[self.pinch_sid].copy(), d.site_xmat[self.pinch_sid].reshape(3, 3).copy()

    def place_cube(self, which, x, y, yaw, z=None, data=None, quat=None):
        if which in self.fixtures:
            return self.place_fixture(which, x, y, yaw)
        d = self.data if data is None else data
        a, v = self.cube_qadr[which], self.cube_dadr[which]
        d.qpos[a:a + 3] = (x, y, self.shapes[which].half[2] if z is None else z)
        d.qpos[a + 3:a + 7] = yaw_q(yaw) if quat is None else quat
        d.qvel[v:v + 6] = 0.0

    def place_fixture(self, n, x, y, yaw, z=0.0):
        """Move a fixture's static base (model-level pose; z > 0: standing on another fixture)."""
        self.model.body_pos[self.fix_bid[n]] = (x, y, z)
        self.model.body_quat[self.fix_bid[n]] = yaw_q(yaw)

    def set_joint(self, n, q_norm: float, data=None):
        d = self.data if data is None else data
        lo, hi = self.fixtures[n].range
        d.qpos[self.joint_qadr[n]] = lo + float(np.clip(q_norm, 0, 1)) * (hi - lo)
        d.qvel[self.joint_dadr[n]] = 0.0

    def reset_fixtures(self, q=None):
        for n in self.fixtures:
            self.set_joint(n, 0.0 if q is None else q.get(n, 0.0))

    # ------------------------------------------------------------ pair view
    def _consts(self, const, role, n_or_shape):
        if isinstance(n_or_shape, Shape):
            sh, art, rng = n_or_shape, 0.0, (0.0, 1.0)
        else:
            sh = self.shape(n_or_shape)
            fx = self.fixtures.get(n_or_shape)
            art, rng = (1.0, fx.range) if fx else (0.0, (0.0, 1.0))
        const[VIEW_SLICES[f"{role}_ext"]] = sh.ext
        const[VIEW_SLICES[f"{role}_sym"]] = sh.sym
        const[VIEW_SLICES[f"{role}_qrange"]] = rng
        const[VIEW_SLICES[f"{role}_art"]] = art
        # opening unit: the body's own width for simple shapes (transfers across shapes); the gripper's
        # range for compound bodies, whose graspable sub-parts differ in width
        const[VIEW_SLICES[f"{role}_gw"]] = GRIPPER_MAX_WIDTH if sh.kind == "compound" else sh.grasp_width

    def pview_spec(self, x: str, target) -> tuple:
        """(constants, destination slots, sensor indices) of the pair view of x and target.  Cached per (x,
        body), and per x for table spots, whose pose is written into a copy of the cached constants (a cache
        keyed by spot pose would grow with every random spot)."""
        spot = not isinstance(target, str)
        key = (x, None if spot else target)
        if key not in self._pviews:
            src = {"ee_pos": "ee_pos", "ee_quat": "ee_quat", "A_pos": f"{x}_pos", "A_quat": f"{x}_quat",
                   "finger1": "finger1", "finger2": "finger2", "lf_A": f"lf_{x}", "rf_A": f"rf_{x}",
                   "hand_table": "hand_table", "A_vel": f"{x}_vel", "hand_A": f"hand_{x}", "A_table": f"{x}_table"}
            if x in self.fixtures:
                src["A_q"] = f"{x}_q"
            const = np.zeros(VIEW_DIM)
            self._consts(const, "A", x)
            if not spot:
                src.update({"B_pos": f"{target}_pos", "B_quat": f"{target}_quat", "A_B": self.pair_contact(x, target),
                            "B_table": f"{target}_table", "hand_B": f"hand_{target}", "B_vel": f"{target}_vel"})
                if target in self.fixtures:
                    src["B_q"] = f"{target}_q"
                self._consts(const, "B", target)
            else:
                const[VIEW_SLICES["B_table"]] = 1.0
                src["A_B"] = f"{x}_table"              # touching the virtual support = touching the table
                self._consts(const, "B", SPOT)
            dst = np.concatenate([VIEW_SLICES[k] for k in src])
            srcc = np.concatenate([self.sidx[v] for v in src.values()])
            self._pviews[key] = (const, dst, srcc)
        const, dst, srcc = self._pviews[key]
        if spot:
            const = const.copy()
            const[VIEW_SLICES["B_pos"]] = (target.x, target.y, -SPOT.half[2])
            const[VIEW_SLICES["B_quat"]] = yaw_q(target.yaw)
        return const, dst, srcc

    def pair_view(self, sens: np.ndarray, x: str, target) -> np.ndarray:
        """Two-body layout (VIEW_S) with A := x and B := target (body name or Spot)."""
        const, dst, srcc = self.pview_spec(x, target)
        out = np.empty(sens.shape[:-1] + (VIEW_DIM,))
        out[...] = const
        out[..., dst] = sens[..., srcc]
        return out

    # ------------------------------------------------------------ yardstick
    def facts(self, sens=None) -> dict:
        """Shape-aware symbolic state of the free bodies: support (body, 'table', 'held', 'other'), clear,
        held, width, and the normalised joint value of every fixture."""
        s = self.sensors() if sens is None else sens
        width = float(s[self.sidx["finger1"][0]] + s[self.sidx["finger2"][0]])
        pos = {n: s[self.sidx[f"{n}_pos"]] for n in self.cubes}
        support, held = {}, None
        for n in self.free:
            e = self.shapes[n].ext
            lf, rf = s[self.sidx[f"lf_{n}"][0]] > 0, s[self.sidx[f"rf_{n}"][0]] > 0
            if lf and rf:
                support[n], held = "held", n
                continue
            sup = None
            for m in self.cubes:
                if m == n or s[self.sidx[self.pair_contact(n, m)][0]] <= 0:
                    continue
                em = self.shape(m).ext
                d = pos[n] - pos[m]
                if m in self.fixtures:
                    if np.all(np.abs(d[:2]) < em[:2]) and d[2] > -em[2]:
                        sup = m
                elif (np.hypot(d[0], d[1]) < np.linalg.norm(em[:2])
                      and abs(d[2] - e[2] - em[2]) < min(e[2], em[2]) / 2):
                    sup = m
            if sup is None and s[self.sidx[f"{n}_table"][0]] > 0:
                sup = "table"
            support[n] = sup or "other"
        clear = {n: not any(v == n for v in support.values()) for n in self.free}
        joints = {}
        for n, fx in self.fixtures.items():
            lo, hi = fx.range
            joints[n] = float((s[self.sidx[f"{n}_q"][0]] - lo) / (hi - lo))
        return {"support": support, "clear": clear, "held": held, "width": width, "joints": joints,
                "pos": {n: p.copy() for n, p in pos.items()}}


_SWAP = [("A_pos", "B_pos"), ("A_quat", "B_quat"), ("A_vel", "B_vel"), ("A_table", "B_table"), ("A_ext", "B_ext"),
         ("A_sym", "B_sym"), ("A_q", "B_q"), ("A_qrange", "B_qrange"), ("A_art", "B_art"), ("A_gw", "B_gw"),
         ("hand_A", "hand_B")]


def swap_view(v: np.ndarray) -> np.ndarray:
    """The same pair view with the roles exchanged (X <-> Y).  Finger contacts are only recorded for A,
    so after the swap they are unknown (set to 0)."""
    out = v.copy()
    for a, b in _SWAP:
        out[..., VIEW_SLICES[a]], out[..., VIEW_SLICES[b]] = v[..., VIEW_SLICES[b]], v[..., VIEW_SLICES[a]]
    out[..., VIEW_SLICES["lf_A"]] = 0.0
    out[..., VIEW_SLICES["rf_A"]] = 0.0
    return out


def is_fixture(kind: str) -> bool:
    return kind in FIXTURES


def kind_shape(kind: str) -> Shape:
    if kind == "spot":
        return SPOT
    return FIXTURES[kind].part if kind in FIXTURES else LIBRARY[kind]


def pair_world(tx: str, ty: str, body: str | None = None) -> ObjWorld:
    """Two-body world of the current body (robot or floating hand), see _pair_world."""
    return _pair_world(tx, ty, body or body_kind())


@lru_cache(maxsize=None)
def _pair_world(tx: str, ty: str, body: str) -> ObjWorld:
    """Two-body world: X = "A" and Y = "B", each a free shape or a fixture of the given kind (Y = "spot":
    no body B, the virtual reference stands in for it)."""
    shapes = {r: LIBRARY[k] for r, k in (("A", tx), ("B", ty)) if k in LIBRARY}
    fixtures = {r: FIXTURES[k] for r, k in (("A", tx), ("B", ty)) if k in FIXTURES}
    env = ObjWorld(shapes, fixtures, tag=f"pair_{tx}_{ty}_{body}", body=body)
    env.kinds = {r: k for r, k in (("A", tx), ("B", ty)) if r in env.cubes}
    return env


def apply_layout(env: ObjWorld, layout: dict, settle_steps: int = 50, joints: dict | None = None) -> None:
    """layout: free body -> (x, y, yaw) on the table, or (support, yaw_offset) on another free body;
    fixture -> (x, y, yaw) of its base.  ``joints``: fixture -> normalised joint value (default closed)."""
    env.reset_robot()
    placed: dict = {}
    for n, v in layout.items():
        if n in env.fixtures:
            env.place_fixture(n, *v)
    env.reset_fixtures(joints)

    def put(n):
        if n in placed:
            return placed[n]
        v = layout[n]
        if isinstance(v[0], str):
            xs, ys, zs, yaws = put(v[0])
            z = zs + env.shapes[v[0]].half[2] + env.shapes[n].half[2] + 0.001
            placed[n] = (xs, ys, z, yaws + v[1])
        else:
            placed[n] = (v[0], v[1], env.shapes[n].half[2], v[2])
        return placed[n]

    for n in layout:
        if n in env.shapes:
            x, y, z, yaw = put(n)
            env.place(n, x, y, yaw, z)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data, nstep=settle_steps)
    env.data.time = 0.0


def realize_world(env: ObjWorld, place: dict, hand=None, *, joints: dict | None = None, settle_s: float = 0.3,
                  depth: float = 0.002):
    """Hand of God in a world with N bodies: keep every body where it is except those in ``place``
    (free body -> (position, yaw)) and ``joints`` (fixture -> normalised value); put the hand at
    ``hand`` = (position, yaw, finger opening, gripper ctrl) or at the home pose; settle; return the
    sensors (None if IK fails or something interpenetrates)."""
    d, m = env.data, env.model
    for n, (p, yaw) in place.items():
        env.place(n, float(p[0]), float(p[1]), float(yaw), float(p[2]))
    for n, q in (joints or {}).items():
        env.set_joint(n, q)
    d.qvel[:] = 0.0
    if env.floating:                                      # the floating hand: home (tasks only use hand=None)
        assert hand is None
        env.set_hand(*HAND_HOME)
        d.qpos[env.finger_qadr] = GRIPPER_MAX_WIDTH / 2
        d.ctrl[env.gi] = 255.0
        mujoco.mj_forward(m, d)
        if env.bad_penetration(d, depth=depth):
            return None
        mujoco.mj_step(m, d, nstep=max(1, int(settle_s / m.opt.timestep)))
        d.time = 0.0
        return env.sensors()
    if hand is None:
        q, w, g = HOME_Q, GRIPPER_MAX_WIDTH, 255.0
    else:
        pE, yawE, w, g = hand
        q, ok = env.solve_ik(np.asarray(pE, float), ee_down_rot(yawE), env.arm_q())
        if not ok:
            q, ok = env.solve_ik(np.asarray(pE, float), ee_down_rot(yawE + np.pi), env.arm_q())
            if not ok:
                return None
    d.qpos[env.arm_qadr] = q
    d.qpos[env.finger_qadr] = float(w) / 2
    d.ctrl[:7] = q
    d.ctrl[7] = g
    mujoco.mj_forward(m, d)
    if env.bad_penetration(d, depth=depth):
        return None
    mujoco.mj_step(m, d, nstep=max(1, int(settle_s / m.opt.timestep)))
    d.time = 0.0
    return env.sensors()


def overlap(pXY: np.ndarray, yawXY: float, sx: Shape, sy: Shape, margin: float = 0.002) -> bool:
    """Do two upright bodies interpenetrate?  pXY = X's centre in Y's frame.  Bounding boxes are tested
    with the separating-axis theorem (conservative for cylinders and compounds; physics has the last
    word through the penetration test)."""
    if sx.kind == "compound" or sy.kind == "compound":
        return False                       # bounding boxes of tools and fixtures overlap legitimately
    if abs(pXY[2]) >= sx.half[2] + sy.half[2] - 1e-4:
        return False
    c, s = np.cos(yawXY), np.sin(yawXY)
    Rx = np.array([[c, -s], [s, c]])
    cx = (np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * sx.ext[:2]) @ Rx.T + pXY[:2]
    cy = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * sy.ext[:2]
    for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), Rx[:, 0], Rx[:, 1]):
        a, b = cx @ ax, cy @ ax
        if a.max() + margin < b.min() or b.max() + margin < a.min():
            return False
    return True


def table_layout(rng, names, shapes, spot_fn, min_gap: float = 0.04) -> dict:
    """Every free body resting on the table, apart from the others."""
    pos: dict = {}
    for n in names:
        r_n = float(np.hypot(*shapes[n].half[:2]))
        for _ in range(2000):
            p = np.array(spot_fn(rng))
            if all(np.linalg.norm(p - np.array(q[:2])) > r_n + float(np.hypot(*shapes[m].half[:2])) + min_gap
                   for m, q in pos.items()):
                pos[n] = (float(p[0]), float(p[1]), shapes[n].sample_yaw(rng))
                break
        else:
            raise RuntimeError("could not place the objects apart")
    return pos


__all__ = ["Shape", "Fixture", "LIBRARY", "FIXTURES", "TYPES", "SPOT", "VIEW_S", "VIEW_DIM", "ObjWorld",
           "pair_world", "apply_layout", "realize_world", "overlap", "table_layout", "Spot", "compound",
           "is_fixture", "kind_shape"]
