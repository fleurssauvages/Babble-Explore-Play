"""Every constant of the pipeline, in one place.

Three kinds, kept apart on purpose:

* WORLD   properties of the robot and the objects, given by the simulator
          (object shapes in bep.world.objects, gripper range, reachable table area).
* LEARNED quantities measured from data (probe statistics, demonstrated rates,
          descent slopes); they are not set here, only how they are measured.
* ALGORITHM generic hyperparameters of the methods.  Geometric ones are written
  in the objects' own units (see bep.world.relations) or in physical units
  supplied by the robot model.

No constant below names a relation, a mode or a task.
"""
from __future__ import annotations

import numpy as np

from .world.env import GRIPPER_MAX_WIDTH

# ------------------------------------------------------------------ WORLD
GRIP_MAX = GRIPPER_MAX_WIDTH         # finger opening range (m)
REACH = (0.25, 0.80)                 # radial reach of the arm on the table (m)
TABLE_RHO = (0.36, 0.68)             # comfortable radial band for placing objects (m)
TABLE_ANGLE = 0.75                   # half-angle of the table area in front of the robot (rad)
ANGLE_CODE_RANGE = 0.25              # |angle code| <= 1/4 (see relations.sym_code)
AXIS_CODE_RANGE = 0.5                # closing-axis code: a 0.5 change (~30 degrees) counts as one object unit

HAND = "hand"            # name of the robot hand in relation instances (never a body name)
CONTROL_DT, SUBSTEPS = 0.01, 2       # 100 Hz control, 2 physics steps each (bep.motor.mppi.MPPIConfig defaults)

# ------------------------------------------------------------------ ALGORITHM: babble (probe -> relations)
# Relative positions in object units: X relative to Y in (Y extent + X radius, sum of half heights),
# the hand relative to X in X's half extents.
PROBE = dict(
    near_frac=0.6,                   # fraction of probes with X close to Y (contact-rich sampling)
    near_xy=1.5, near_z=1.8,
    # otherwise X stands apart from Y, but never farther than one probe sweep (segments x segment) beyond
    # contact, and the hand starts within one sweep of X: every probe can make contact
    hand_near_frac=0.7,
    hand_near_sd=1.0, hand_near_z=(-0.4, 2.4),
    part_frac=0.5,                   # with fixtures: fraction of probes whose X is a moving part
    joint_limit_frac=0.5,            # fraction of moving parts started at an end of their joint range
    settle_s=0.4,
    segments=4, segment=0.8,         # one probe sweep: segments x segment lengths of the subject
    joint_motion_min=0.1,           # minimum movement as a share of joint range
    rest_speed=0.4,                  # characteristic lengths per second below which the body is at rest
    tilt_margin=0.01,                # tolerance in the tilt measure (1 - cos) beyond observed members
    focus_sd=0.25,                   # a restart from a probe's configuration perturbs it this much (object units,
                                     # and radians for the yaws)
)


# ------------------------------------------------------------------ ALGORITHM: which relations get a primitive
PRIMS = dict(
    min_members=100,                 # object-object relations need this much probe support
    min_lift_precision=0.3,          # hand-object relations must let the hand carry the body
)

# ------------------------------------------------------------------ ALGORITHM: reverse Go-Explore
EXPLORE = dict(
    eps=0.5,                         # epsilon-ball radius in the normalised metric (object units)
    novelty=1.2,                     # candidate is redundant within novelty * eps of a solved node
    # proposal arms: the backward operators, plus "dir" (step the solved node closest to a start state toward
    # it).  Arms are chosen by a bandit on their yield: new solved nodes per simulated second (+ realisation
    # cost), with a floor so that every arm keeps being tried.
    ops=("XY", "EX", "rot", "grip", "rel", "joint", "swap", "dir"),
    arm_floor=0.04, arm_window=60, arm_prior=(1.0, 4.0),   # floor probability; recent tests per arm; prior
                                     # (successes, cost in seconds) of every arm
    realize_cost_s=0.3,              # charged per test for the Hand-of-God realisation (seconds of simulation)
    joint_step=0.25,                 # joint operator: step in normalised joint units
    rot_step=0.6,                    # rot operator: rotation of the hand (rad, per axis); directed: max turn
    precheck_tries=10,               # resample a proposal until the Hand of God's geometric checks pass
    step0={"XY": 1.2, "EX": 1.6},    # steps measured in characteristic lengths of the active roles
    step_bounds=(0.4, 6.0),
    target_rate=(0.35, 0.6),         # adapt steps to keep validation success in this band
    step_gain=1.2, window=40, min_window=12,
    goal_hold_s=0.2,                 # goal must hold this long
    goal_select_weight=0.1,          # goal nodes need little expansion
    n_goal_seeds=6, directed_top=5,
    chain_p=1.0,                     # a chained primitive starts where the earlier ones end (composition)
    exec_timeout_s=12.0,             # simulated seconds per executor test
    goal_share=0.33,                 # new goal states are added only while goals are < this share of the rest
    settle_s=0.3,
    knock_factor=3.0,                # Y counts as knocked over beyond this multiple of the resting tilt
    stall_s=0.5, stall_rel=0.02,     # abort a validation whose distance to its target has not improved by
                                     # stall_rel for stall_s seconds
    inflight_per_worker=2,           # asynchronous queue depth (candidates in flight per worker)
    prefilter=0.5,                   # skip a proposal whose predicted state lies within prefilter * novelty * eps
                                     # of a solved node (checked before simulating)
    revalidate_p=0.1,                # share of tests that re-run a validated edge: edges keep a success count
                                     # and Dijkstra prices them by expected time c / p
    metric_clip=(0.5, 2.0),          # learned metric: per mode, the fixed weights times the field's relative
                                     # rates, clipped to this range (eps keeps its meaning)
    node_cap=500,                    # beyond this many nodes, a new node must lie farther from the solved nodes of its
                                     # mode, by sqrt(nodes / node_cap) (0: no cap)
    node_budget=1500,                # most nodes a schema's graph keeps: beyond it, after each explore block, the most
                                     # redundant nodes no shortest path runs through are removed (0: no budget)
    typical_seeds=True,              # goal seeds: the most typical observed goal states first (else at random)
)

# ------------------------------------------------------------------ ALGORITHM: field Phi
FIELD = dict(
    rate_quantile=75.0, min_pairs=50,
    kappa_quantile=0.5, kappa_default=1.0,   # slope of Phi off the samples: the median demonstrated descent
    tau=0.05, k=16, cap_margin=3.0, min_samples=20,
    compact=True,          # prune samples the envelope already predicts (within the soft-min's resolution tau ln k)
    unexplored_s=1.5,      # a mode without samples of its own: pooled estimate + this (seconds), so the
                           # controller never prefers an unexplored mode (e.g. dropping a carried body)
    metric="exploration",  # Phi's distances: "exploration" (the learned, clipped metric) or "rate" (1 / rates)
)
FIELD_LEARN = dict(        # the field exploration builds for itself (stop rule, executor tests)
    kappa_quantile=0.1, compact=True, metric="rate",
)

# ------------------------------------------------------------------ ALGORITHM: primitive execution (Spike-MPPI cost)
EXEC = dict(
    phi_weight=10.0, terminal_weight=5.0,     # same cost shape as the baseline field cost
    side_effect_weight=1.0,                   # weight of every "leave the rest of the world alone" term
    timeout_s=15.0, hold_s=0.2, watch_s=0.1,     # done once the goal held 0.2 s (still controlled: a confirmation, not a wait)
    contact_grace_s=0.15,           # tolerate intermittent fingertip contact during an established transfer
    recoveries=2,                    # floating hand: re-entries into the primitive's graph after a stall
    stall_s=1.5, stall_rel=0.02,     # abort a call whose field value has not improved for stall_s seconds
    roadmap=True,                    # execute by hops along the validated graph (else: descend the field)
    hop_mode_penalty=0.05, hop_terminal=3.0,   # the hop cost = exploration's validation cost (LocalTargetCost)
    hop_stall=3.0,                   # a hop not completed within hop_stall x (typical edge time + 0.5 s) stalls
    hop_join=2.0,                    # farther than this many typical hops from the graph: descend the field
    barrier_weight=1.0,              # the barrier at contact, per pair and control step (~ 10 s of Phi per second)
    barrier_sigma=0.005,             # the barrier: barrier_weight * exp(-distance / sigma) per pair and step, from the
                                     # signed distance of the pair (contact: the full weight). Small: a grasp next to a
                                     # support has the fingers ~2 cm from it
    barrier_sigma_table=0.002,       # the table's barrier: barrier_weight * exp(-height / sigma) per part of the moving
                                     # body and step, its height above the table (short: grasps go near the table)
    mode_blend=0.01,                 # Phi across contacts: a contact absent but nearer than this (m) blends Phi toward
                                     # the mode with the contact, in proportion (1 at contact): continuous (0: off)
    side_effects=False,              # task execution: Y's tilt and speed, kept relations, leave-alone, carry drop
    transport=False,                 # task execution: add the carried body's distance to its destination to Phi
    arm_horizon=50,                  # the arm's Spike-MPPI horizon in tasks (control steps; the floating hand's: HOG)
    goal_reward=50.0,                # reward scaled by the share of rollout steps in the primitive's goal state
)

# ------------------------------------------------------------------ the Hand-of-God motor (learning's body)
HOG = dict(
    v_max=0.25,                      # m/s of the pinch point
    w_max=1.5,                       # rad/s of the hand
    grip_s=0.3,                      # a hop ends this long after the gripper is actuated
    grip_switch_cost=1.0,            # Spike-MPPI: cost of each gripper change in a sampled schedule (where the gripper
                                     # does not matter it is kept, not switched at random; the goal reward outweighs it)
    max_reentries=4,                 # a call stalls after this many failed graph entries (a failed node also excludes
                                     # the nodes sharing its successor: retries try another way to the same place)
    regrip_aims=2,                   # execution: re-aims of a hop whose fingers would close off the node's place on X
    horizon=50,                      # the floating hand's planner (bep.motor.hand_mppi): control steps ahead
)

# ------------------------------------------------------------------ ALGORITHM: play (executed calls only)
PLAY = dict(
    chain_max=8,                     # executed calls chained per episode (each from where the previous ended)
    timeout_s=12.0,
    random_frac=0.3,                 # executed calls chosen uniformly; the rest where the models disagree most
    ensemble=5,                      # bootstrap success models per primitive (their disagreement = what to try)
    min_fit=10,                      # executed calls of a primitive before it has a success model
    min_per_prim=20,                 # calls of primitives with fewer calls than this come first ...
    min_per_context=4,               # ... and calls in relational contexts (bep.play.play.context) met fewer times
    rare_p=0.7,                      # (calls in rarer contexts are chosen first, this often)
    pre_filter=0.1,                  # calls the schema's precondition predicts below this success are not tried
    pre_bypass=0.1,                  # ... except in this share of the play steps (a precondition learned while a schema
                                     # failed would otherwise keep it from ever being tried again)
    history_cap=4000,                # the most recent executed calls the effect models of an episode, and the schemas'
                                     # preconditions at sleep (bep.play.schemas), are fitted on
)




# ------------------------------------------------------------------ babble (bep.babble.probe, bep.babble.events)
# Probes are unscripted: from a random relational configuration (bep.babble.probe.sample_config), the hand
# issues ``commands`` random motor commands, each a straight move in a random direction or a finger command
# (close / open).  After every command the scene is left to settle with the command held (a settle branch,
# not followed): an event is the change between two settled states.  At the end the hand lets go and
# withdraws, and the scene rests (what stays after letting go).
BABBLE = dict(
    commands=6,                  # motor commands per probe
    down_p=0.75,                 # body schema: the hand starts in the robot's working posture (pointing down, random
                                 # yaw) this often, otherwise in a random orientation
    move_len=(0.4, 1.6),         # a move's length, in characteristic lengths of X
    move_up_bias=10.0,           # moves pointing down are flattened: v_z >= -move_up_bias * |v_xy| (10: barely)
    move_s=0.4,                  # seconds per move (speed = length / move_s)
    grip_p=0.35,                 # share of finger commands among the motor commands
    grip_s=0.3,                  # seconds a finger command acts
    settle_s=0.3,                # settle branch after each command
    release_s=0.25, retract=1.2, rest_s=0.4,   # end of a probe: open, withdraw (X lengths), rest
    restart_p=0.5,               # share of probes restarting from an archived state with a rare event key
    restart_pool=4000,
    schema_p=0.3,                # share of the restarts from states the schemas reached (once there are schemas)
    block=512,                   # probes per babble block
    max_probes=30000,            # the event store keeps the most recent probes (memory over long runs)
    min_probes=3000,             # probes before the first relations are read
)
EVENTS = dict(
    follow_frac=0.5,             # X follows: it moved >= follow_frac of the hand's displacement, along it
    moved=0.25,                  # X moved: beyond this share of its length
    tipped=0.15,                 # X tipped: its tilt measure changed by more than this
    min_members=15,              # minimum events of a key
    min_context=20,              # minimum events of a precondition context to measure a rate in it
    min_in_context=5,            # ... and at least this many events of the key in that context
    lift=3.0,                    # contingent: some context raises the key's rate this many times above its base rate
    precision=0.05,              # ... to at least this rate
    constant=0.95,               # a contact flag enters a signature when this share of the states agree
    mirror=0.5,                  # two keys are mirrors when this share of one's events are the other's swapped
    chunk_p=0.6,                 # a pair of consecutive keys is a chunk when the second follows the first this often
)

# ------------------------------------------------------------------ the epoch and its activities
LOOP = dict(
    epoch_s=60.0,                # wall-time budget of an epoch's activities (sleep not included)
    explore_rounds=20,           # rounds of reverse Go-Explore per explore block
    exec_tests=16,               # executor tests after an explore block (half from play failures when any)
    ends_per_block=16,           # successful end states kept per explore block (chained schemas start there)
    fail_starts=200,             # failure start states kept per schema
    fail_p=0.35,                 # share of directed proposals aimed at a failure start
    workers=None,                # default: all cores
)
ARBITER = dict(
    window=2,                    # learning progress: |mean of the last window - mean of the window before|
    floor=0.08,                  # each activity keeps this probability (shared)
    optimism=1.0,                # learning progress of an activity tried fewer than 2 * window times
    temperature=0.15,            # softmax over learning progress
)

# ------------------------------------------------------------------ sleep: the temporal-distance map (bep.explore.qmap)
QMAP = dict(
    hidden=128, embed=32,        # encoder 2 x hidden -> 2 x embed (symmetric and asymmetric parts, MRN head)
    ensemble=3,
    steps=900, warm_steps=500,   # Adam steps (first fit, warm-started refits)
    batch=1024, lr=2e-3,
    goals=32,                    # goal states per schema kept for Phi (farthest-point subsample)
    tau=0.05,                    # soft minimum over goal states (seconds)
    traj_w=0.5,                  # weight of the trajectory upper bounds d(s_t, s_t') <= t' - t
    spread_w=0.05,               # weight of pushing unrelated pairs apart (up to the label cap)
    threads=12,                  # torch threads at sleep (the pool's workers are idle then)
    uncertainty=0.0,             # Phi = mean + uncertainty * std over the ensemble
    holdout=0.15,                # share of each schema's samples held out to judge the map against its field
    max_samples=60000,           # labelled samples per schema used per fit (subsampled: long runs)
    support_q=0.005,             # the data's extent per mode: the [q, 1 - q] quantiles of each coordinate (0: off);
                                 # outside it Phi is the map at the state clipped into it + kappa x the clipped
                                 # distance (weighted L1), so that Phi never decreases away from the data
)

# ------------------------------------------------------------------ schemas (bep.play.schemas)
SCHEMA = dict(
    min_calls=24,                # calls of a schema before a split is considered
    split_gain=0.08,             # information gain (bits) a discrete indicator needs to split a schema
    split_min=8,                 # calls on each side of a split
    merge_overlap=0.8,           # schemas whose goal relations fire on mostly the same states are merged
    ensemble=5, hidden=32, steps=300,
)

# ------------------------------------------------------------------ decide (bep.brain.brain)
BRAIN = dict(
    real_prior=2.0,              # weight (in calls) of the model's success probability against reality
    real_weight=5.0,             # a real call counts as this many play calls when the effect model is refit
    max_states=20000,            # most scene states expanded per search
    goal_weight_s=40.0,          # planning priority penalty per unmet goal literal (in expected execution seconds)
    min_p=0.05,                  # calls less likely than this to succeed are not considered
    habit_trust=3,               # a cached decision is used after this many successes in a row
    spots=8,                     # table spots sampled per placement on a spot (scored by Phi)
    phi_spots=True,              # choose a placement's table spot by Phi (else the clearest spot)
)


def _overrides() -> None:
    """Constants overridden for one run (experiments): BEP_CONFIG='{"LOOP": {"epoch_s": 30}}' (inherited by the
    pool's workers)."""
    import json
    import os
    for name, values in json.loads(os.environ.get("BEP_CONFIG", "{}") or "{}").items():
        globals()[name].update(values)


_overrides()


def progress(done, total, msg: str = "") -> None:
    """One machine-readable progress line (read by bep.learn)."""
    import json
    print("@progress " + json.dumps({"done": done, "total": total, "msg": msg}), flush=True)


def z_scale() -> np.ndarray:
    """Normalisation of the 14-D relational state z: positions, opening and joint values are already in
    object units; the yaw code is divided by its range, the closing-axis code (unit-vector components) by
    AXIS_CODE_RANGE."""
    return np.array([1.0] * 3 + [ANGLE_CODE_RANGE] * 2 + [1.0] * 3 + [AXIS_CODE_RANGE] * 3 + [1.0, 1.0, 1.0])


def table_spot(rng) -> tuple[float, float]:
    r, a = rng.uniform(*TABLE_RHO), rng.uniform(-TABLE_ANGLE, TABLE_ANGLE)
    return float(r * np.cos(a)), float(r * np.sin(a))


def task_seed(seed: int, index: int) -> int:
    """A reproducible per-task seed (workers must not seed from their process id)."""
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])
