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
    grasp_frac=0.4,                  # grasp probes (close the fingers, lift); the rest are motion probes
    grasp_envelope_frac=0.5,         # share of lift probes sampled from the object's reachable opposing faces
    lift=1.2,                        # lift and retraction in characteristic lengths of the subject
    segments=4, segment=0.8, segment_s=0.3,       # random motion segments, lengths of the subject
    follow_cos=0.5, follow_frac=0.4,  # the body "follows": moved >= follow_frac of the hand's way, along it
    joint_motion_min=0.1,           # minimum movement as a share of joint range
    rest_speed=0.4,                  # characteristic lengths per second below which the body is at rest
    tilt_margin=0.01,                # tolerance in the tilt measure (1 - cos) beyond observed members
    min_members=15,                  # minimum examples per discrete group
    batch=2000, min_probes=4000, stable_checks=2,    # adaptive probing: rediscover every batch, stop once the
                                     # set of *primitive-worthy* relations (see PRIMS) is unchanged for
                                     # stable_checks batches (small clusters near the threshold are ignored)
    min_precision=0.2,               # a hand-object cluster must be followed by the body at least this often
    # focused probing: after the first batch, this fraction of probes restarts from the configuration of an
    # earlier probe in which something happened (the body followed the hand, or ended in contact with the
    # other body), perturbed by focus_sd (object units, and radians for the yaws)
    focus_frac=0.5, focus_sd=0.25, focus_pool=4000,
)

# ------------------------------------------------------------------ ALGORITHM: hand probes (bep.babble.hand_probe)
# Two probes act on the bodies with the hand only: a grasp probe (approach, close, lift, then placements
# branched from the lifted state) and an interaction probe (approach, then random motions and articulations,
# each until the touching body pairs change, branched from an archive of the states reached).  The hand is
# pointed down (random yaw) or, for side_frac of the probes, sideways (horizontal approach, random yaw,
# fingers closing horizontally or vertically).
HPROBE = dict(
    side_frac=0.25,
    speed=0.25,                      # m/s of the hand (the Hand of God's speed, HOG['v_max'])
    yaw_rate=0.6,                    # rad/s scale of the hand's random turns about the vertical
    standoff=0.3,                    # a hand starts this many body lengths outside the body's bounding sphere
    overshoot=0.5,                   # an approach runs at most this many body lengths past its target point
    close_s=0.8, still_s=0.1,        # closing stops once the opening has not changed for still_s
    open_s=0.4, hold_s=0.2,
    place_branches=3,                # grasp probe: placements tried from every lifted state
    place_drop=3.0,                  # a placement descends at most this many body lengths
    move_s=0.4, grip_s=0.4,          # interaction probe: longest motion / articulation
    debounce=3,                      # control steps a contact change must persist to count
    after_len=1.0,                   # interaction probe: a motion goes on for this many body lengths (of the body it
                                     # acts on, at the hand's speed) after the touching pairs change
    depth=6, branches=4,             # interaction probe: steps, and motions branched at each step
    settle_s=0.4,                    # rest state: everything let go and left to rest this long
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
    exec_tests=32,                   # once the field converged: the real executor from this many fresh starts;
    exec_z=1.64,                     # exploring goes on while its success rate improves significantly
                                     # (one-sided two-proportion z-test at this z)
    retry_below=0.5,                  # try fresh graphs (new seeds) while the best graph fails most executor checks,
    retries=3,                        # at most this many times; the best graph is kept
    exec_timeout_s=12.0,             # (simulated seconds per test run)
    end_runs=64,                     # at the end: the executor from this many fresh starts; its successful end
                                     # states seed the primitives chained after this one
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
    stop_every=10, stop_window=3, stop_tol=0.05, min_rounds=30,   # stop when the compacted field's values at
                                     # fixed start states and at a frozen sample of graph states (every mode)
                                     # change by < stop_tol for stop_window checks in a row
    stop_sample=300,
    uncovered_tol=0.05,              # ... and either at most this share of start states is uncovered, or that
    coverage_plateau=3,              # share has not improved for this many checks (some starts are unreachable)
    metric_clip=(0.5, 2.0),          # learned metric: per mode, the fixed weights times the field's relative
                                     # rates, clipped to this range (eps keeps its meaning)
)

# ------------------------------------------------------------------ ALGORITHM: field Phi
FIELD = dict(
    rate_quantile=75.0, min_pairs=50,
    kappa_quantile=0.5, kappa_default=1.0,   # slope of Phi off the samples: the median demonstrated descent
    tau=0.05, k=16, cap_margin=3.0, min_samples=20,
    compact=False,         # prune samples the envelope already predicts (False: Phi from every graph sample)
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
    barrier_weight=1.0,              # cost per undesired contact per control step (~ 10 s of Phi per second)
    side_effects=False,              # task execution: Y's tilt and speed, kept relations, leave-alone, carry drop
    transport=False,                 # task execution: add the carried body's distance to its destination to Phi
    goal_reward=50.0,                # reward scaled by the share of rollout steps in the primitive's goal state
)

# ------------------------------------------------------------------ the Hand-of-God motor (learning's body)
HOG = dict(
    v_max=0.25,                      # m/s of the pinch point
    w_max=1.5,                       # rad/s of the hand
    grip_s=0.3,                      # a hop ends this long after the gripper is actuated
    max_reentries=4,                 # a call stalls after this many failed graph entries
    horizon=50,                      # the floating hand's planner (bep.motor.hand_mppi): control steps ahead
)

# ------------------------------------------------------------------ ALGORITHM: play (executed calls only)
PLAY = dict(
    chain_max=8,                     # executed calls chained per episode (each from where the previous ended)
    timeout_s=12.0,
    random_frac=0.3,                 # executed calls chosen uniformly; the rest where the models disagree most
    ensemble=5,                      # bootstrap success models per primitive (their disagreement = what to try)
    min_fit=10,                      # executed calls of a primitive before it has a success model
    min_per_prim=20,                 # play stops only once every primitive has at least this many calls,
    min_per_context=4,               # ... every relational context met (bep.play.play.context) this many ...
    rare_p=0.7,                      # (calls in rarer contexts are chosen first, this often)
    stop_window=30, stop_tol=0.01,   # ... and the prequential log-loss of the last stop_window calls of fitted
    min_calls=60, max_calls=400,     # primitives no longer improves on the stop_window before by stop_tol
)

# ------------------------------------------------------------------ ALGORITHM: validation on the robot (bep.validate)
VALIDATE = dict(
    tests=32,                        # Spike-MPPI runs of every primitive from fresh starts
    calls=60,                        # play calls on the robot
    weight=5.0,                      # a robot call counts as this many floating-hand play calls
)

# ------------------------------------------------------------------ ALGORITHM: decision (bep.brain)
BRAIN = dict(
    real_prior=2.0,                  # weight (in calls) of the model's success probability against reality
    real_weight=5.0,                 # a real call counts as this many play calls when the effect model is refit
    snapshot_max_states=20000,       # snapshot brain: most scene states settled per search
    min_p=0.05,                      # calls less likely than this to succeed are not considered
    # the brain (bep.brain.search): tree search over the relation graphs the effect models predict
    max_iters=5000,                  # search iterations per decision ...
    max_plan_s=3.0,                  # ... or this long (anytime)
    explore_c=0.0,                   # exploration bonus (units of a typical call's cost); 0: greedy on optimistic
                                     # values (real-time dynamic programming)
    dead_end_s=60.0,                 # extra cost of a state from which no call is available
    h_first=12,                      # literal costs: the cheapest first calls followed by every second call
)


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
