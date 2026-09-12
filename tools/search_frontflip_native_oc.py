#!/usr/bin/env python3
"""Native-MuJoCo full-motion optimal-control search for a skate front flip.

This is deliberately a feasibility optimizer, not an RL trainer.  It optimizes
one continuous 50 Hz actuator-target trajectory from rolling entry through
takeoff, tuck, opening, tire touchdown, and a short stable exit.  Native
MuJoCo is the source of truth because MJX-Warp contact timing does not match
the acceptance simulator closely enough for this maneuver.

The search is locked to wheel frictionloss 0.003 and a 1.75 A motor-current
limit.  It uses several phase-local evolutionary islands so launch authority,
clearance, rotation, and landing basins are preserved instead of averaged into
one conservative CEM distribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from bam.model import load_model


CONTROL_DT = 0.02
PHYSICS_DT = 0.005
WHEEL_RADIUS = 0.0175
CURRENT_LIMIT_A = 1.75
WHEEL_FRICTION = 0.003
MIN_FLIP_DEG = 300.0
MAX_FLIP_DEG = 420.0
STABLE_SECONDS = 0.25
SERVO_COUNT = 14
COMPACT_DOF = 7
KNOT_TIMES = np.asarray(
    [0.00, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64,
     0.72, 0.78, 0.80, 0.84, 0.88, 0.90, 0.96, 1.00, 1.04,
     1.08, 1.14, 1.22, 1.32, 1.40, 1.44, 1.60],
    dtype=np.float64,
)
INTERIOR_KNOTS = len(KNOT_TIMES) - 2
PARAM_DIM = INTERIOR_KNOTS * COMPACT_DOF
_TLS = threading.local()


ISLANDS = (
    {"name": "preload", "objective": "impulse", "knots": range(0, 6)},
    {"name": "push", "objective": "impulse", "knots": range(2, 9)},
    {"name": "lift", "objective": "lift", "knots": range(1, 10)},
    {"name": "pitch", "objective": "pitch", "knots": range(3, 11)},
    {"name": "tuck", "objective": "rotation", "knots": range(6, 13)},
    {"name": "opening", "objective": "landing", "knots": range(9, 15)},
    {"name": "clean", "objective": "clean", "knots": range(0, 15)},
    {"name": "goal", "objective": "goal", "knots": range(0, 15)},
)

CONTACT_REPAIR_ISLANDS = (
    {"name": "head_tuck", "objective": "repair", "knots": range(6, 12), "dofs": (5, 6)},
    {"name": "late_head", "objective": "repair", "knots": range(8, 15), "dofs": (5, 6)},
    {"name": "leg_open", "objective": "repair", "knots": range(8, 15), "dofs": (0, 1, 2, 3, 4)},
    {"name": "flight_shape", "objective": "clean", "knots": range(6, 14)},
    {"name": "late_all", "objective": "repair", "knots": range(8, 15)},
    {"name": "line_repair", "objective": "repair", "knots": range(5, 15)},
    {"name": "landing", "objective": "landing", "knots": range(7, 15)},
    {"name": "goal", "objective": "goal", "knots": range(5, 15)},
)

JAW_REPAIR_ISLANDS = (
    {"name": "early_head_tuck", "objective": "jaw_repair", "knots": range(5, 11), "dofs": (5, 6)},
    {"name": "late_head_tuck", "objective": "jaw_repair", "knots": range(8, 15), "dofs": (5, 6)},
    {"name": "tuck_shape", "objective": "jaw_repair", "knots": range(5, 13)},
    {"name": "late_shape", "objective": "jaw_repair", "knots": range(8, 15)},
    {"name": "lift_preserve", "objective": "lift", "knots": range(2, 10)},
    {"name": "pitch_preserve", "objective": "pitch", "knots": range(3, 11)},
    {"name": "contact_free", "objective": "jaw_repair", "knots": range(4, 15)},
    {"name": "complete_goal", "objective": "goal", "knots": range(5, 15)},
)

IMPULSE_REPAIR_ISLANDS = (
    {"name": "launch_head", "objective": "impulse_repair", "knots": range(3, 10), "dofs": (5, 6)},
    {"name": "early_tuck", "objective": "impulse_repair", "knots": range(4, 12)},
    {"name": "launch_legs", "objective": "impulse_repair", "knots": range(2, 10), "dofs": (0, 1, 2, 3, 4)},
    {"name": "mid_shape", "objective": "impulse_repair", "knots": range(5, 13)},
    {"name": "line_lock", "objective": "impulse_repair", "knots": range(2, 13)},
    {"name": "pitch_authority", "objective": "pitch", "knots": range(2, 10)},
    {"name": "clearance", "objective": "impulse_repair", "knots": range(2, 14)},
    {"name": "complete_goal", "objective": "goal", "knots": range(3, 15)},
)

HIRES_REPAIR_ISLANDS = (
    {"name": "launch_head_hires", "objective": "jaw_repair", "knots": range(8, 17), "dofs": (5, 6)},
    {"name": "flight_head_hires", "objective": "jaw_repair", "knots": range(10, 23), "dofs": (5, 6)},
    {"name": "launch_shape_hires", "objective": "jaw_repair", "knots": range(7, 17)},
    {"name": "flight_shape_hires", "objective": "jaw_repair", "knots": range(10, 23)},
    {"name": "one_second_window", "objective": "jaw_repair", "knots": range(12, 19)},
    {"name": "lift_hires", "objective": "lift", "knots": range(4, 15)},
    {"name": "pitch_hires", "objective": "pitch", "knots": range(6, 17)},
    {"name": "complete_goal_hires", "objective": "goal", "knots": range(6, 23)},
)

# V83 proved that late-flight timing resolution was not the remaining blocker:
# the clean family stalled near 224 degrees while the high-energy family had
# enough angular impulse but destroyed its body geometry.  This mode changes
# only the launch-side knots and freezes the proven late-flight shape.  Its
# objective rewards additional launch authority only while retaining the
# clean family’s rotation, jaw clearance, and line control.
AUTHORITY_BRIDGE_ISLANDS = (
    {"name": "preload_legs", "objective": "authority_bridge", "knots": range(0, 8), "dofs": (0, 1, 2, 3, 4)},
    {"name": "push_legs", "objective": "authority_bridge", "knots": range(3, 12), "dofs": (0, 1, 2, 3, 4)},
    {"name": "launch_head", "objective": "authority_bridge", "knots": range(3, 12), "dofs": (5, 6)},
    {"name": "launch_all", "objective": "authority_bridge", "knots": range(2, 12)},
    {"name": "snap_window", "objective": "authority_bridge", "knots": range(7, 13)},
    {"name": "lift_window", "objective": "authority_bridge", "knots": range(4, 12)},
    {"name": "line_locked_launch", "objective": "authority_bridge", "knots": range(2, 11)},
    {"name": "authority_goal", "objective": "authority_bridge", "knots": range(0, 13)},
)

# Once a candidate is clean beyond 225 degrees, early-launch exploration is no
# longer the bottleneck. These islands preserve that launch and concentrate
# resolution around the late-flight/contact window. Dense signed jaw clearance
# at 220/240/270/300 degrees creates reachable continuation milestones.
COMPLETION_REPAIR_ISLANDS = (
    {"name": "jaw_approach", "objective": "completion_repair", "knots": range(10, 18), "dofs": (5, 6)},
    {"name": "jaw_escape", "objective": "completion_repair", "knots": range(13, 23), "dofs": (5, 6)},
    {"name": "tuck_hold", "objective": "completion_repair", "knots": range(10, 20)},
    {"name": "leg_clearance", "objective": "completion_repair", "knots": range(12, 23), "dofs": (0, 1, 2, 3, 4)},
    {"name": "late_shape", "objective": "completion_repair", "knots": range(14, 23)},
    {"name": "line_locked_escape", "objective": "completion_repair", "knots": range(10, 23)},
    {"name": "open_for_tires", "objective": "landing", "knots": range(16, 23)},
    {"name": "complete_goal", "objective": "goal", "knots": range(10, 23)},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--seed-reference", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-per-island", type=int, default=192)
    parser.add_argument("--generations", type=int, default=800)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--start-speeds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--contact-repair", action="store_true",
        help="Concentrate search on preserving a proven launch while removing late body contact",
    )
    parser.add_argument(
        "--jaw-repair", action="store_true",
        help="Use dense native jaw-to-ground clearance at 180/200 degrees to repair a late jaw strike",
    )
    parser.add_argument(
        "--impulse-repair", action="store_true",
        help="Repair a high-energy launch progressively using jaw clearance at 90/120/150 degrees",
    )
    parser.add_argument(
        "--highres-repair", action="store_true",
        help="Add independent 20-40 ms control knots around takeoff and the late jaw-contact window",
    )
    parser.add_argument(
        "--authority-bridge", action="store_true",
        help="Freeze the proven late-flight shape and raise launch authority without losing clean rotation",
    )
    parser.add_argument(
        "--completion-repair", action="store_true",
        help="Repair a 225+ degree late jaw strike using dense 220/240/270/300-degree clearance",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = a
    W, X, Y, Z = b
    return np.asarray([
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    ])


def quat_delta_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous = previous / max(float(np.linalg.norm(previous)), 1e-12)
    current = current / max(float(np.linalg.norm(current)), 1e-12)
    inverse = previous * np.asarray([1.0, -1.0, -1.0, -1.0])
    delta = quat_mul(inverse, current)
    delta /= max(float(np.linalg.norm(delta)), 1e-12)
    if delta[0] < 0.0:
        delta = -delta
    length = float(np.linalg.norm(delta[1:]))
    if length < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(length, max(float(delta[0]), 0.0))
    return delta[1:] / length * angle


def validate_reference(payload: dict, path: Path) -> None:
    if not math.isclose(float(payload["wheel_frictionloss"]), WHEEL_FRICTION, abs_tol=1e-12):
        raise SystemExit(f"{path}: wheel friction must be exactly 0.003")
    if not math.isclose(float(payload["current_limit_a"]), CURRENT_LIMIT_A, abs_tol=1e-12):
        raise SystemExit(f"{path}: current limit must be exactly 1.75 A")


def resample_reference(payload: dict, time_scale: float = 1.0) -> np.ndarray:
    old_times = np.asarray(payload["knot_times_s"], dtype=np.float64)
    old_nodes = np.asarray(
        payload.get("max_rotation_full_nodes", payload["full_nodes"]), dtype=np.float64
    )
    if old_nodes.ndim != 2 or old_nodes.shape[1] != SERVO_COUNT:
        raise SystemExit(f"reference nodes must be Nx14, got {old_nodes.shape}")
    query = np.clip(KNOT_TIMES * time_scale, old_times[0], old_times[-1])
    nodes = np.stack([
        np.interp(query, old_times, old_nodes[:, actuator])
        for actuator in range(SERVO_COUNT)
    ], axis=1)
    nodes[0] = old_nodes[0]
    nodes[-1] = old_nodes[-1]
    return nodes


def compact_from_nodes(nodes: np.ndarray) -> np.ndarray:
    return np.concatenate((nodes[1:-1, :5], nodes[1:-1, 5:7]), axis=1).reshape(-1)


def expand_nodes(params: np.ndarray, default: np.ndarray) -> np.ndarray:
    compact = params.reshape(-1, INTERIOR_KNOTS, COMPACT_DOF)
    nodes = np.broadcast_to(
        default, (compact.shape[0], len(KNOT_TIMES), SERVO_COUNT)
    ).copy()
    nodes[:, 1:-1, :5] = compact[:, :, :5]
    nodes[:, 1:-1, 5:7] = compact[:, :, 5:7]
    nodes[:, 1:-1, 9:14] = -compact[:, :, :5]
    return nodes


def clamp_params(params: np.ndarray) -> np.ndarray:
    shaped = params.reshape(-1, INTERIOR_KNOTS, COMPACT_DOF)
    low = np.asarray([-0.40, -0.38, -1.50, -1.50, -1.50, -1.50, -1.50])
    high = np.asarray([0.40, 0.38, 1.50, 1.50, 1.50, 1.00, 1.50])
    np.clip(shaped, low, high, out=shaped)
    return params


def target_at(time_s: float, nodes: np.ndarray) -> np.ndarray:
    index = int(np.searchsorted(KNOT_TIMES, time_s, side="right") - 1)
    index = max(0, min(index, len(KNOT_TIMES) - 2))
    span = max(float(KNOT_TIMES[index + 1] - KNOT_TIMES[index]), 1e-9)
    alpha = float(np.clip((time_s - KNOT_TIMES[index]) / span, 0.0, 1.0))
    return (1.0 - alpha) * nodes[index] + alpha * nodes[index + 1]


def make_context(scene: str, default: list[float], duration: float) -> None:
    model = mujoco.MjModel.from_xml_path(scene)
    model.opt.timestep = PHYSICS_DT
    torque_limit = float(load_model(motor_name="xl330", model="m6").kt.value) * CURRENT_LIMIT_A
    model.actuator_forcerange[:, :] = (-torque_limit, torque_limit)
    model.actuator_forcelimited[:] = 1

    wheel_dofs: list[int] = []
    wheel_bodies: dict[int, str] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if re.match(r"^passive_.*wheel", name):
            dof = int(model.jnt_dofadr[joint_id])
            body = int(model.jnt_bodyid[joint_id])
            model.dof_frictionloss[dof] = WHEEL_FRICTION
            wheel_dofs.append(dof)
            wheel_bodies[body] = "left" if name.startswith("passive_L") else "right"
    wheel_geoms = {
        geom_id: wheel_bodies[int(model.geom_bodyid[geom_id])]
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in wheel_bodies
    }
    ground_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == 0
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    jaw_bodies = {
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "") == "jaw_soft"
    }
    jaw_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in jaw_bodies
    ]
    leg_bodies = {
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        in {"leg", "leg_2"}
    }
    leg_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in leg_bodies
    ]
    trunk_bodies = {
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        == "trunk_base"
    }
    trunk_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in trunk_bodies
    ]
    if not ground_geoms or not jaw_geoms:
        raise RuntimeError(
            f"expected ground plane and jaw_soft geometry; ground={ground_geoms}, jaw={jaw_geoms}"
        )
    forbidden_bodies = set(range(1, model.nbody)) - set(wheel_bodies)
    free_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_body = int(model.jnt_bodyid[free_joint])
    qpos_adr = int(model.jnt_qposadr[free_joint])
    qvel_adr = int(model.jnt_dofadr[free_joint])
    actuator_qpos = np.asarray([
        int(model.jnt_qposadr[int(model.actuator_trnid[index, 0])])
        for index in range(model.nu)
    ])
    if len(actuator_qpos) != SERVO_COUNT:
        raise RuntimeError(f"expected 14 actuators, found {len(actuator_qpos)}")
    _TLS.context = {
        "model": model,
        "data": mujoco.MjData(model),
        "default": np.asarray(default, dtype=np.float64),
        "duration": duration,
        "qpos_adr": qpos_adr,
        "qvel_adr": qvel_adr,
        "root_body": root_body,
        "actuator_qpos": actuator_qpos,
        "wheel_dofs": wheel_dofs,
        "wheel_geoms": wheel_geoms,
        "ground_geoms": ground_geoms,
        "jaw_geoms": jaw_geoms,
        "leg_geoms": leg_geoms,
        "trunk_geoms": trunk_geoms,
        "forbidden_bodies": forbidden_bodies,
        "torque_limit": torque_limit,
    }


def scan_contacts(context: dict) -> tuple[set[str], bool, list[str]]:
    model: mujoco.MjModel = context["model"]
    data: mujoco.MjData = context["data"]
    sides: set[str] = set()
    body_hit = False
    names: list[str] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        for geom in (geom1, geom2):
            side = context["wheel_geoms"].get(geom)
            if side is not None:
                sides.add(side)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        robot_geom = None
        robot_body = None
        if body1 == 0 and body2 in context["forbidden_bodies"]:
            robot_geom, robot_body = geom2, body2
        elif body2 == 0 and body1 in context["forbidden_bodies"]:
            robot_geom, robot_body = geom1, body1
        if robot_geom is not None:
            body_hit = True
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
            ) or f"geom-{robot_geom}"
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, robot_body
            ) or f"body-{robot_body}"
            names.append(f"{geom_name}@{body_name}")
    return sides, body_hit, sorted(set(names))


def simulate(nodes: np.ndarray, start_speed: float) -> dict:
    context = _TLS.context
    model: mujoco.MjModel = context["model"]
    data: mujoco.MjData = context["data"]
    qa, va = context["qpos_adr"], context["qvel_adr"]
    default = context["default"]
    mujoco.mj_resetData(model, data)
    data.qpos[qa:qa + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    data.qvel[va] = start_speed
    for wheel_dof in context["wheel_dofs"]:
        data.qvel[wheel_dof] = start_speed / WHEEL_RADIUS
    data.qpos[context["actuator_qpos"]] = default
    data.ctrl[:] = default
    mujoco.mj_forward(model, data)

    start_y = float(data.qpos[qa + 1])
    start_z = float(data.qpos[qa + 2])
    previous_quat = data.qpos[qa + 3:qa + 7].copy()
    support_seen = len(scan_contacts(context)[0]) == 2
    takeoff = False
    landed = False
    stable_latch = False
    body_contact = False
    pre_takeoff_body_contact = False
    first_body_contact_time = None
    first_body_contact_geoms: list[str] = []
    takeoff_time = None
    landing_time = None
    landing_tilt = math.pi
    landing_signed_pitch = 0.0
    landing_skate_ahead_of_com = 0.0
    landing_body_speed = 20.0
    landing_angular_speed = 20.0
    landing_pitch_rate = 20.0
    landing_vertical_speed = 20.0
    landing_forward_speed = 0.0
    takeoff_vz = 0.0
    takeoff_pitch_rate = 0.0
    clean_rotation = 0.0
    total_rotation = 0.0
    max_clean_rotation = 0.0
    max_total_rotation = 0.0
    offaxis = 0.0
    peak_clearance = 0.0
    peak_pitch_rate = 0.0
    stable_steps = 0
    saturation_steps = 0
    jaw_clearance_at_180 = None
    jaw_clearance_at_200 = None
    jaw_clearance_at_90 = None
    jaw_clearance_at_120 = None
    jaw_clearance_at_150 = None
    jaw_clearance_at_220 = None
    jaw_clearance_at_240 = None
    jaw_clearance_at_270 = None
    jaw_clearance_at_300 = None
    jaw_clearance_at_max_clean = 0.0
    minimum_pre_takeoff_leg_clearance = 1.0
    minimum_post_landing_trunk_clearance = 1.0
    maximum_post_landing_tilt = 0.0
    maximum_post_landing_speed = 0.0
    physics_steps = int(round(context["duration"] / PHYSICS_DT))
    stable_needed = int(round(STABLE_SECONDS / PHYSICS_DT))

    for physics_step in range(physics_steps):
        time_s = physics_step * PHYSICS_DT
        if physics_step % int(round(CONTROL_DT / PHYSICS_DT)) == 0:
            data.ctrl[:] = target_at(time_s, nodes)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
        sides, body_now, body_names = scan_contacts(context)
        both_grounded = len(sides) == 2
        both_airborne = len(sides) == 0
        support_seen |= both_grounded
        if body_now and first_body_contact_time is None:
            first_body_contact_time = (physics_step + 1) * PHYSICS_DT
            first_body_contact_geoms = body_names
        pre_takeoff_body_contact |= body_now and not takeoff
        body_contact |= body_now
        if not takeoff and context["leg_geoms"]:
            minimum_pre_takeoff_leg_clearance = min(
                minimum_pre_takeoff_leg_clearance,
                min(
                    mujoco.mj_geomDistance(model, data, ground_geom, leg_geom, 0.10, None)
                    for ground_geom in context["ground_geoms"]
                    for leg_geom in context["leg_geoms"]
                ),
            )

        vz = float(data.qvel[va + 2])
        if not takeoff and support_seen and both_airborne and vz > 0.02 and not body_now:
            takeoff = True
            takeoff_time = (physics_step + 1) * PHYSICS_DT
            takeoff_vz = vz
            takeoff_pitch_rate = max(0.0, float(data.qvel[va + 4]))

        current_quat = data.qpos[qa + 3:qa + 7].copy()
        if takeoff and not landed and both_airborne:
            delta = quat_delta_vector(previous_quat, current_quat)
            positive_pitch = max(0.0, float(delta[1]))
            total_rotation += positive_pitch
            max_total_rotation = max(max_total_rotation, total_rotation)
            if not body_contact:
                clean_rotation += positive_pitch
                if clean_rotation > max_clean_rotation:
                    max_clean_rotation = clean_rotation
                    jaw_clearance_at_max_clean = min(
                        mujoco.mj_geomDistance(model, data, ground_geom, jaw_geom, 1.0, None)
                        for ground_geom in context["ground_geoms"]
                        for jaw_geom in context["jaw_geoms"]
                    )
                clean_degrees = math.degrees(clean_rotation)
                if jaw_clearance_at_90 is None and clean_degrees >= 90.0:
                    jaw_clearance_at_90 = jaw_clearance_at_max_clean
                if jaw_clearance_at_120 is None and clean_degrees >= 120.0:
                    jaw_clearance_at_120 = jaw_clearance_at_max_clean
                if jaw_clearance_at_150 is None and clean_degrees >= 150.0:
                    jaw_clearance_at_150 = jaw_clearance_at_max_clean
                if jaw_clearance_at_180 is None and clean_degrees >= 180.0:
                    jaw_clearance_at_180 = jaw_clearance_at_max_clean
                if jaw_clearance_at_200 is None and clean_degrees >= 200.0:
                    jaw_clearance_at_200 = jaw_clearance_at_max_clean
                if jaw_clearance_at_220 is None and clean_degrees >= 220.0:
                    jaw_clearance_at_220 = jaw_clearance_at_max_clean
                if jaw_clearance_at_240 is None and clean_degrees >= 240.0:
                    jaw_clearance_at_240 = jaw_clearance_at_max_clean
                if jaw_clearance_at_270 is None and clean_degrees >= 270.0:
                    jaw_clearance_at_270 = jaw_clearance_at_max_clean
                if jaw_clearance_at_300 is None and clean_degrees >= 300.0:
                    jaw_clearance_at_300 = jaw_clearance_at_max_clean
            offaxis += math.hypot(float(delta[0]), float(delta[2]))
            peak_pitch_rate = max(peak_pitch_rate, positive_pitch / PHYSICS_DT)
            peak_clearance = max(
                peak_clearance, float(data.qpos[qa + 2]) - start_z
            )
        previous_quat = current_quat

        if (
            takeoff and not landed and both_grounded and not body_contact
            and math.radians(MIN_FLIP_DEG) <= max_total_rotation
            <= math.radians(MAX_FLIP_DEG)
        ):
            landed = True
            landing_time = (physics_step + 1) * PHYSICS_DT
            landing_up_z = float(np.clip(
                1.0 - 2.0 * (current_quat[1] * current_quat[1]
                             + current_quat[2] * current_quat[2]),
                -1.0, 1.0,
            ))
            landing_tilt = math.acos(landing_up_z)
            qw, qx, qy, qz = (float(value) for value in current_quat)
            landing_signed_pitch = math.asin(float(np.clip(
                2.0 * (qw * qy - qz * qx), -1.0, 1.0
            )))
            robot_com_x = float(data.subtree_com[context["root_body"], 0])
            skate_center_x = float(np.mean([
                data.geom_xpos[geom_id, 0]
                for geom_id in context["wheel_geoms"]
            ]))
            landing_skate_ahead_of_com = skate_center_x - robot_com_x
            landing_velocity = data.qvel[va:va + 6]
            landing_body_speed = float(np.linalg.norm(landing_velocity))
            landing_angular_speed = float(np.linalg.norm(landing_velocity[3:6]))
            landing_pitch_rate = abs(float(landing_velocity[4]))
            landing_vertical_speed = abs(float(landing_velocity[2]))
            landing_forward_speed = float(landing_velocity[0])
        if landed:
            q = current_quat
            up_z = float(np.clip(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), -1.0, 1.0))
            tilt = math.acos(up_z)
            velocity = data.qvel[va:va + 6]
            if context["trunk_geoms"]:
                minimum_post_landing_trunk_clearance = min(
                    minimum_post_landing_trunk_clearance,
                    min(
                        mujoco.mj_geomDistance(model, data, ground_geom, trunk_geom, 0.20, None)
                        for ground_geom in context["ground_geoms"]
                        for trunk_geom in context["trunk_geoms"]
                    ),
                )
            maximum_post_landing_tilt = max(maximum_post_landing_tilt, tilt)
            maximum_post_landing_speed = max(
                maximum_post_landing_speed, float(np.linalg.norm(velocity))
            )
            stable = bool(
                both_grounded and not body_now and tilt <= math.radians(15.0)
                and 0.0 <= float(velocity[0]) <= 1.5
                and abs(float(velocity[1])) <= 0.15
                and abs(float(velocity[2])) <= 0.15
                and float(np.linalg.norm(velocity[3:6])) <= 1.0
            )
            stable_steps = stable_steps + 1 if stable else 0
            stable_latch |= stable_steps >= stable_needed
        if np.any(np.abs(data.actuator_force) >= 0.995 * context["torque_limit"]):
            saturation_steps += 1

    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    clean_end_time = (
        first_body_contact_time if first_body_contact_time is not None
        else (landing_time if landing_time is not None else context["duration"])
    )
    clean_flight_time = max(0.0, clean_end_time - (takeoff_time or clean_end_time))
    ballistic_rotation = math.degrees(
        max(takeoff_pitch_rate, 0.0) * max(2.0 * takeoff_vz / 9.81, 0.0)
    )
    second_difference = np.diff(nodes, n=2, axis=0)
    smoothness = float(np.mean(second_difference * second_difference))
    current_quat = data.qpos[qa + 3:qa + 7]
    final_up_z = float(np.clip(
        1.0 - 2.0 * (current_quat[1] * current_quat[1] + current_quat[2] * current_quat[2]),
        -1.0, 1.0,
    ))
    final_tilt = math.acos(final_up_z)
    final_body_speed = float(np.linalg.norm(data.qvel[va:va + 6]))
    post_landing_body_contact_gap = (
        max(0.0, (first_body_contact_time or context["duration"]) - landing_time)
        if landing_time is not None else 0.0
    )
    return {
        "takeoff": takeoff,
        "takeoff_time_s": takeoff_time if takeoff_time is not None else context["duration"],
        "takeoff_vertical_velocity_mps": takeoff_vz,
        "takeoff_pitch_rate_rad_s": takeoff_pitch_rate,
        "ballistic_rotation_estimate_deg": ballistic_rotation,
        "peak_forward_pitch_rate_rad_s": peak_pitch_rate,
        "peak_clearance_m": max(0.0, peak_clearance),
        "clean_flight_time_s": clean_flight_time,
        "clean_forward_rotation_deg": math.degrees(max_clean_rotation),
        "jaw_clearance_at_90_deg_m": (
            jaw_clearance_at_90 if jaw_clearance_at_90 is not None else -0.05
        ),
        "jaw_clearance_at_120_deg_m": (
            jaw_clearance_at_120 if jaw_clearance_at_120 is not None else -0.05
        ),
        "jaw_clearance_at_150_deg_m": (
            jaw_clearance_at_150 if jaw_clearance_at_150 is not None else -0.05
        ),
        "jaw_clearance_at_180_deg_m": (
            jaw_clearance_at_180 if jaw_clearance_at_180 is not None else -0.05
        ),
        "jaw_clearance_at_200_deg_m": (
            jaw_clearance_at_200 if jaw_clearance_at_200 is not None else -0.05
        ),
        "jaw_clearance_at_220_deg_m": (
            jaw_clearance_at_220 if jaw_clearance_at_220 is not None else -0.05
        ),
        "jaw_clearance_at_240_deg_m": (
            jaw_clearance_at_240 if jaw_clearance_at_240 is not None else -0.05
        ),
        "jaw_clearance_at_270_deg_m": (
            jaw_clearance_at_270 if jaw_clearance_at_270 is not None else -0.05
        ),
        "jaw_clearance_at_300_deg_m": (
            jaw_clearance_at_300 if jaw_clearance_at_300 is not None else -0.05
        ),
        "jaw_clearance_at_max_clean_m": jaw_clearance_at_max_clean,
        "forward_rotation_deg": math.degrees(max_total_rotation),
        "first_body_contact_time_s": (
            first_body_contact_time if first_body_contact_time is not None
            else context["duration"]
        ),
        "first_body_contact_geoms": first_body_contact_geoms,
        "body_contact": body_contact,
        "pre_takeoff_body_contact": pre_takeoff_body_contact,
        "minimum_pre_takeoff_leg_clearance_m": minimum_pre_takeoff_leg_clearance,
        "landing": landed,
        "landing_time_s": landing_time if landing_time is not None else context["duration"],
        "landing_tilt_deg": math.degrees(landing_tilt),
        "landing_signed_pitch_deg": math.degrees(landing_signed_pitch),
        "landing_skate_ahead_of_com_m": landing_skate_ahead_of_com,
        "landing_body_speed": landing_body_speed,
        "landing_angular_speed_rad_s": landing_angular_speed,
        "landing_pitch_rate_rad_s": landing_pitch_rate,
        "landing_vertical_speed_mps": landing_vertical_speed,
        "landing_forward_speed_mps": landing_forward_speed,
        "stable": stable_latch,
        "post_landing_body_contact_gap_s": post_landing_body_contact_gap,
        "minimum_post_landing_trunk_clearance_m": minimum_post_landing_trunk_clearance,
        "maximum_post_landing_tilt_deg": math.degrees(maximum_post_landing_tilt),
        "maximum_post_landing_speed": maximum_post_landing_speed,
        "final_tilt_deg": math.degrees(final_tilt),
        "final_body_speed": final_body_speed,
        "drift_m": abs(float(data.qpos[qa + 1]) - start_y),
        "offaxis_deg": math.degrees(offaxis),
        "saturation_fraction": saturation_steps / max(physics_steps, 1),
        "smoothness": smoothness,
        "finite": finite,
    }


def evaluate_candidate(task: tuple[int, np.ndarray, tuple[float, ...]]) -> tuple[int, dict]:
    index, nodes, speeds = task
    rollouts = [simulate(nodes, speed) for speed in speeds]

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rollouts], dtype=np.float64)

    metrics = {
        "takeoff_rate": float(values("takeoff").mean()),
        "minimum_takeoff_vertical_velocity_mps": float(values("takeoff_vertical_velocity_mps").min()),
        "minimum_takeoff_pitch_rate_rad_s": float(values("takeoff_pitch_rate_rad_s").min()),
        "minimum_ballistic_rotation_estimate_deg": float(values("ballistic_rotation_estimate_deg").min()),
        "minimum_peak_pitch_rate_rad_s": float(values("peak_forward_pitch_rate_rad_s").min()),
        "minimum_clearance_m": float(values("peak_clearance_m").min()),
        "minimum_clean_flight_time_s": float(values("clean_flight_time_s").min()),
        "minimum_clean_rotation_deg": float(values("clean_forward_rotation_deg").min()),
        "mean_clean_rotation_deg": float(values("clean_forward_rotation_deg").mean()),
        "minimum_jaw_clearance_at_90_deg_m": float(values("jaw_clearance_at_90_deg_m").min()),
        "minimum_jaw_clearance_at_120_deg_m": float(values("jaw_clearance_at_120_deg_m").min()),
        "minimum_jaw_clearance_at_150_deg_m": float(values("jaw_clearance_at_150_deg_m").min()),
        "minimum_jaw_clearance_at_180_deg_m": float(values("jaw_clearance_at_180_deg_m").min()),
        "minimum_jaw_clearance_at_200_deg_m": float(values("jaw_clearance_at_200_deg_m").min()),
        "minimum_jaw_clearance_at_220_deg_m": float(values("jaw_clearance_at_220_deg_m").min()),
        "minimum_jaw_clearance_at_240_deg_m": float(values("jaw_clearance_at_240_deg_m").min()),
        "minimum_jaw_clearance_at_270_deg_m": float(values("jaw_clearance_at_270_deg_m").min()),
        "minimum_jaw_clearance_at_300_deg_m": float(values("jaw_clearance_at_300_deg_m").min()),
        "minimum_jaw_clearance_at_max_clean_m": float(values("jaw_clearance_at_max_clean_m").min()),
        "minimum_full_rotation_deg": float(values("forward_rotation_deg").min()),
        "body_contact_rate": float(values("body_contact").mean()),
        "pre_takeoff_body_contact_rate": float(values("pre_takeoff_body_contact").mean()),
        "minimum_pre_takeoff_leg_clearance_m": float(values("minimum_pre_takeoff_leg_clearance_m").min()),
        "first_body_contact_time_s": float(values("first_body_contact_time_s").min()),
        "landing_rate": float(values("landing").mean()),
        "maximum_landing_time_s": float(values("landing_time_s").max()),
        "maximum_landing_tilt_deg": float(values("landing_tilt_deg").max()),
        "minimum_landing_signed_pitch_deg": float(values("landing_signed_pitch_deg").min()),
        "maximum_landing_signed_pitch_deg": float(values("landing_signed_pitch_deg").max()),
        "minimum_landing_skate_ahead_of_com_m": float(values("landing_skate_ahead_of_com_m").min()),
        "maximum_landing_skate_ahead_of_com_m": float(values("landing_skate_ahead_of_com_m").max()),
        "maximum_landing_body_speed": float(values("landing_body_speed").max()),
        "maximum_landing_angular_speed_rad_s": float(values("landing_angular_speed_rad_s").max()),
        "maximum_landing_pitch_rate_rad_s": float(values("landing_pitch_rate_rad_s").max()),
        "maximum_landing_vertical_speed_mps": float(values("landing_vertical_speed_mps").max()),
        "minimum_landing_forward_speed_mps": float(values("landing_forward_speed_mps").min()),
        "maximum_landing_forward_speed_mps": float(values("landing_forward_speed_mps").max()),
        "stable_rate": float(values("stable").mean()),
        "minimum_post_landing_body_contact_gap_s": float(values("post_landing_body_contact_gap_s").min()),
        "minimum_post_landing_trunk_clearance_m": float(values("minimum_post_landing_trunk_clearance_m").min()),
        "maximum_post_landing_tilt_deg": float(values("maximum_post_landing_tilt_deg").max()),
        "maximum_post_landing_speed": float(values("maximum_post_landing_speed").max()),
        "maximum_final_tilt_deg": float(values("final_tilt_deg").max()),
        "maximum_final_body_speed": float(values("final_body_speed").max()),
        "maximum_drift_m": float(values("drift_m").max()),
        "maximum_offaxis_deg": float(values("offaxis_deg").max()),
        "maximum_saturation_fraction": float(values("saturation_fraction").max()),
        "smoothness": float(values("smoothness").mean()),
        "finite_rate": float(values("finite").mean()),
        "first_body_contact_geoms": sorted({
            name for row in rollouts for name in row["first_body_contact_geoms"]
        }),
        "rollouts": rollouts,
    }
    return index, metrics


def score(metrics: dict, objective: str, duration: float) -> float:
    if (
        metrics["finite_rate"] < 1.0
        or metrics["pre_takeoff_body_contact_rate"] > 0.0
        or metrics["takeoff_rate"] < 1.0
    ):
        return -1.0e9
    vz = metrics["minimum_takeoff_vertical_velocity_mps"]
    omega = metrics["minimum_takeoff_pitch_rate_rad_s"]
    peak_omega = metrics["minimum_peak_pitch_rate_rad_s"]
    ballistic = metrics["minimum_ballistic_rotation_estimate_deg"]
    clearance = metrics["minimum_clearance_m"]
    rotation = metrics["minimum_clean_rotation_deg"]
    clean_time = metrics["minimum_clean_flight_time_s"]
    contact_time = metrics["first_body_contact_time_s"]
    line_penalty = 500.0 * metrics["maximum_drift_m"] + 1.5 * metrics["maximum_offaxis_deg"]
    smooth_penalty = 8.0 * metrics["smoothness"]
    authority = 420.0 * vz + 24.0 * omega + 8.0 * peak_omega + 2.0 * ballistic
    flight = 1400.0 * clearance + 180.0 * clean_time + 40.0 * contact_time / duration
    if objective == "impulse":
        value = 2.0 * rotation + 2.0 * authority + flight
    elif objective == "lift":
        value = 2.0 * rotation + 650.0 * vz + 2400.0 * clearance + 260.0 * clean_time
    elif objective == "pitch":
        value = 4.0 * rotation + 50.0 * omega + 15.0 * peak_omega + flight
    elif objective == "rotation":
        value = 12.0 * rotation + authority + flight
    elif objective == "landing":
        value = 10.0 * rotation + authority + flight
        value += 2500.0 * metrics["landing_rate"] + 5000.0 * metrics["stable_rate"]
    elif objective == "clean":
        value = 15.0 * rotation + authority + 2.0 * flight
        value -= 300.0 * metrics["body_contact_rate"]
    elif objective == "repair":
        # Once a trajectory already rotates past 180 degrees, the useful
        # frontier is delaying/removing the first non-skate contact without
        # surrendering launch authority.  A contact-free rollout receives a
        # discontinuous bonus because it opens a qualitatively new basin.
        value = 18.0 * rotation + authority + 2.0 * flight
        value += 900.0 * contact_time / duration
        value += 5000.0 * (1.0 - metrics["body_contact_rate"])
        value += 2500.0 * metrics["landing_rate"] + 5000.0 * metrics["stable_rate"]
    elif objective == "jaw_repair":
        # The late-failure frontier already has enough angular momentum for a
        # flip, but its jaw reaches the floor around 210 degrees.  Signed
        # geom distance at fixed rotation landmarks supplies a dense signal
        # for tucking the head away from the floor before contact.  Preserve
        # the first 180 degrees, then strongly reward additional clean motion.
        preserved = min(rotation, 180.0)
        continuation = max(0.0, min(rotation, 360.0) - 180.0)
        jaw180 = metrics["minimum_jaw_clearance_at_180_deg_m"]
        jaw200 = metrics["minimum_jaw_clearance_at_200_deg_m"]
        value = 12.0 * preserved + 42.0 * continuation + authority + flight
        value += 8000.0 * float(np.clip(jaw180, -0.03, 0.05))
        value += 14000.0 * float(np.clip(jaw200, -0.03, 0.05))
        value += 1200.0 * contact_time / duration
        value += 7000.0 * (1.0 - metrics["body_contact_rate"])
        value += 3500.0 * metrics["landing_rate"] + 8000.0 * metrics["stable_rate"]
    elif objective == "impulse_repair":
        # A second continuation path starts from the higher-energy launch
        # family.  It cannot yet reach the 180/200-degree landmarks, so use
        # earlier signed-distance waypoints while retaining ballistic launch
        # authority.  This avoids the all-or-nothing reward cliff that caused
        # previous high-impulse trajectories to be discarded.
        jaw90 = metrics["minimum_jaw_clearance_at_90_deg_m"]
        jaw120 = metrics["minimum_jaw_clearance_at_120_deg_m"]
        jaw150 = metrics["minimum_jaw_clearance_at_150_deg_m"]
        value = 25.0 * min(rotation, 120.0) + 55.0 * max(0.0, min(rotation, 240.0) - 120.0)
        value += 3.0 * authority + 2.0 * flight
        value += 5000.0 * float(np.clip(jaw90, -0.03, 0.05))
        value += 9000.0 * float(np.clip(jaw120, -0.03, 0.05))
        value += 12000.0 * float(np.clip(jaw150, -0.03, 0.05))
        value += 1200.0 * contact_time / duration
        value += 7000.0 * (1.0 - metrics["body_contact_rate"])
        value += 3500.0 * metrics["landing_rate"] + 8000.0 * metrics["stable_rate"]
    elif objective == "authority_bridge":
        # Do not let a high-energy crash beat the 224-degree clean family.
        # Within the protected basin, favor vertical velocity and forward
        # pitch authority so the next late-flight repair starts with enough
        # energy to cross 300 degrees.
        deficit = max(0.0, 205.0 - rotation)
        jaw180 = metrics["minimum_jaw_clearance_at_180_deg_m"]
        jaw200 = metrics["minimum_jaw_clearance_at_200_deg_m"]
        value = 20.0 * min(rotation, 230.0) + 5.0 * authority + 2.0 * flight
        value -= 600.0 * deficit
        value += 5000.0 * float(np.clip(jaw180, -0.03, 0.05))
        value += 8000.0 * float(np.clip(jaw200, -0.03, 0.05))
        value -= 2500.0 * max(0.0, metrics["maximum_drift_m"] - 0.03)
        value -= 80.0 * max(0.0, metrics["maximum_offaxis_deg"] - 15.0)
    elif objective == "completion_repair":
        preserved = min(rotation, 220.0)
        continuation = max(0.0, min(rotation, 360.0) - 220.0)
        value = 16.0 * preserved + 85.0 * continuation + authority + 2.0 * flight
        for landmark, weight in (
            (220, 10000.0), (240, 18000.0),
            (270, 24000.0), (300, 32000.0),
        ):
            jaw = metrics[f"minimum_jaw_clearance_at_{landmark}_deg_m"]
            value += weight * float(np.clip(jaw, -0.03, 0.05))
        value += 1800.0 * contact_time / duration
        value += 12000.0 * (1.0 - metrics["body_contact_rate"])
        value += 5000.0 * metrics["landing_rate"] + 12000.0 * metrics["stable_rate"]
        value -= 3500.0 * max(0.0, metrics["maximum_drift_m"] - 0.05)
        value -= 100.0 * max(0.0, metrics["maximum_offaxis_deg"] - 20.0)
    elif objective == "goal":
        value = 14.0 * min(rotation, 360.0) + authority + flight
        value += 3500.0 * metrics["landing_rate"] + 8000.0 * metrics["stable_rate"]
        value -= 10.0 * max(0.0, metrics["minimum_full_rotation_deg"] - 390.0)
    else:
        raise ValueError(objective)
    return float(value - line_penalty - smooth_penalty)


def phase_mask(spec: dict) -> np.ndarray:
    mask = np.zeros((INTERIOR_KNOTS, COMPACT_DOF), dtype=bool)
    dofs = list(spec.get("dofs", range(COMPACT_DOF)))
    mask[np.ix_(list(spec["knots"]), dofs)] = True
    return mask.reshape(-1)


def record(generation: int, island: str, index: int, params: np.ndarray,
           nodes: np.ndarray, metrics: dict) -> dict:
    return {
        "generation": generation,
        "island": island,
        "candidate": index,
        "backend": "native-mujoco-full-motion-oc",
        "wheel_frictionloss": WHEEL_FRICTION,
        "current_limit_a": CURRENT_LIMIT_A,
        "control_hz": round(1.0 / CONTROL_DT),
        "physics_hz": round(1.0 / PHYSICS_DT),
        "knot_times_s": KNOT_TIMES.tolist(),
        "params": params[index].tolist(),
        "full_nodes": nodes[index].tolist(),
        **metrics,
    }


def brief(item: dict | None) -> dict | None:
    if item is None:
        return None
    hidden = {"params", "full_nodes", "rollouts", "knot_times_s"}
    return {key: value for key, value in item.items() if key not in hidden}


def clean_key(item: dict) -> tuple[float, ...]:
    return (
        float(item["minimum_clean_rotation_deg"]),
        float(item["minimum_clean_flight_time_s"]),
        float(item["minimum_clearance_m"]),
        -float(item["maximum_offaxis_deg"]),
        -float(item["maximum_drift_m"]),
    )


def impulse_key(item: dict) -> tuple[float, ...]:
    return (
        float(item["minimum_ballistic_rotation_estimate_deg"]),
        float(item["minimum_takeoff_pitch_rate_rad_s"]),
        float(item["minimum_takeoff_vertical_velocity_mps"]),
        float(item["minimum_clean_rotation_deg"]),
    )


def authority_key(item: dict) -> tuple[float, ...]:
    qualified = (
        float(item["minimum_clean_rotation_deg"]) >= 205.0
        and float(item["maximum_drift_m"]) <= 0.04
        and float(item["maximum_offaxis_deg"]) <= 20.0
    )
    authority = (
        420.0 * float(item["minimum_takeoff_vertical_velocity_mps"])
        + 24.0 * float(item["minimum_takeoff_pitch_rate_rad_s"])
        + 8.0 * float(item["minimum_peak_pitch_rate_rad_s"])
        + 2.0 * float(item["minimum_ballistic_rotation_estimate_deg"])
    )
    return (
        float(qualified),
        authority if qualified else float(item["minimum_clean_rotation_deg"]),
        float(item["minimum_clean_rotation_deg"]),
        float(item["minimum_jaw_clearance_at_200_deg_m"]),
    )


def goal_key(item: dict) -> tuple[float, ...]:
    return (
        float(item["stable_rate"]),
        float(item["landing_rate"]),
        min(360.0, float(item["minimum_clean_rotation_deg"])),
        -float(item["body_contact_rate"]),
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.population_per_island = min(args.population_per_island, 64)
        args.generations = min(args.generations, 2)
        args.workers = min(args.workers, 12)
    if args.population_per_island < 32:
        raise SystemExit("population per island must be at least 32")
    reference = json.loads(args.reference.read_text())
    validate_reference(reference, args.reference)
    seed_payloads = [reference]
    for path in args.seed_reference:
        payload = json.loads(path.read_text())
        validate_reference(payload, path)
        seed_payloads.append(payload)

    base_nodes = resample_reference(reference)
    default = base_nodes[0].copy()
    seed_params = [compact_from_nodes(base_nodes)]
    for payload in seed_payloads[1:]:
        seed_params.append(compact_from_nodes(resample_reference(payload)))
    # Timing variants make the optimizer test sharper and slower V69 launches
    # without adding a discontinuous time-warp variable to every candidate.
    if not (
        args.contact_repair or args.jaw_repair
        or args.impulse_repair or args.highres_repair or args.authority_bridge
        or args.completion_repair
    ):
        seed_params.extend([
            compact_from_nodes(resample_reference(reference, 0.88)),
            compact_from_nodes(resample_reference(reference, 1.12)),
        ])

    scene = (
        Path(__file__).resolve().parents[1]
        / "upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_rollers.xml"
    )
    if not scene.exists():
        raise SystemExit(f"scene not found: {scene}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.completion_repair:
        islands = COMPLETION_REPAIR_ISLANDS
    elif args.authority_bridge:
        islands = AUTHORITY_BRIDGE_ISLANDS
    elif args.highres_repair:
        islands = HIRES_REPAIR_ISLANDS
    elif args.impulse_repair:
        islands = IMPULSE_REPAIR_ISLANDS
    elif args.jaw_repair:
        islands = JAW_REPAIR_ISLANDS
    elif args.contact_repair:
        islands = CONTACT_REPAIR_ISLANDS
    else:
        islands = ISLANDS
    island_count = len(islands)
    masks = np.stack([phase_mask(spec) for spec in islands])
    means = np.stack([seed_params[index % len(seed_params)] for index in range(island_count)])
    stds = np.full((island_count, PARAM_DIM), 0.20, dtype=np.float64)
    stds.reshape(island_count, INTERIOR_KNOTS, COMPACT_DOF)[:, :, :2] = 0.08
    stds.reshape(island_count, INTERIOR_KNOTS, COMPACT_DOF)[:, :, 5:7] = 0.30
    if (
        args.contact_repair or args.jaw_repair or args.impulse_repair
        or args.highres_repair or args.authority_bridge or args.completion_repair
    ):
        stds *= 0.45
    minimum_std = np.full(PARAM_DIM, 0.025, dtype=np.float64)
    minimum_std.reshape(INTERIOR_KNOTS, COMPACT_DOF)[:, :2] = 0.015
    minimum_std.reshape(INTERIOR_KNOTS, COMPACT_DOF)[:, 5:7] = 0.040
    best_clean = None
    best_impulse = None
    best_authority = None
    best_goal = None
    archive: dict[str, dict] = {}
    start_generation = 0

    if args.resume is not None and args.resume.exists():
        state = json.loads(args.resume.read_text())
        means = np.asarray(state["means"], dtype=np.float64)
        stds = np.asarray(state["stds"], dtype=np.float64)
        rng.bit_generator.state = state["rng_state"]
        best_clean = state.get("best_clean")
        best_impulse = state.get("best_impulse")
        best_authority = state.get("best_authority")
        best_goal = state.get("best_goal")
        archive = state.get("archive", {})
        start_generation = int(state["generation"]) + 1

    total_population = island_count * args.population_per_island
    print(json.dumps({
        "event": "start",
        "backend": "native-mujoco-full-motion-oc",
        "scene": str(scene),
        "mode": (
            "completion-repair" if args.completion_repair
            else "authority-bridge" if args.authority_bridge
            else "highres-repair" if args.highres_repair
            else "impulse-repair" if args.impulse_repair
            else "jaw-repair" if args.jaw_repair
            else "contact-repair" if args.contact_repair
            else "full-discovery"
        ),
        "islands": [dict(name=x["name"], objective=x["objective"]) for x in islands],
        "population_per_island": args.population_per_island,
        "total_population": total_population,
        "generations": args.generations,
        "workers": args.workers,
        "duration_s": args.duration,
        "start_speeds_mps": args.start_speeds,
        "wheel_frictionloss": WHEEL_FRICTION,
        "current_limit_a": CURRENT_LIMIT_A,
        "parameter_count": PARAM_DIM,
    }, sort_keys=True), flush=True)

    # Native MuJoCo's Python calls do not release enough of the GIL for a
    # thread pool to use all cores during contact-heavy rollouts.  Independent
    # worker processes provide real CPU parallelism while keeping one reusable
    # model/data pair per worker.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=make_context,
        initargs=(str(scene), default.tolist(), args.duration),
    ) as executor:
        for generation in range(start_generation, args.generations):
            started = time.monotonic()
            groups = []
            champion_params = [
                np.asarray(item["params"], dtype=np.float64)
                for item in (best_clean, best_impulse, best_authority, best_goal)
                if item is not None
            ]
            archive_values = list(archive.values())
            for island_index, spec in enumerate(islands):
                count = args.population_per_island
                inactive_scale = 0.0 if args.authority_bridge else 0.0015 if (
                    args.contact_repair or args.jaw_repair or args.impulse_repair
                    or args.highres_repair or args.completion_repair
                ) else 0.006
                active_scale = np.where(
                    masks[island_index], stds[island_index], inactive_scale
                )
                group = means[island_index] + rng.standard_normal((count, PARAM_DIM)) * active_scale
                group[0] = means[island_index]
                cursor = 1
                for seed in seed_params + champion_params:
                    if cursor >= count:
                        break
                    group[cursor] = seed
                    cursor += 1
                # Differential proposals splice genuinely different launch and
                # landing families rather than collapsing them into one mean.
                if len(archive_values) >= 3:
                    for local in range(min(count // 6, len(archive_values))):
                        a, b, c = rng.choice(len(archive_values), 3, replace=False)
                        pa = np.asarray(archive_values[a]["params"])
                        pb = np.asarray(archive_values[b]["params"])
                        pc = np.asarray(archive_values[c]["params"])
                        proposal = pa + rng.uniform(0.35, 0.85) * (pb - pc)
                        group[-1 - local] = np.where(
                            masks[island_index], proposal, group[-1 - local]
                        )
                groups.append(group)
            candidates = clamp_params(np.concatenate(groups, axis=0))
            nodes = expand_nodes(candidates, default)
            tasks = (
                (index, nodes[index], tuple(args.start_speeds))
                for index in range(total_population)
            )
            rows: list[dict | None] = [None] * total_population
            for index, metrics in executor.map(evaluate_candidate, tasks, chunksize=4):
                rows[index] = metrics
            if any(row is None for row in rows):
                raise RuntimeError("incomplete native population")
            metrics_rows: list[dict] = [row for row in rows if row is not None]

            island_reports = []
            for island_index, spec in enumerate(islands):
                begin = island_index * args.population_per_island
                end = begin + args.population_per_island
                island_scores = np.asarray([
                    score(metrics_rows[index], spec["objective"], args.duration)
                    for index in range(begin, end)
                ])
                elite_count = max(12, args.population_per_island // 10)
                elite_local = np.argpartition(island_scores, -elite_count)[-elite_count:]
                elite = candidates[begin:end][elite_local]
                elite_mean = elite.mean(axis=0)
                elite_std = np.maximum(elite.std(axis=0), minimum_std)
                active = masks[island_index]
                means[island_index] = np.where(
                    active,
                    0.72 * means[island_index] + 0.28 * elite_mean,
                    0.97 * means[island_index] + 0.03 * elite_mean,
                )
                stds[island_index] = np.maximum(
                    np.where(
                        active,
                        0.80 * stds[island_index] + 0.20 * elite_std,
                        stds[island_index],
                    ),
                    minimum_std,
                )
                winner = begin + int(np.argmax(island_scores))
                item = record(
                    generation, spec["name"], winner, candidates, nodes,
                    metrics_rows[winner],
                )
                island_reports.append({
                    "island": spec["name"],
                    "objective": spec["objective"],
                    "score": float(island_scores[winner - begin]),
                    "winner": brief(item),
                })

            for index, metrics in enumerate(metrics_rows):
                if (
                    metrics["finite_rate"] < 1.0
                    or metrics["pre_takeoff_body_contact_rate"] > 0.0
                    or metrics["takeoff_rate"] < 1.0
                ):
                    continue
                island_name = islands[index // args.population_per_island]["name"]
                item = record(generation, island_name, index, candidates, nodes, metrics)
                if best_clean is None or clean_key(item) > clean_key(best_clean):
                    best_clean = item
                    atomic_json(output / "best-clean.json", item)
                if best_impulse is None or impulse_key(item) > impulse_key(best_impulse):
                    best_impulse = item
                    atomic_json(output / "best-impulse.json", item)
                if best_authority is None or authority_key(item) > authority_key(best_authority):
                    best_authority = item
                    atomic_json(output / "best-authority.json", item)
                if best_goal is None or goal_key(item) > goal_key(best_goal):
                    best_goal = item
                    atomic_json(output / "best-goal.json", item)
                rotation_bin = int(max(0.0, metrics["minimum_clean_rotation_deg"]) // 20.0)
                vz_bin = int(max(0.0, metrics["minimum_takeoff_vertical_velocity_mps"]) // 0.10)
                omega_bin = int(max(0.0, metrics["minimum_takeoff_pitch_rate_rad_s"]) // 3.0)
                key = f"r{rotation_bin:02d}-v{vz_bin:02d}-w{omega_bin:02d}"
                old = archive.get(key)
                if old is None or goal_key(item) > goal_key(old):
                    archive[key] = item

            # Keep the archive bounded while retaining its most useful cells.
            if len(archive) > 384:
                ordered = sorted(
                    archive.items(), key=lambda pair: clean_key(pair[1]), reverse=True
                )[:384]
                archive = dict(ordered)
            state = {
                "generation": generation,
                "means": means.tolist(),
                "stds": stds.tolist(),
                "rng_state": rng.bit_generator.state,
                "best_clean": best_clean,
                "best_impulse": best_impulse,
                "best_authority": best_authority,
                "best_goal": best_goal,
                "archive": archive,
            }
            atomic_json(output / "search-state.json", state)
            atomic_json(output / "latest-generation.json", {
                "generation": generation,
                "seconds": time.monotonic() - started,
                "islands": island_reports,
                "best_clean": brief(best_clean),
                "best_impulse": brief(best_impulse),
                "best_authority": brief(best_authority),
                "best_goal": brief(best_goal),
                "archive_cells": len(archive),
            })
            print(json.dumps({
                "generation": generation,
                "seconds": round(time.monotonic() - started, 3),
                "archive_cells": len(archive),
                "best_clean": brief(best_clean),
                "best_impulse": brief(best_impulse),
                "best_authority": brief(best_authority),
                "best_goal": brief(best_goal),
            }, sort_keys=True), flush=True)

            if (
                best_goal is not None
                and best_goal["minimum_clean_rotation_deg"] >= MIN_FLIP_DEG
                and best_goal["landing_rate"] >= 1.0
                and best_goal["stable_rate"] >= 1.0
                and best_goal["body_contact_rate"] <= 0.0
            ):
                atomic_json(output / "COMPLETED.json", best_goal)
                print(json.dumps({"event": "completed", "champion": brief(best_goal)}), flush=True)
                break


if __name__ == "__main__":
    main()
