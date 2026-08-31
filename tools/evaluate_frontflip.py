#!/usr/bin/env python3
"""Deterministic, unassisted CPU evaluation of a deployed front-flip policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
INFER_PATH = UPSTREAM / "scripts" / "infer_policy.py"
ROLLER_SCENE = UPSTREAM / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
CONTROL_DT = 0.02
MANEUVER_PERIOD = 4.0
STAND_HEIGHT = 0.115
TAKEOFF_CLEARANCE = 0.010
MIN_LANDING_ROTATION = math.radians(300.0)
MAX_LANDING_ROTATION = math.radians(420.0)
SETTLE_SECONDS = 0.5


def load_inference_module():
    spec = importlib.util.spec_from_file_location("microduck_infer_policy", INFER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference module: {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = a
    W, X, Y, Z = b
    return np.asarray(
        [w * W - x * X - y * Y - z * Z,
         w * X + x * W + y * Z - z * Y,
         w * Y - x * Z + y * W + z * X,
         w * Z + x * Y - y * X + z * W],
        dtype=np.float64,
    )


def quaternion_delta_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the shortest signed local rotation vector between quaternions."""
    previous = previous / np.linalg.norm(previous)
    current = current / np.linalg.norm(current)
    delta = quat_mul(previous * np.asarray([1.0, -1.0, -1.0, -1.0]), current)
    delta /= np.linalg.norm(delta)
    if delta[0] < 0.0:
        delta = -delta
    length = float(np.linalg.norm(delta[1:]))
    if length < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return delta[1:] / length * (2.0 * math.atan2(length, max(0.0, float(delta[0]))))


def quat_tilt(q: np.ndarray) -> float:
    _, x, y, _ = q
    return math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def run_episode(model, data, controller, episode: int, episodes: int, metadata: dict) -> dict:
    mujoco.mj_resetData(model, data)
    qpos_adr = metadata["qpos_adr"]
    qvel_adr = metadata["qvel_adr"]
    data.qpos[qpos_adr:qpos_adr + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    # Cover the nominal rolling-start range without injecting takeoff or pitch
    # assistance.  This is deterministic across machines and invocations.
    start_vx = 0.12 + 0.16 * ((episode + 0.5) / max(1, episodes))
    data.qvel[qvel_adr] = start_vx
    for index, joint_qpos in enumerate(controller.joint_qpos_indices):
        data.qpos[joint_qpos] = controller.default_pose[index]
    data.ctrl[:] = controller.default_pose
    controller.last_action.fill(0.0)
    controller.command.fill(0.0)
    controller.vel_cmd.fill(0.0)
    if getattr(controller, "use_delay", False):
        controller.action_buffer.fill(0.0)
        controller.buffer_index = 0
    mujoco.mj_forward(model, data)

    start_xy = data.qpos[qpos_adr:qpos_adr + 2].copy()
    previous_quat = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
    support_seen = False
    takeoff = False
    landing = False
    body_contact = False
    settled_steps = 0
    forward_rotation = 0.0
    max_forward_rotation = 0.0
    offaxis_rotation = 0.0
    peak_clearance = 0.0
    max_tilt_after_landing = 0.0
    max_speed_after_landing = 0.0
    takeoff_time = None
    landing_time = None
    decimation = int(round(CONTROL_DT / model.opt.timestep))

    for step in range(int(round(MANEUVER_PERIOD / CONTROL_DT))):
        phase = step * CONTROL_DT / MANEUVER_PERIOD
        command = np.asarray(
            [math.cos(2.0 * math.pi * phase), math.sin(2.0 * math.pi * phase), 0.0],
            dtype=np.float32,
        )
        controller.vel_cmd = command
        controller.command[:3] = command
        action = controller.infer()
        controller.apply_action(action)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        grounded_sides: set[str] = set()
        hit_body = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            for geom in (geom1, geom2):
                side = metadata["wheel_geom_sides"].get(geom)
                if side is not None:
                    grounded_sides.add(side)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            hit_body |= (
                (body1 in metadata["forbidden_body_ids"] and body2 == 0)
                or (body2 in metadata["forbidden_body_ids"] and body1 == 0)
            )
        both_grounded = len(grounded_sides) == 2
        both_airborne = len(grounded_sides) == 0
        support_seen |= both_grounded
        body_contact |= hit_body

        z = float(data.qpos[qpos_adr + 2])
        clearance = max(0.0, z - STAND_HEIGHT)
        if not takeoff and support_seen and both_airborne and clearance >= TAKEOFF_CLEARANCE:
            takeoff = True
            takeoff_time = (step + 1) * CONTROL_DT
        current_quat = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
        if takeoff and not landing and both_airborne:
            delta = quaternion_delta_vector(previous_quat, current_quat)
            forward_rotation += max(0.0, float(delta[1]))
            max_forward_rotation = max(max_forward_rotation, forward_rotation)
            offaxis_rotation += math.hypot(float(delta[0]), float(delta[2]))
            peak_clearance = max(peak_clearance, clearance)
        previous_quat = current_quat

        if takeoff and not landing and both_grounded and max_forward_rotation >= MIN_LANDING_ROTATION:
            landing = True
            landing_time = (step + 1) * CONTROL_DT
        if landing:
            tilt = quat_tilt(current_quat)
            speed = float(np.linalg.norm(data.qvel[qvel_adr:qvel_adr + 6]))
            max_tilt_after_landing = max(max_tilt_after_landing, tilt)
            max_speed_after_landing = max(max_speed_after_landing, speed)
            stable = both_grounded and tilt <= math.radians(15.0) and speed <= 0.15 and not hit_body
            settled_steps = settled_steps + 1 if stable else 0

    drift = float(np.linalg.norm(data.qpos[qpos_adr:qpos_adr + 2] - start_xy))
    rotation_valid = MIN_LANDING_ROTATION <= max_forward_rotation <= MAX_LANDING_ROTATION
    settled = settled_steps >= int(round(SETTLE_SECONDS / CONTROL_DT))
    success = bool(
        takeoff and landing and rotation_valid and not body_contact and settled
        and peak_clearance >= 0.05 and offaxis_rotation <= math.radians(60.0)
        and drift <= 0.12
    )
    return {
        "start_speed_mps": start_vx,
        "takeoff_detected": takeoff,
        "landing_detected": landing,
        "body_contact": body_contact,
        "settled": settled,
        "success": success,
        "takeoff_time_s": takeoff_time,
        "landing_time_s": landing_time,
        "peak_clearance_m": peak_clearance,
        "forward_rotation_deg": math.degrees(max_forward_rotation),
        "offaxis_rotation_deg": math.degrees(offaxis_rotation),
        "horizontal_drift_m": drift,
        "max_post_landing_tilt_deg": math.degrees(max_tilt_after_landing),
        "max_post_landing_speed": max_speed_after_landing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path, help="Normalizer-aware ONNX policy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--current-limit", type=float, default=1.75)
    parser.add_argument("--wheel-friction", type=float, default=0.003)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise SystemExit(f"Policy not found: {policy_path}")

    infer = load_inference_module()
    model = mujoco.MjModel.from_xml_path(str(ROLLER_SCENE))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    if args.current_limit > 0:
        from bam.model import load_model
        torque_limit = load_model(motor_name="xl330", model="m6").kt.value * args.current_limit
        model.actuator_forcerange[:, :] = (-torque_limit, torque_limit)
        model.actuator_forcelimited[:] = 1
    wheel_body_sides: dict[int, str] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and re.match(r"^passive_.*wheel", name):
            dof = int(model.jnt_dofadr[joint_id])
            model.dof_frictionloss[dof] = args.wheel_friction
            wheel_body_sides[int(model.jnt_bodyid[joint_id])] = "left" if name.startswith("passive_L") else "right"
    wheel_geom_sides = {
        geom_id: wheel_body_sides[int(model.geom_bodyid[geom_id])]
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in wheel_body_sides
    }
    controller = infer.PolicyInference(
        model, data, walking_onnx_path=str(policy_path),
        new_cmd_obs=True, use_projected_gravity=True,
    )
    free_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    forbidden_body_ids = {
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        in {"trunk_base", "jaw_soft"}
    }
    metadata = {
        "qpos_adr": int(model.jnt_qposadr[free_joint]),
        "qvel_adr": int(model.jnt_dofadr[free_joint]),
        "wheel_geom_sides": wheel_geom_sides,
        "forbidden_body_ids": forbidden_body_ids,
    }
    rollouts = [run_episode(model, data, controller, i, args.episodes, metadata) for i in range(args.episodes)]
    rate = lambda key: float(np.mean([float(row[key]) for row in rollouts]))
    aggregate = {
        "episodes": args.episodes,
        "unassisted": True,
        "success_rate": rate("success"),
        "takeoff_rate": rate("takeoff_detected"),
        "landing_rate": rate("landing_detected"),
        "body_contact_rate": rate("body_contact"),
        "settled_rate": rate("settled"),
        "median_peak_clearance_m": percentile([row["peak_clearance_m"] for row in rollouts], 50),
        "median_forward_rotation_deg": percentile([row["forward_rotation_deg"] for row in rollouts], 50),
        "p10_forward_rotation_deg": percentile([row["forward_rotation_deg"] for row in rollouts], 10),
        "median_offaxis_rotation_deg": percentile([row["offaxis_rotation_deg"] for row in rollouts], 50),
        "median_horizontal_drift_m": percentile([row["horizontal_drift_m"] for row in rollouts], 50),
    }
    result = {
        "policy": str(policy_path),
        "control_hz": 50,
        "physics_hz": 200,
        "current_limit_a": args.current_limit,
        "wheel_frictionloss": args.wheel_friction,
        "frontflip": aggregate,
        "rollouts": rollouts,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
