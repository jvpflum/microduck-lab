#!/usr/bin/env python3
"""Headless CPU evaluation of a deployed Roller Hop ONNX policy."""

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
HOP_PERIOD = 3.0
STAND_HEIGHT = 0.115
TARGET_CLEARANCE = 0.020


def load_inference_module():
    spec = importlib.util.spec_from_file_location("microduck_infer_policy", INFER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference module: {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quat_tilt(q: np.ndarray) -> float:
    _, x, y, _ = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path, help="Normalizer-aware ONNX policy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--current-limit", type=float, default=1.75)
    parser.add_argument("--wheel-friction", type=float, default=0.003)
    args = parser.parse_args()

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
            wheel_body_sides[int(model.jnt_bodyid[joint_id])] = (
                "left" if name.startswith("passive_L") else "right"
            )
    wheel_geom_sides = {
        geom_id: wheel_body_sides[int(model.geom_bodyid[geom_id])]
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in wheel_body_sides
    }

    controller = infer.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy_path),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    free_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[free_joint])
    qvel_adr = int(model.jnt_dofadr[free_joint])
    data.qpos[qpos_adr : qpos_adr + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    for index, joint_qpos in enumerate(controller.joint_qpos_indices):
        data.qpos[joint_qpos] = controller.default_pose[index]
    data.ctrl[:] = controller.default_pose
    mujoco.mj_forward(model, data)

    decimation = int(round(CONTROL_DT / model.opt.timestep))
    steps = int(round(HOP_PERIOD / CONTROL_DT))
    rows: list[dict[str, float]] = []
    support_seen = False
    takeoff_step: int | None = None
    landing_step: int | None = None
    previous_vz = float(data.qvel[qvel_adr + 2])
    start_xy = data.qpos[qpos_adr : qpos_adr + 2].copy()

    for step in range(steps):
        phase = (step * CONTROL_DT / HOP_PERIOD) % 1.0
        phase_command = np.asarray(
            [math.cos(2.0 * math.pi * phase), math.sin(2.0 * math.pi * phase), 0.0],
            dtype=np.float32,
        )
        # The hop uses the walking policy slot only as a phase carrier. Assign
        # it directly to keep batch evaluation logs readable (set_vel_cmd is an
        # interactive helper that prints every update).
        controller.vel_cmd = phase_command
        controller.command[:3] = phase_command
        action = controller.infer()
        controller.apply_action(action)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        grounded_sides: set[str] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            for geom in (int(contact.geom1), int(contact.geom2)):
                side = wheel_geom_sides.get(geom)
                if side is not None:
                    grounded_sides.add(side)
        both_grounded = len(grounded_sides) == 2
        both_airborne = len(grounded_sides) == 0
        support_seen |= both_grounded

        z = float(data.qpos[qpos_adr + 2])
        clearance = max(0.0, z - STAND_HEIGHT)
        if takeoff_step is None and support_seen and both_airborne and clearance >= 0.004:
            takeoff_step = step
        if takeoff_step is not None and landing_step is None and both_grounded:
            landing_step = step

        vz = float(data.qvel[qvel_adr + 2])
        rows.append(
            {
                "time_s": (step + 1) * CONTROL_DT,
                "trunk_z": z,
                "clearance": clearance if takeoff_step is not None and both_airborne else 0.0,
                "tilt": quat_tilt(data.qpos[qpos_adr + 3 : qpos_adr + 7]),
                "horizontal_speed": float(np.linalg.norm(data.qvel[qvel_adr : qvel_adr + 2])),
                "vertical_speed": vz,
                "vertical_accel": (vz - previous_vz) / CONTROL_DT,
                "both_grounded": float(both_grounded),
                "both_airborne": float(both_airborne),
            }
        )
        previous_vz = vz

    final_xy = data.qpos[qpos_adr : qpos_adr + 2].copy()
    final_window = rows[-max(1, int(round(0.4 / CONTROL_DT))) :]
    flight_rows = rows[takeoff_step : landing_step] if takeoff_step is not None else []
    peak_clearance = max((row["clearance"] for row in rows), default=0.0)
    result = {
        "policy": str(policy_path),
        "control_hz": 50,
        "physics_hz": 200,
        "current_limit_a": args.current_limit,
        "wheel_frictionloss": args.wheel_friction,
        "hop": {
            "target_clearance_m": TARGET_CLEARANCE,
            "support_seen": support_seen,
            "takeoff_detected": takeoff_step is not None,
            "landing_detected": landing_step is not None,
            "takeoff_time_s": None if takeoff_step is None else round((takeoff_step + 1) * CONTROL_DT, 4),
            "landing_time_s": None if landing_step is None else round((landing_step + 1) * CONTROL_DT, 4),
            "air_time_s": round(len(flight_rows) * CONTROL_DT, 4),
            "peak_clearance_m": peak_clearance,
            "peak_height_m": max(row["trunk_z"] for row in rows),
            "horizontal_drift_m": float(np.linalg.norm(final_xy - start_xy)),
            "flight_tilt_max_deg": float(
                np.degrees(max((row["tilt"] for row in flight_rows), default=math.pi / 2))
            ),
            "final_tilt_mean_deg": float(np.degrees(np.mean([row["tilt"] for row in final_window]))),
            "final_both_grounded_fraction": float(np.mean([row["both_grounded"] for row in final_window])),
            "final_speed_mean_mps": float(
                np.mean(
                    [math.hypot(row["horizontal_speed"], row["vertical_speed"]) for row in final_window]
                )
            ),
            "peak_landing_deceleration_mps2": float(
                max((abs(row["vertical_accel"]) for row in rows), default=0.0)
            ),
        },
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
