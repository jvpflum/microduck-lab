#!/usr/bin/env python3
"""Headless CPU evaluation of a deployed MicroDuck swizzle ONNX policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
INFER_PATH = UPSTREAM / "scripts" / "infer_policy.py"
ROLLER_SCENE = UPSTREAM / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"


def load_inference_module():
    spec = importlib.util.spec_from_file_location("microduck_infer_policy", INFER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference module: {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    command_x: float
    heading_error: float = 0.0


CONTROL_DT = 0.02

PHASES = (
    Phase("settle", 2.0, 0.0),
    Phase("forward", 8.0, 0.3),
    Phase("stop_forward", 4.0, 0.0),
    Phase("reverse", 8.0, -0.3),
    Phase("stop_reverse", 4.0, 0.0),
    Phase("heading_left", 6.0, 0.2, 0.3),
    Phase("heading_right", 6.0, 0.2, -0.3),
)


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_tilt(q: np.ndarray) -> float:
    w, x, y, z = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    forward = values("forward_speed")
    lateral = values("lateral_speed")
    trunk_z = values("trunk_z")
    tilt = values("tilt")
    separation = values("skate_separation")
    actions = np.asarray([row["action_acc"] for row in rows], dtype=np.float64)
    grounded = values("both_grounded")
    wheel_speed = values("mean_abs_wheel_speed")
    command_x = values("command_x")
    yaw = np.unwrap(values("yaw"))
    cycles = 0
    separation_range = float(np.ptp(separation))
    # Ignore numerical/contact chatter. A real outward/inward stroke must move
    # the ankles by at least 5 mm and traverse most of that measured range.
    if separation_range >= 0.005:
        low = float(separation.min() + 0.25 * separation_range)
        high = float(separation.min() + 0.75 * separation_range)
        expanded = False
        for sample in separation:
            if not expanded and sample >= high:
                expanded = True
            elif expanded and sample <= low:
                cycles += 1
                expanded = False

    duration = float(rows[-1]["phase_time"] if rows else 0.0)
    tracking_rmse = float(np.sqrt(np.mean(np.square(forward - command_x))))
    nonzero_command = abs(float(command_x[-1])) > 1e-6
    direction = math.copysign(1.0, float(command_x[-1])) if nonzero_command else 0.0
    response_time = None
    if nonzero_command:
        threshold = 0.8 * abs(float(command_x[-1]))
        reached = np.flatnonzero(direction * forward >= threshold)
        if reached.size:
            response_time = float((reached[0] + 1) * CONTROL_DT)

    stop_time = None
    if not nonzero_command:
        quiet = np.abs(forward) <= 0.05
        window = max(1, int(round(0.5 / CONTROL_DT)))
        for index in range(0, max(0, len(quiet) - window + 1)):
            if bool(np.all(quiet[index : index + window])):
                stop_time = float((index + 1) * CONTROL_DT)
                break

    return {
        "duration_s": duration,
        "command_x_mps": float(command_x[-1]),
        "mean_forward_speed_mps": float(forward.mean()),
        "mean_abs_forward_speed_mps": float(np.abs(forward).mean()),
        "peak_abs_forward_speed_mps": float(np.abs(forward).max()),
        "end_abs_forward_speed_mps": float(np.abs(forward[-min(len(forward), 25) :]).mean()),
        "forward_distance_m": float(np.sum(forward) * CONTROL_DT),
        "tracking_rmse_mps": tracking_rmse,
        "command_direction_fraction": float(np.mean(direction * forward > 0.02)) if nonzero_command else 0.0,
        "response_time_80pct_s": response_time,
        "stop_time_below_0_05_mps_s": stop_time,
        "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])) if len(yaw) > 1 else 0.0,
        "mean_yaw_rate_rad_s": float((yaw[-1] - yaw[0]) / max(CONTROL_DT, duration - CONTROL_DT)) if len(yaw) > 1 else 0.0,
        "mean_abs_lateral_speed_mps": float(np.abs(lateral).mean()),
        "trunk_height_mean_m": float(trunk_z.mean()),
        "trunk_height_std_m": float(trunk_z.std()),
        "trunk_height_peak_to_peak_m": float(np.ptp(trunk_z)),
        "tilt_rms_deg": float(np.degrees(np.sqrt(np.mean(tilt**2)))),
        "tilt_max_deg": float(np.degrees(tilt.max())),
        "both_blades_grounded_fraction": float(grounded.mean()),
        "skate_separation_mean_m": float(separation.mean()),
        "skate_separation_peak_to_peak_m": separation_range,
        "estimated_swizzle_cycles": cycles,
        "mean_abs_wheel_speed_rad_s": float(wheel_speed.mean()),
        "mean_action_acceleration": float(actions.mean()),
        "max_action_acceleration": float(actions.max()),
    }


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

        kt = load_model(motor_name="xl330", model="m6").kt.value
        torque_limit = kt * args.current_limit
        model.actuator_forcerange[:, :] = (-torque_limit, torque_limit)
        model.actuator_forcelimited[:] = 1

    wheel_dofs: list[int] = []
    wheel_body_sides: dict[int, str] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and re.match(r"^passive_.*wheel", name):
            dof = int(model.jnt_dofadr[joint_id])
            wheel_dofs.append(dof)
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

    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_v1")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_v1")
    if left_body < 0 or right_body < 0:
        raise SystemExit("Roller model is missing left/right ankle bodies")

    control_dt = CONTROL_DT
    decimation = int(round(control_dt / model.opt.timestep))
    all_results: dict[str, dict[str, float]] = {}
    previous_action = np.zeros(model.nu, dtype=np.float32)
    previous_delta = np.zeros(model.nu, dtype=np.float32)

    for phase in PHASES:
        controller.set_vel_cmd(phase.command_x, 0.0, phase.heading_error)
        rows: list[dict[str, float]] = []
        steps = int(round(phase.duration_s / control_dt))
        for step in range(steps):
            action = controller.infer()
            controller.apply_action(action)
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            quat = data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
            yaw = quat_to_yaw(quat)
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            vx, vy = float(data.qvel[qvel_adr]), float(data.qvel[qvel_adr + 1])
            forward_speed = cos_yaw * vx + sin_yaw * vy
            lateral_speed = -sin_yaw * vx + cos_yaw * vy

            left_xy = data.xpos[left_body, :2]
            right_xy = data.xpos[right_body, :2]
            separation = float(np.linalg.norm(left_xy - right_xy))

            grounded_sides: set[str] = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                for geom in (int(contact.geom1), int(contact.geom2)):
                    side = wheel_geom_sides.get(geom)
                    if side is not None:
                        grounded_sides.add(side)

            delta = action - previous_action
            action_acc = float(np.mean(np.abs(delta - previous_delta)))
            previous_action = action.copy()
            previous_delta = delta.copy()
            rows.append(
                {
                    "phase_time": (step + 1) * control_dt,
                    "command_x": phase.command_x,
                    "forward_speed": forward_speed,
                    "lateral_speed": lateral_speed,
                    "trunk_z": float(data.qpos[qpos_adr + 2]),
                    "tilt": quat_tilt(quat),
                    "yaw": yaw,
                    "skate_separation": separation,
                    "both_grounded": float(len(grounded_sides) == 2),
                    "mean_abs_wheel_speed": float(
                        np.mean(np.abs(data.qvel[wheel_dofs])) if wheel_dofs else 0.0
                    ),
                    "action_acc": action_acc,
                }
            )

        # Keep transition data for response/stopping metrics.  Steady-state
        # fields remain available separately for unbiased speed comparison.
        all_results[phase.name] = summarize(rows)
        steady_rows = rows[min(50, max(0, len(rows) - 1)) :]
        steady = summarize(steady_rows or rows)
        for name in (
            "mean_forward_speed_mps",
            "mean_abs_lateral_speed_mps",
            "tracking_rmse_mps",
            "tilt_rms_deg",
            "both_blades_grounded_fraction",
            "mean_action_acceleration",
        ):
            all_results[phase.name][f"steady_{name}"] = steady[name]

    result = {
        "policy": str(policy_path),
        "control_hz": 50,
        "physics_hz": 200,
        "current_limit_a": args.current_limit,
        "wheel_frictionloss": args.wheel_friction,
        "phases": all_results,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
