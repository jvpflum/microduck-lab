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
    reset_before: bool = False


CONTROL_DT = 0.02
RACE5_LONG_RUN_DISTANCE_M = 100.0 * 0.3048

PHASES = (
    Phase("settle", 2.0, 0.0),
    Phase("forward", 8.0, 0.3),
    Phase("stop_forward", 4.0, 0.0),
    Phase("reverse", 8.0, -0.3),
    Phase("stop_reverse", 4.0, 0.0),
    Phase("heading_left", 6.0, 0.2, 0.3),
    Phase("heading_right", 6.0, 0.2, -0.3),
)

SPRINT_PHASES = (
    Phase("settle", 2.0, 0.0),
    Phase("speed_020", 6.0, 0.20),
    Phase("stop_020", 3.0, 0.0),
    Phase("speed_030", 6.0, 0.30),
    Phase("stop_030", 3.0, 0.0),
    Phase("speed_040", 7.0, 0.40),
    Phase("stop_040", 3.0, 0.0),
    Phase("speed_050", 8.0, 0.50),
    Phase("stop_050", 4.0, 0.0),
    Phase("speed_055", 8.0, 0.55),
    Phase("stop_055", 4.0, 0.0),
)

SPRINT_EXTENDED_PHASES = (
    Phase("settle", 2.0, 0.0),
    Phase("speed_055", 8.0, 0.55),
    Phase("stop_055", 4.0, 0.0),
    Phase("speed_060", 8.0, 0.60),
    Phase("stop_060", 4.0, 0.0),
    Phase("speed_065", 8.0, 0.65),
    Phase("stop_065", 4.0, 0.0),
    Phase("speed_070", 8.0, 0.70),
    Phase("stop_070", 4.0, 0.0),
)

RACE_PHASES = (
    Phase("settle", 2.0, 0.0),
    Phase("race", 14.0, 0.80),
)

RACE5_PHASES = (
    Phase("settle", 2.0, 0.0),
    # Retention circuit: cruise -> brake is continuous so braking is measured
    # from motion. Steering and the race heat start from identical clean poses
    # so one test cannot corrupt the next test's heading or speed.
    Phase("cruise", 5.0, 0.30),
    Phase("stop_cruise", 3.0, 0.0),
    Phase("turn_left", 4.0, 0.20, 0.30, reset_before=True),
    Phase("turn_right", 4.0, 0.20, -0.30, reset_before=True),
    # Race effort token; 5 mph is a measured outcome, not an OOD command.
    Phase("race", 8.0, 0.80, reset_before=True),
    # Match the browser's enlarged runway with a long, unassisted acceleration
    # trial. The scene floor is effectively unbounded, so this measures the
    # policy rather than a wall collision. Ninety seconds gives even Pollen's
    # roughly 1 mph gait enough time to traverse a nominal 100-foot course.
    Phase("max_speed", 90.0, 0.80, reset_before=True),
)

# Cheap checkpoint screen used before spending the full retention circuit and
# 90-second official heat.  Twenty seconds is long enough to reject a policy
# that begins the same looping failure as v5/v8 while preserving the exact
# physics, command, and telemetry used by the official evaluator.
RACE_SCREEN_PHASES = (
    Phase("max_speed", 20.0, 0.80, reset_before=True),
)

# Reproduces opening the browser arena and touching only steering. This probes
# policies such as V47 that move despite a zero forward command.
IDLE_LAUNCH_PHASES = (
    Phase("idle_launch", 20.0, 0.0, reset_before=True),
)


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_tilt(q: np.ndarray) -> float:
    w, x, y, z = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


def launch_yaw_pulse(
    phase_time_s: float,
    start_time_s: float,
    yaw_command: float,
    pulse_duration_s: float,
    pulse_count: int,
    pulse_gap_s: float,
) -> float:
    """Return a launch-only yaw command that mimics discrete arrow-key taps."""
    if yaw_command == 0.0 or pulse_duration_s <= 0.0 or pulse_count <= 0:
        return 0.0
    elapsed_s = phase_time_s - max(0.0, start_time_s)
    if elapsed_s < 0.0:
        return 0.0
    pulse_period_s = pulse_duration_s + max(0.0, pulse_gap_s)
    pulse_index = int(elapsed_s / pulse_period_s)
    if pulse_index >= pulse_count:
        return 0.0
    pulse_time_s = elapsed_s - pulse_index * pulse_period_s
    return yaw_command if pulse_time_s < pulse_duration_s else 0.0


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    forward = values("forward_speed")
    world_forward = values("world_forward_speed")
    lateral = values("lateral_speed")
    lateral_position = values("lateral_position")
    trunk_z = values("trunk_z")
    tilt = values("tilt")
    separation = values("skate_separation")
    actions = np.asarray([row["action_acc"] for row in rows], dtype=np.float64)
    grounded = values("both_grounded")
    wheel_speed = values("mean_abs_wheel_speed")
    command_x = values("command_x")
    auto_steering = values("auto_steering")
    forward_position = values("forward_position")
    yaw = np.unwrap(values("yaw"))
    horizontal_speed = np.hypot(world_forward, values("world_lateral_speed"))
    # A one-sample contact impulse is not a defensible top-speed record.
    # Report a 0.5 s rolling peak (25 control samples at 50 Hz), alongside
    # the raw peak for diagnostics. This mirrors a short timing trap while
    # preserving enough resolution to capture the fastest part of the run.
    top_window = min(len(horizontal_speed), max(1, int(round(0.5 / CONTROL_DT))))
    verified_top_speed = float(
        np.convolve(
            horizontal_speed,
            np.full(top_window, 1.0 / top_window, dtype=np.float64),
            mode="valid",
        ).max()
    )
    lateral_drift = np.abs(lateral_position - lateral_position[0])
    heading_error = np.abs(yaw - yaw[0])
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

    commanded_speed = direction * forward if nonzero_command else np.abs(forward)
    first_second_index = min(len(commanded_speed) - 1, max(1, int(round(1.0 / CONTROL_DT)) - 1))
    acceleration_first_second = float(
        (commanded_speed[first_second_index] - commanded_speed[0])
        / max(CONTROL_DT, first_second_index * CONTROL_DT)
    )
    half_speed = np.flatnonzero(commanded_speed >= 0.5)
    time_to_half = float((half_speed[0] + 1) * CONTROL_DT) if half_speed.size else None

    stop_time = None
    if not nonzero_command:
        quiet = np.abs(forward) <= 0.05
        window = max(1, int(round(0.5 / CONTROL_DT)))
        for index in range(0, max(0, len(quiet) - window + 1)):
            if bool(np.all(quiet[index : index + window])):
                stop_time = float((index + 1) * CONTROL_DT)
                break

    progress = forward_position - forward_position[0]
    finish_indices = np.flatnonzero(progress >= 5.0)
    finish_time = float((finish_indices[0] + 1) * CONTROL_DT) if finish_indices.size else None

    def split_time(distance_m: float) -> float | None:
        indices = np.flatnonzero(progress >= distance_m)
        if not indices.size:
            return None
        index = int(indices[0])
        if index == 0:
            return CONTROL_DT
        p0, p1 = float(progress[index - 1]), float(progress[index])
        fraction = (distance_m - p0) / max(p1 - p0, 1e-9)
        return float((index + fraction) * CONTROL_DT)

    split_10ft = split_time(10.0 * 0.3048)
    split_25ft = split_time(25.0 * 0.3048)
    split_50ft = split_time(50.0 * 0.3048)
    split_100ft = split_time(RACE5_LONG_RUN_DISTANCE_M)
    trap_speed_100ft = None
    if split_100ft is not None:
        trap_rows = (progress >= 80.0 * 0.3048) & (progress <= RACE5_LONG_RUN_DISTANCE_M)
        if np.any(trap_rows):
            trap_speed_100ft = float(world_forward[trap_rows].mean())

    world_forward_integral = float(world_forward.sum() * CONTROL_DT)
    measured_progress = float(progress[-1])
    integration_error = abs(world_forward_integral - measured_progress)
    distance_time_speed = float(
        RACE5_LONG_RUN_DISTANCE_M / split_100ft
        if split_100ft is not None
        else measured_progress / max(duration, CONTROL_DT)
    )

    return {
        "duration_s": duration,
        "command_x_mps": float(command_x[-1]),
        "mean_forward_speed_mps": float(forward.mean()),
        "mean_forward_speed_mph": float(forward.mean() * 2.2369362921),
        "mean_abs_forward_speed_mps": float(np.abs(forward).mean()),
        "peak_abs_forward_speed_mps": float(np.abs(forward).max()),
        "peak_abs_forward_speed_mph": float(np.abs(forward).max() * 2.2369362921),
        "mean_world_forward_speed_mps": float(world_forward.mean()),
        "mean_world_forward_speed_mph": float(world_forward.mean() * 2.2369362921),
        "peak_world_forward_speed_mps": float(world_forward.max()),
        "peak_world_forward_speed_mph": float(world_forward.max() * 2.2369362921),
        "peak_horizontal_speed_mps": float(horizontal_speed.max()),
        "peak_horizontal_speed_mph": float(horizontal_speed.max() * 2.2369362921),
        "verified_top_speed_0_5s_mps": verified_top_speed,
        "verified_top_speed_0_5s_mph": verified_top_speed * 2.2369362921,
        "max_lateral_drift_m": float(lateral_drift.max()),
        "max_lateral_drift_ft": float(lateral_drift.max() * 3.280839895),
        "final_lateral_offset_m": float(lateral_position[-1] - lateral_position[0]),
        "final_lateral_offset_ft": float(
            (lateral_position[-1] - lateral_position[0]) * 3.280839895
        ),
        "mean_world_lateral_speed_mps": float(values("world_lateral_speed").mean()),
        "mean_abs_auto_steering_rad_s": float(np.abs(auto_steering).mean()),
        "max_abs_auto_steering_rad_s": float(np.abs(auto_steering).max()),
        "auto_steering_percent": float(100.0 * np.abs(auto_steering).mean() / 0.30),
        "max_heading_error_deg": float(np.degrees(heading_error.max())),
        "acceleration_first_second_mps2": acceleration_first_second,
        "time_to_0_5_mps_s": time_to_half,
        "end_abs_forward_speed_mps": float(np.abs(forward[-min(len(forward), 25) :]).mean()),
        "forward_distance_m": float(np.sum(forward) * CONTROL_DT),
        "forward_progress_m": float(progress[-1]),
        "distance_time_average_speed_mps": distance_time_speed,
        "distance_time_average_speed_mph": distance_time_speed * 2.2369362921,
        "world_forward_velocity_integral_m": world_forward_integral,
        "position_velocity_integration_error_m": integration_error,
        "position_velocity_integration_error_percent": float(
            100.0 * integration_error / max(abs(measured_progress), 1e-9)
        ),
        "finished_5m": finish_time is not None,
        "finish_time_5m_s": finish_time,
        "finished_100ft": split_100ft is not None,
        "finish_time_100ft_s": split_100ft,
        "split_time_10ft_s": split_10ft,
        "split_time_25ft_s": split_25ft,
        "split_time_50ft_s": split_50ft,
        "trap_speed_100ft_mps": trap_speed_100ft,
        "trap_speed_100ft_mph": trap_speed_100ft * 2.2369362921 if trap_speed_100ft is not None else None,
        "distance_remaining_100ft_ft": float(max(0.0, RACE5_LONG_RUN_DISTANCE_M - progress.max()) * 3.280839895),
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
    parser.add_argument("--line-hold", action="store_true")
    parser.add_argument("--line-yaw-kp", type=float, default=0.55)
    parser.add_argument("--line-lateral-kp", type=float, default=0.10)
    parser.add_argument("--line-yaw-kd", type=float, default=0.08)
    parser.add_argument("--line-max-wz", type=float, default=0.18)
    parser.add_argument("--line-launch-bias-wz", type=float, default=0.0)
    parser.add_argument("--line-launch-bias-distance", type=float, default=1.5)
    parser.add_argument("--command-max-wz", type=float, default=0.30)
    parser.add_argument(
        "--launch-yaw-command",
        type=float,
        default=0.0,
        help="Launch-only yaw command; negative mimics tapping right",
    )
    parser.add_argument("--launch-yaw-start", type=float, default=0.0)
    parser.add_argument("--launch-yaw-pulse-duration", type=float, default=0.10)
    parser.add_argument("--launch-yaw-pulse-count", type=int, default=0)
    parser.add_argument("--launch-yaw-pulse-gap", type=float, default=0.10)
    parser.add_argument(
        "--profile",
        choices=(
            "swizzle", "sprint", "sprint-extended", "race", "race-5mph",
            "race-screen", "idle-launch",
        ),
        default="swizzle",
    )
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
    def reset_rollout() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr : qpos_adr + 7] = [
            0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0
        ]
        for index, joint_qpos in enumerate(controller.joint_qpos_indices):
            data.qpos[joint_qpos] = controller.default_pose[index]
        data.ctrl[:] = controller.default_pose
        controller.last_action.fill(0.0)
        mujoco.mj_forward(model, data)

    reset_rollout()

    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_v1")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_v1")
    if left_body < 0 or right_body < 0:
        raise SystemExit("Roller model is missing left/right ankle bodies")

    control_dt = CONTROL_DT
    decimation = int(round(control_dt / model.opt.timestep))
    all_results: dict[str, dict[str, float]] = {}
    previous_action = np.zeros(model.nu, dtype=np.float32)
    previous_delta = np.zeros(model.nu, dtype=np.float32)

    phases = {
        "swizzle": PHASES,
        "sprint": SPRINT_PHASES,
        "sprint-extended": SPRINT_EXTENDED_PHASES,
        "race": RACE_PHASES,
        "race-5mph": RACE5_PHASES,
        "race-screen": RACE_SCREEN_PHASES,
        "idle-launch": IDLE_LAUNCH_PHASES,
    }[args.profile]
    for phase in phases:
        if phase.reset_before:
            reset_rollout()
            previous_action.fill(0.0)
            previous_delta.fill(0.0)
        controller.set_vel_cmd(phase.command_x, 0.0, phase.heading_error)
        rows: list[dict[str, float]] = []
        phase_start_x = float(data.qpos[qpos_adr])
        phase_start_y = float(data.qpos[qpos_adr + 1])
        phase_start_yaw = quat_to_yaw(data.qpos[qpos_adr + 3 : qpos_adr + 7])
        steps = int(round(phase.duration_s / control_dt))
        for step in range(steps):
            auto_steering = 0.0
            if args.line_hold and phase.name in {"race", "max_speed"} and phase.command_x > 0.05:
                yaw = quat_to_yaw(data.qpos[qpos_adr + 3 : qpos_adr + 7])
                yaw_error = math.atan2(
                    math.sin(yaw - phase_start_yaw), math.cos(yaw - phase_start_yaw)
                )
                lateral_error = float(data.qpos[qpos_adr + 1]) - phase_start_y
                yaw_rate = float(data.qvel[qvel_adr + 5])
                launch_distance = max(
                    0.0, float(data.qpos[qpos_adr]) - phase_start_x
                )
                launch_bias_scale = max(
                    0.0,
                    1.0 - launch_distance / max(1e-6, args.line_launch_bias_distance),
                )
                auto_steering = float(np.clip(
                    args.line_launch_bias_wz * launch_bias_scale
                    - args.line_yaw_kp * yaw_error
                    - args.line_lateral_kp * lateral_error
                    - args.line_yaw_kd * yaw_rate,
                    -args.line_max_wz,
                    args.line_max_wz,
                ))
            launch_steering = 0.0
            if phase.name in {"race", "max_speed", "idle_launch"}:
                launch_steering = launch_yaw_pulse(
                    step * control_dt,
                    args.launch_yaw_start,
                    args.launch_yaw_command,
                    args.launch_yaw_pulse_duration,
                    args.launch_yaw_pulse_count,
                    args.launch_yaw_pulse_gap,
                )
            controller.set_vel_cmd(
                phase.command_x,
                0.0,
                float(np.clip(
                    phase.heading_error + auto_steering + launch_steering,
                    -args.command_max_wz,
                    args.command_max_wz,
                )),
            )
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
                    "auto_steering": auto_steering,
                    "forward_speed": forward_speed,
                    "world_forward_speed": vx,
                    "world_lateral_speed": vy,
                    "forward_position": float(data.qpos[qpos_adr]),
                    "lateral_position": float(data.qpos[qpos_adr + 1]),
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
            # Race-v1 terminates an episode at the finish line. Do not score
            # post-finish wandering that the training environment never sees.
            if (
                args.profile == "race"
                and phase.name == "race"
                and float(data.qpos[qpos_adr]) - phase_start_x >= 5.0
            ):
                break
            if (
                args.profile == "race-5mph"
                and phase.name == "max_speed"
                and float(data.qpos[qpos_adr]) - phase_start_x >= RACE5_LONG_RUN_DISTANCE_M
            ):
                break

        # Keep transition data for response/stopping metrics.  Steady-state
        # fields remain available separately for unbiased speed comparison.
        all_results[phase.name] = summarize(rows)
        steady_rows = rows[min(50, max(0, len(rows) - 1)) :]
        steady = summarize(steady_rows or rows)
        for name in (
            "mean_forward_speed_mps",
            "mean_world_forward_speed_mps",
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
        "profile": args.profile,
        "line_hold": {
            "enabled": args.line_hold,
            "yaw_kp": args.line_yaw_kp,
            "lateral_kp": args.line_lateral_kp,
            "yaw_kd": args.line_yaw_kd,
            "max_wz": args.line_max_wz,
            "launch_bias_wz": args.line_launch_bias_wz,
            "launch_bias_distance_m": args.line_launch_bias_distance,
        },
        "launch_yaw_pulses": {
            "command": args.launch_yaw_command,
            "start_s": args.launch_yaw_start,
            "duration_s": args.launch_yaw_pulse_duration,
            "count": args.launch_yaw_pulse_count,
            "gap_s": args.launch_yaw_pulse_gap,
        },
        "phases": all_results,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
