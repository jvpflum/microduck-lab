#!/usr/bin/env python3
"""Nominal headless evaluation for MicroDuck roller speed-discovery policies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
INFER_PATH = UPSTREAM / "scripts" / "infer_policy.py"
ROLLER_SCENE = (
    UPSTREAM / "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
)
CONTROL_DT = 0.02
MPS_TO_MPH = 2.2369362921
FALL_ANGLE_RAD = math.radians(70.0)


def load_inference_module():
    spec = importlib.util.spec_from_file_location(
        "microduck_speed_discovery_infer", INFER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference module: {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quat_tilt(q: np.ndarray) -> float:
    w, x, y, z = q
    del w, z
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


def gait_cycles(separation: np.ndarray) -> int:
    if separation.size < 2:
        return 0
    span = float(np.ptp(separation))
    if span < 0.005:
        return 0
    low = float(separation.min() + 0.25 * span)
    high = float(separation.min() + 0.75 * span)
    expanded = False
    cycles = 0
    for sample in separation:
        if not expanded and sample >= high:
            expanded = True
        elif expanded and sample <= low:
            cycles += 1
            expanded = False
    return cycles


def rolling_best(values: np.ndarray, window_s: float) -> float:
    if values.size == 0:
        return 0.0
    window = min(values.size, max(1, int(round(window_s / CONTROL_DT))))
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return float(np.convolve(values, kernel, mode="valid").max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--command-mps", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--initial-joint-noise", type=float, default=0.01)
    parser.add_argument("--current-limit", type=float, default=1.75)
    parser.add_argument("--wheel-friction", type=float, default=0.0)
    parser.add_argument(
        "--race-line-control",
        action="store_true",
        help="Feed closed-loop heading/lateral correction through command yaw.",
    )
    parser.add_argument("--yaw-kp", type=float, default=0.55)
    parser.add_argument("--lateral-kp", type=float, default=0.10)
    parser.add_argument("--yaw-kd", type=float, default=0.08)
    parser.add_argument("--max-correction", type=float, default=0.18)
    args = parser.parse_args()
    if args.episodes < 1 or args.duration <= 0.0:
        raise SystemExit("--episodes and --duration must be positive")

    policy = args.policy.resolve()
    if not policy.is_file():
        raise SystemExit(f"Policy not found: {policy}")

    infer = load_inference_module()
    model = mujoco.MjModel.from_xml_path(str(ROLLER_SCENE))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    if args.current_limit > 0.0:
        from bam.model import load_model

        torque_limit = load_model(motor_name="xl330", model="m6").kt.value * args.current_limit
        model.actuator_forcerange[:, :] = (-torque_limit, torque_limit)
        model.actuator_forcelimited[:] = 1

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name.startswith("passive_") and "wheel" in name:
            model.dof_frictionloss[int(model.jnt_dofadr[joint_id])] = args.wheel_friction

    controller = infer.PolicyInference(
        model,
        data,
        walking_onnx_path=str(policy),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    free_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[free_joint])
    qvel_adr = int(model.jnt_dofadr[free_joint])
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_v1")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_v1")
    rng = np.random.default_rng(args.seed)
    decimation = int(round(CONTROL_DT / model.opt.timestep))
    episode_rows: list[dict[str, float | int | bool]] = []
    all_forward: list[float] = []

    for episode in range(args.episodes):
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr : qpos_adr + 7] = [
            0.0,
            0.0,
            0.1385,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        noise = rng.uniform(
            -args.initial_joint_noise,
            args.initial_joint_noise,
            size=len(controller.joint_qpos_indices),
        )
        for index, joint_qpos in enumerate(controller.joint_qpos_indices):
            data.qpos[joint_qpos] = controller.default_pose[index] + noise[index]
        data.ctrl[:] = controller.default_pose
        controller.last_action.fill(0.0)
        controller.set_vel_cmd(args.command_mps, 0.0, 0.0)
        mujoco.mj_forward(model, data)

        start_x = float(data.qpos[qpos_adr])
        start_y = float(data.qpos[qpos_adr + 1])
        world_forward: list[float] = []
        body_forward: list[float] = []
        separation: list[float] = []
        lateral_offsets: list[float] = []
        headings: list[float] = []
        fell = False
        steps = int(round(args.duration / CONTROL_DT))
        for _ in range(steps):
            if args.race_line_control:
                quat_now = data.qpos[qpos_adr + 3 : qpos_adr + 7]
                w, x, y, z = quat_now
                yaw_now = math.atan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z),
                )
                lateral_error = float(data.qpos[qpos_adr + 1]) - start_y
                yaw_rate = float(data.qvel[qvel_adr + 5])
                correction = np.clip(
                    -args.yaw_kp * yaw_now
                    - args.lateral_kp * lateral_error
                    - args.yaw_kd * yaw_rate,
                    -args.max_correction,
                    args.max_correction,
                )
                # Avoid the interactive helper's per-step console print.  The
                # policy remains in its already-selected locomotion session.
                controller.vel_cmd[:] = (args.command_mps, 0.0, float(correction))
                controller._update_command()
            action = controller.infer()
            controller.apply_action(action)
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            quat = data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
            w, x, y, z = quat
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            vx = float(data.qvel[qvel_adr])
            vy = float(data.qvel[qvel_adr + 1])
            world_forward.append(vx)
            body_forward.append(math.cos(yaw) * vx + math.sin(yaw) * vy)
            lateral_offsets.append(float(data.qpos[qpos_adr + 1]) - start_y)
            headings.append(yaw)
            separation.append(
                float(np.linalg.norm(data.xpos[left_body, :2] - data.xpos[right_body, :2]))
            )
            if not np.isfinite(data.qpos).all() or quat_tilt(quat) >= FALL_ANGLE_RAD:
                fell = True
                break

        forward = np.asarray(world_forward, dtype=np.float64)
        body = np.asarray(body_forward, dtype=np.float64)
        sep = np.asarray(separation, dtype=np.float64)
        lateral = np.asarray(lateral_offsets, dtype=np.float64)
        heading = np.asarray(headings, dtype=np.float64)
        time_alive = len(forward) * CONTROL_DT
        cycles = gait_cycles(sep)
        world_x_displacement = float(data.qpos[qpos_adr]) - start_x
        body_forward_distance = float(body.sum() * CONTROL_DT)
        row = {
            "episode": episode + 1,
            "mean_forward_speed_mps": float(body.mean()) if body.size else 0.0,
            "mean_body_forward_speed_mps": float(body.mean()) if body.size else 0.0,
            "mean_world_forward_speed_mps": float(forward.mean()) if forward.size else 0.0,
            "max_forward_speed_mps": float(body.max()) if body.size else 0.0,
            "best_1s_forward_speed_mps": rolling_best(body, 1.0),
            "distance_m": body_forward_distance,
            "world_x_displacement_m": world_x_displacement,
            "sustained_world_x_speed_mps": world_x_displacement / args.duration,
            "max_abs_lateral_deviation_m": (
                float(np.abs(lateral).max()) if lateral.size else 0.0
            ),
            "mean_abs_heading_deg": (
                math.degrees(float(np.abs(heading).mean())) if heading.size else 0.0
            ),
            "final_abs_heading_deg": (
                math.degrees(abs(float(heading[-1]))) if heading.size else 0.0
            ),
            # Horizon-normalized speed treats all post-fall time as zero.  This
            # prevents a brief launch-and-crash policy from beating a genuinely
            # sustained gait in checkpoint selection.
            "sustained_mean_forward_speed_mps": body_forward_distance / args.duration,
            "time_alive_s": time_alive,
            "survived": not fell and time_alive >= args.duration - CONTROL_DT,
            "falls": int(fell),
            "gait_cycles": cycles,
            "gait_frequency_hz": cycles / max(time_alive, CONTROL_DT),
        }
        episode_rows.append(row)
        all_forward.extend(body_forward)

    samples = np.asarray(all_forward, dtype=np.float64)
    mean_speed = float(np.mean([row["mean_forward_speed_mps"] for row in episode_rows]))
    sustained_speed = float(
        np.mean([row["sustained_mean_forward_speed_mps"] for row in episode_rows])
    )
    peak_speed = float(samples.max()) if samples.size else 0.0
    best_1s = float(max(row["best_1s_forward_speed_mps"] for row in episode_rows))
    survival = float(np.mean([row["survived"] for row in episode_rows]))
    sustained_world_x = float(
        np.mean([row["sustained_world_x_speed_mps"] for row in episode_rows])
    )
    summary = {
        "average_forward_speed_mps": mean_speed,
        "average_forward_speed_mph": mean_speed * MPS_TO_MPH,
        "sustained_mean_forward_speed_mps": sustained_speed,
        "sustained_mean_forward_speed_mph": sustained_speed * MPS_TO_MPH,
        "sustained_world_x_speed_mps": sustained_world_x,
        "sustained_world_x_speed_mph": sustained_world_x * MPS_TO_MPH,
        "maximum_forward_speed_mps": peak_speed,
        "maximum_forward_speed_mph": peak_speed * MPS_TO_MPH,
        "best_1s_forward_speed_mps": best_1s,
        "best_1s_forward_speed_mph": best_1s * MPS_TO_MPH,
        "average_episode_distance_m": float(
            np.mean([row["distance_m"] for row in episode_rows])
        ),
        "average_time_alive_s": float(
            np.mean([row["time_alive_s"] for row in episode_rows])
        ),
        "survival_fraction": survival,
        "falls": int(sum(row["falls"] for row in episode_rows)),
        "mean_gait_frequency_hz": float(
            np.mean([row["gait_frequency_hz"] for row in episode_rows])
        ),
        "mean_max_abs_lateral_deviation_m": float(
            np.mean([row["max_abs_lateral_deviation_m"] for row in episode_rows])
        ),
        "mean_abs_heading_deg": float(
            np.mean([row["mean_abs_heading_deg"] for row in episode_rows])
        ),
        "mean_final_abs_heading_deg": float(
            np.mean([row["final_abs_heading_deg"] for row in episode_rows])
        ),
    }
    result = {
        "policy": str(policy),
        "model": "MicroDuck passive rollers",
        "command_mps": args.command_mps,
        "command_mph": args.command_mps * MPS_TO_MPH,
        "episodes": args.episodes,
        "duration_s": args.duration,
        "control_hz": int(round(1.0 / CONTROL_DT)),
        "physics_hz": int(round(1.0 / model.opt.timestep)),
        "current_limit_a": args.current_limit,
        "wheel_frictionloss": args.wheel_friction,
        "race_line_control": args.race_line_control,
        "line_hold": {
            "yaw_kp": args.yaw_kp,
            "lateral_kp": args.lateral_kp,
            "yaw_kd": args.yaw_kd,
            "max_correction": args.max_correction,
        },
        "initial_joint_noise_rad": args.initial_joint_noise,
        "summary": summary,
        "episode_results": episode_rows,
        "checkpoint_rank": {
            "primary_sustained_mean_mps": sustained_speed,
            "secondary_survival_fraction": survival,
            "tertiary_peak_mps": peak_speed,
            "world_x_sustained_mps": sustained_world_x,
        },
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
