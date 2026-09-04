#!/usr/bin/env python3
"""GPU-batched CEM search for a clean rolling-entry MicroDuck front flip.

The candidate is an eight-knot, bilateral-symmetric joint-position primitive.
Thousands of candidates are simulated concurrently with MuJoCo Warp.  Search
uses the exact 0.003 wheel friction and 1.75 A torque cap, and treats any
trunk/jaw terrain contact during preload as invalid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch

from bam.model import load_model
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_roller_frontflip_ballistic_env_cfg import (
    make_microduck_roller_frontflip_ballistic_env_cfg,
)


CONTROL_DT = 0.02
WHEEL_RADIUS = 0.0175
STAND_HEIGHT = 0.115
TAKEOFF_CLEARANCE = 0.010
MIN_TAKEOFF_VERTICAL_SPEED = 0.05
MIN_LANDING_ROTATION = math.radians(300.0)
MAX_LANDING_ROTATION = math.radians(420.0)
COMPACT_DOF = 7
INTERIOR_KNOTS = 6
PARAM_DIM = COMPACT_DOF * INTERIOR_KNOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--seed-reference", type=Path, action="append", default=[],
        help=(
            "Additional packaged primitives with matching physics and knot times. "
            "They are retained as anchors and recombined phase-by-phase."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument(
        "--simulation-batch", type=int, default=0,
        help=(
            "Candidates simulated concurrently. Zero uses the full population. "
            "Smaller reusable batches allow a much larger search population "
            "without exhausting WSL host memory."
        ),
    )
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elite-fraction", type=float, default=0.0625)
    parser.add_argument("--update-rate", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--wheel-friction", type=float, default=0.003)
    parser.add_argument("--current-limit-a", type=float, default=1.75)
    parser.add_argument("--start-speeds", type=float, nargs="+", default=[0.75, 0.85])
    parser.add_argument("--seed", type=int, default=5173)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-envs", type=int, default=32768,
        help="Explicit VRAM safety ceiling for population * rollouts",
    )
    parser.add_argument(
        "--trace-output", type=Path,
        help="Write a single-candidate, single-rollout per-step simulator trace and exit",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def quat_delta_vector(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    previous = previous / previous.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    pw, px, py, pz = previous.unbind(-1)
    cw, cx, cy, cz = current.unbind(-1)
    # conjugate(previous) * current
    w = pw * cw + px * cx + py * cy + pz * cz
    x = pw * cx - px * cw - py * cz + pz * cy
    y = pw * cy + px * cz - py * cw - pz * cx
    z = pw * cz - px * cy + py * cx - pz * cw
    delta = torch.stack((w, x, y, z), dim=-1)
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    delta = torch.where((delta[:, :1] < 0.0), -delta, delta)
    vector = delta[:, 1:]
    length = vector.norm(dim=-1)
    angle = 2.0 * torch.atan2(length, delta[:, 0].clamp_min(0.0))
    return vector / length.unsqueeze(-1).clamp_min(1.0e-9) * angle.unsqueeze(-1)


def compact_from_nodes(nodes: torch.Tensor) -> torch.Tensor:
    return torch.cat((nodes[1:-1, :5], nodes[1:-1, 5:7]), dim=-1).reshape(-1)


def expand_nodes(params: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    compact = params.reshape(-1, INTERIOR_KNOTS, COMPACT_DOF)
    full = default.view(1, 1, 14).repeat(params.shape[0], INTERIOR_KNOTS + 2, 1)
    full[:, 1:-1, :5] = compact[:, :, :5]
    full[:, 1:-1, 5:7] = compact[:, :, 5:7]
    full[:, 1:-1, 9:14] = -compact[:, :, :5]
    return full


def clamp_params(params: torch.Tensor) -> torch.Tensor:
    compact = params.reshape(-1, INTERIOR_KNOTS, COMPACT_DOF)
    compact[:, :, 0].clamp_(-0.40, 0.40)
    compact[:, :, 1].clamp_(-0.38, 0.38)
    compact[:, :, 2:5].clamp_(-1.50, 1.50)
    compact[:, :, 5].clamp_(-1.50, 1.00)
    compact[:, :, 6].clamp_(-1.50, 1.50)
    return params


def apply_current_limit(env: ManagerBasedRlEnv, current_limit_a: float) -> float:
    motor = load_model(motor_name="xl330", model="m6")
    torque_limit = float(current_limit_a * motor.kt.value)
    env.sim.model.actuator_forcerange[..., 0] = -torque_limit
    env.sim.model.actuator_forcerange[..., 1] = torque_limit
    return torque_limit


def reset_flip_buffers(env: ManagerBasedRlEnv) -> None:
    mdp._roller_backflip_state(env)
    for name in (
        "_roller_backflip_accum", "_roller_backflip_max", "_roller_backflip_paid",
        "_roller_backflip_peak_clearance", "_roller_backflip_paid_clearance",
        "_roller_backflip_assist_omega", "_roller_backflip_readiness_max",
        "_roller_backflip_readiness_paid", "_roller_backflip_peak_takeoff_pitch",
        "_roller_backflip_paid_takeoff_pitch", "_roller_frontflip_peak_supported_angmom",
        "_roller_frontflip_paid_supported_angmom", "_roller_frontflip_recovery_max",
        "_roller_frontflip_recovery_paid",
    ):
        getattr(env, name).zero_()
    for name in (
        "_roller_backflip_supported", "_roller_backflip_takeoff",
        "_roller_backflip_landed", "_roller_backflip_invalid",
        "_roller_backflip_just_landed", "_roller_backflip_demo_reset",
        "_roller_frontflip_stable_latch",
    ):
        getattr(env, name).zero_()
    env._roller_frontflip_stable_steps.zero_()
    env._roller_backflip_last_update_step = -1


def evaluate_population(
    env: ManagerBasedRlEnv,
    candidates: torch.Tensor,
    knot_times: torch.Tensor,
    default: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    population = candidates.shape[0]
    rollouts = args.rollouts
    if getattr(args, "candidate_nodes_mode", False):
        if candidates.ndim != 3 or candidates.shape[-1] != 14:
            raise ValueError(
                "candidate_nodes_mode expects candidates shaped (population, knots, 14)"
            )
        env_nodes = candidates.repeat_interleave(rollouts, dim=0)
    else:
        env_nodes = expand_nodes(candidates, default).repeat_interleave(
            rollouts, dim=0
        )
    num_envs = population * rollouts
    env.reset()
    reset_flip_buffers(env)

    robot = env.scene["robot"]
    handoff_qpos = getattr(args, "handoff_qpos", None)
    handoff_qvel = getattr(args, "handoff_qvel", None)
    handoff_time = float(getattr(args, "handoff_time", 0.0))
    if handoff_qpos is not None or handoff_qvel is not None:
        if handoff_qpos is None or handoff_qvel is None:
            raise ValueError("handoff_qpos and handoff_qvel must be provided together")
        qpos = torch.as_tensor(handoff_qpos, device=env.device, dtype=torch.float32)
        qvel = torch.as_tensor(handoff_qvel, device=env.device, dtype=torch.float32)
        if qpos.ndim != 1 or qvel.ndim != 1:
            raise ValueError("handoff state must be one-dimensional")
        env.sim.data.qpos[:] = qpos.unsqueeze(0)
        env.sim.data.qvel[:] = qvel.unsqueeze(0)
        env.sim.forward()
    else:
        speeds = torch.as_tensor(
            args.start_speeds, device=env.device, dtype=torch.float32
        )
        if len(speeds) != rollouts:
            raise ValueError("the number of --start-speeds must equal --rollouts")
        # Match the independent native evaluator exactly. Training reset noise is
        # useful for robustness later, but it obscures the motion-planning signal.
        env.sim.data.qpos[:, 0:7] = torch.tensor(
            [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0],
            device=env.device,
        )
        env.sim.data.qvel.zero_()
        forward_speed = speeds.repeat(population)
        env.sim.data.qvel[:, 0] = forward_speed
        wheel_ids, _ = robot.find_joints(r"^passive_.*wheel$")
        wheel_qvel = torch.as_tensor(
            [6 + joint_id for joint_id in wheel_ids],
            device=env.device,
            dtype=torch.long,
        )
        # A rolling reset is only physically consistent when wheel surface speed
        # matches base speed.  All four passive wheel coordinates use positive
        # omega for forward motion in this MJCF.
        env.sim.data.qvel[:, wheel_qvel] = (
            forward_speed / WHEEL_RADIUS
        ).unsqueeze(1)
        servo_ids = mdp._servo_joint_ids(env, robot)
        servo_qpos = torch.as_tensor(
            [7 + joint_id for joint_id in servo_ids],
            device=env.device,
            dtype=torch.long,
        )
        env.sim.data.qpos[:, servo_qpos] = default.unsqueeze(0)
        env.sim.forward()
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    previous_quat = robot.data.root_link_quat_w.clone()
    starts_airborne = handoff_qpos is not None
    support_seen = torch.full(
        (num_envs,), starts_airborne, dtype=torch.bool, device=env.device
    )
    takeoff = torch.full_like(support_seen, starts_airborne)
    clean_takeoff = torch.full_like(support_seen, starts_airborne)
    landed = torch.zeros_like(support_seen)
    any_body = torch.zeros_like(support_seen)
    pre_takeoff_body = torch.zeros_like(support_seen)
    first_body_step = torch.full(
        (num_envs,), 10_000, dtype=torch.long, device=env.device
    )
    initial_rotation = float(getattr(args, "initial_rotation_rad", 0.0))
    forward_rotation = torch.full(
        (num_envs,), initial_rotation, dtype=torch.float32, device=env.device
    )
    max_rotation = forward_rotation.clone()
    clean_forward_rotation = forward_rotation.clone()
    max_clean_rotation = forward_rotation.clone()
    offaxis_rotation = torch.zeros_like(forward_rotation)
    peak_clearance = torch.zeros_like(forward_rotation)
    peak_clean_clearance = torch.zeros_like(forward_rotation)
    clean_flight_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    peak_clean_pitch_rate = torch.zeros_like(forward_rotation)
    peak_clean_vertical_speed = torch.zeros_like(forward_rotation)
    peak_clean_supported_vertical_speed = torch.zeros_like(forward_rotation)
    takeoff_vertical_velocity = torch.where(
        takeoff,
        robot.data.root_link_lin_vel_w[:, 2],
        torch.zeros_like(forward_rotation),
    )
    takeoff_pitch_rate = torch.where(
        takeoff,
        torch.clamp(robot.data.root_link_ang_vel_b[:, 1], min=0.0),
        torch.zeros_like(forward_rotation),
    )
    stable_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    stable_latch = torch.zeros_like(support_seen)
    landing_signed_pitch_deg = torch.full(
        (num_envs,), 90.0, dtype=torch.float32, device=env.device
    )
    landing_tilt_deg = torch.full_like(landing_signed_pitch_deg, 180.0)
    landing_pitch_rate = torch.full_like(landing_signed_pitch_deg, 20.0)
    landing_vertical_speed = torch.full_like(landing_signed_pitch_deg, 20.0)
    landing_forward_speed = torch.zeros_like(landing_signed_pitch_deg)
    landing_body_speed = torch.full_like(landing_signed_pitch_deg, 20.0)
    finite = torch.ones_like(support_seen)
    jaw_ids, _ = robot.find_bodies("jaw_soft")
    jaw_body_id = jaw_ids[0]
    # The jaw body origin remains several centimetres above the floor even
    # when the beak/shell has rotated into it.  Track the lowest of three
    # physical reference sites on that shell instead.  This is a smooth
    # pre-contact signal that can steer CEM away from the jaw-plant exploit;
    # the contact sensor below remains the authoritative validity check.
    head_site_ids, _ = robot.find_sites(
        ("mouth_tip", "tof", "head_camera"), preserve_order=True
    )
    min_jaw_height = torch.full(
        (num_envs,), float("inf"), dtype=torch.float32, device=env.device
    )
    min_head_reference_height = torch.full_like(min_jaw_height, float("inf"))

    steps = int(round((args.duration - handoff_time) / float(env.step_dt)))
    if steps <= 0:
        raise ValueError("handoff_time must be earlier than duration")
    trace_rows: list[dict] = []
    with torch.no_grad():
        for step in range(steps):
            time_s = handoff_time + step * float(env.step_dt)
            # Preserve the robot's 50 Hz actuator command cadence even when a
            # GPU search requests 5 ms manager steps for contact-event fidelity.
            control_time_s = math.floor((time_s + 1.0e-9) / 0.02) * 0.02
            index = int(torch.searchsorted(knot_times, torch.tensor(control_time_s, device=env.device), right=True).item()) - 1
            index = max(0, min(index, len(knot_times) - 2))
            span = float(knot_times[index + 1] - knot_times[index])
            blend = max(0.0, min(1.0, (control_time_s - float(knot_times[index])) / max(span, 1.0e-9)))
            target = (1.0 - blend) * env_nodes[:, index] + blend * env_nodes[:, index + 1]
            action = target - default.unsqueeze(0)
            _, _, terminated, _, _ = env.step(action)

            body_now = mdp._contact_any(env, "backflip_body_ground_contact")
            both_contact, both_airborne = mdp._roller_foot_contact_masks(
                env, "feet_ground_contact"
            )
            body_seen_before_step = any_body.clone()
            first_body_step = torch.where(
                body_now & ~any_body,
                torch.full_like(first_body_step, step + 1),
                first_body_step,
            )
            pre_takeoff_body |= body_now & ~takeoff
            any_body |= body_now
            support_seen |= both_contact
            root_z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
            clearance = torch.clamp(root_z - STAND_HEIGHT, min=0.0)
            jaw_height = (
                robot.data.body_link_pos_w[:, jaw_body_id, 2]
                - env.scene.terrain.env_origins[:, 2]
            )
            head_reference_height = (
                robot.data.site_pos_w[:, head_site_ids, 2].amin(dim=1)
                - env.scene.terrain.env_origins[:, 2]
            )
            preload = (~takeoff) & (time_s <= 0.60)
            min_jaw_height = torch.where(
                preload, torch.minimum(min_jaw_height, jaw_height), min_jaw_height
            )
            min_head_reference_height = torch.where(
                preload,
                torch.minimum(min_head_reference_height, head_reference_height),
                min_head_reference_height,
            )
            launch_prep = (
                (~takeoff) & (~any_body) & both_contact & (time_s >= 0.10)
            )
            peak_clean_supported_vertical_speed = torch.maximum(
                peak_clean_supported_vertical_speed,
                torch.where(
                    launch_prep,
                    torch.clamp(robot.data.root_link_lin_vel_w[:, 2], min=0.0),
                    torch.zeros_like(peak_clean_supported_vertical_speed),
                ),
            )
            takeoff_condition = (
                support_seen & both_airborne & (clearance >= TAKEOFF_CLEARANCE)
                & (robot.data.root_link_lin_vel_w[:, 2] >= MIN_TAKEOFF_VERTICAL_SPEED)
            )
            new_takeoff = takeoff_condition & ~takeoff
            takeoff_vertical_velocity = torch.where(
                new_takeoff,
                robot.data.root_link_lin_vel_w[:, 2],
                takeoff_vertical_velocity,
            )
            takeoff_pitch_rate = torch.where(
                new_takeoff,
                torch.clamp(robot.data.root_link_ang_vel_b[:, 1], min=0.0),
                takeoff_pitch_rate,
            )
            takeoff |= takeoff_condition
            clean_takeoff |= takeoff & ~any_body

            current_quat = robot.data.root_link_quat_w.clone()
            active_flight = takeoff & both_airborne & ~landed
            delta = quat_delta_vector(previous_quat, current_quat)
            forward_rotation += active_flight.float() * torch.clamp(delta[:, 1], min=0.0)
            clean_forward_rotation += (
                (active_flight & ~any_body).float()
                * torch.clamp(delta[:, 1], min=0.0)
            )
            offaxis_rotation += active_flight.float() * torch.linalg.vector_norm(
                delta[:, (0, 2)], dim=-1
            )
            max_rotation = torch.maximum(max_rotation, forward_rotation)
            max_clean_rotation = torch.maximum(
                max_clean_rotation, clean_forward_rotation
            )
            peak_clearance = torch.maximum(
                peak_clearance, torch.where(active_flight, clearance, torch.zeros_like(clearance))
            )
            clean_flight = active_flight & ~any_body
            peak_clean_clearance = torch.maximum(
                peak_clean_clearance,
                torch.where(clean_flight, clearance, torch.zeros_like(clearance)),
            )
            clean_flight_steps += clean_flight.long()
            clean_state = ~any_body
            peak_clean_pitch_rate = torch.maximum(
                peak_clean_pitch_rate,
                torch.where(
                    clean_state,
                    torch.clamp(robot.data.root_link_ang_vel_b[:, 1], min=0.0),
                    torch.zeros_like(peak_clean_pitch_rate),
                ),
            )
            peak_clean_vertical_speed = torch.maximum(
                peak_clean_vertical_speed,
                torch.where(
                    clean_flight,
                    torch.clamp(robot.data.root_link_lin_vel_w[:, 2], min=0.0),
                    torch.zeros_like(peak_clean_vertical_speed),
                ),
            )
            previous_quat = current_quat

            landing_now = (
                takeoff & both_contact & (max_rotation >= MIN_LANDING_ROTATION)
                & (max_rotation <= MAX_LANDING_ROTATION)
                # Warp advances four 5 ms physics steps per 20 ms manager step.
                # A tire touchdown can therefore precede a trunk contact inside
                # the same manager step.  Preserve that touchdown signal for GPU
                # search, while native 5 ms certification remains authoritative.
                & ~body_seen_before_step
            )
            new_landing = landing_now & ~landed
            quat = robot.data.root_link_quat_w
            up_z = torch.clamp(1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0)
            tilt = torch.acos(up_z)
            lin = robot.data.root_link_lin_vel_w
            ang = robot.data.root_link_ang_vel_b
            signed_pitch = torch.asin(
                torch.clamp(
                    2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]),
                    -1.0,
                    1.0,
                )
            ) * (180.0 / math.pi)
            landing_signed_pitch_deg = torch.where(
                new_landing, signed_pitch, landing_signed_pitch_deg
            )
            landing_tilt_deg = torch.where(
                new_landing, tilt * (180.0 / math.pi), landing_tilt_deg
            )
            landing_pitch_rate = torch.where(
                new_landing, ang[:, 1].abs(), landing_pitch_rate
            )
            landing_vertical_speed = torch.where(
                new_landing, lin[:, 2].abs(), landing_vertical_speed
            )
            landing_forward_speed = torch.where(
                new_landing, lin[:, 0], landing_forward_speed
            )
            landing_body_speed = torch.where(
                new_landing,
                torch.linalg.vector_norm(torch.cat((lin, ang), dim=-1), dim=-1),
                landing_body_speed,
            )
            landed |= landing_now
            rolling_stable = (
                landed & both_contact & ~any_body & (tilt <= math.radians(15.0))
                & (lin[:, 0] >= 0.0) & (lin[:, 0] <= 1.50)
                & (lin[:, 1].abs() <= 0.15) & (lin[:, 2].abs() <= 0.15)
                & (ang.norm(dim=-1) <= 1.0)
            )
            stable_steps = torch.where(rolling_stable, stable_steps + 1, torch.zeros_like(stable_steps))
            stable_latch |= stable_steps >= int(round(0.50 / float(env.step_dt)))
            finite &= (
                torch.isfinite(robot.data.root_link_pos_w).all(dim=-1)
                & torch.isfinite(current_quat).all(dim=-1)
                & ~terminated
            )
            if args.trace_output is not None:
                if population != 1 or rollouts != 1:
                    raise ValueError("--trace-output requires --population 1 --rollouts 1")
                body_sensor = env.scene.sensors["backflip_body_ground_contact"]
                feet_sensor = env.scene.sensors["feet_ground_contact"]
                trace_rows.append({
                    "step": step,
                    "time_s": (step + 1) * float(env.step_dt),
                    "target": target[0].detach().cpu().tolist(),
                    "qpos_root": env.sim.data.qpos[0, :7].detach().cpu().tolist(),
                    "qvel_root": env.sim.data.qvel[0, :6].detach().cpu().tolist(),
                    "entity_root_pos": robot.data.root_link_pos_w[0].detach().cpu().tolist(),
                    "entity_root_quat": robot.data.root_link_quat_w[0].detach().cpu().tolist(),
                    "servo_qpos": env.sim.data.qpos[0, servo_qpos].detach().cpu().tolist(),
                    "body_sensor_found": body_sensor.data.found[0].detach().cpu().reshape(-1).tolist(),
                    "feet_sensor_found": feet_sensor.data.found[0].detach().cpu().reshape(-1).tolist(),
                    "body_now": bool(body_now[0].item()),
                    "both_contact": bool(both_contact[0].item()),
                    "both_airborne": bool(both_airborne[0].item()),
                    "takeoff": bool(takeoff[0].item()),
                    "forward_rotation_deg": math.degrees(float(forward_rotation[0].item())),
                    "clean_forward_rotation_deg": math.degrees(float(clean_forward_rotation[0].item())),
                })

    drift = (robot.data.root_link_pos_w[:, 1] - start_xy[:, 1]).abs()
    clean = ~any_body
    min_jaw_height = torch.nan_to_num(
        min_jaw_height, nan=0.0, posinf=0.0, neginf=0.0
    )
    min_head_reference_height = torch.nan_to_num(
        min_head_reference_height, nan=0.0, posinf=0.0, neginf=0.0
    )
    clean_rotation_score = torch.clamp(
        max_clean_rotation / MIN_LANDING_ROTATION, 0.0, 1.4
    )
    total_rotation_score = torch.clamp(
        max_rotation / MIN_LANDING_ROTATION, 0.0, 1.4
    )
    clearance_score = torch.clamp(peak_clearance / 0.05, 0.0, 1.5)
    clean_clearance_score = torch.clamp(
        peak_clean_clearance / 0.05, 0.0, 1.5
    )
    head_safety_score = torch.clamp(
        (min_head_reference_height - 0.008) / 0.055, 0.0, 1.25
    )
    first_body_time = first_body_step.clamp_max(steps).float() * float(env.step_dt)
    contact_delay_score = torch.clamp(first_body_time / min(args.duration, 0.80), 0.0, 1.0)
    pitch_rate_score = torch.clamp(peak_clean_pitch_rate / 10.0, 0.0, 1.5)
    vertical_speed_score = torch.clamp(peak_clean_vertical_speed / 0.30, 0.0, 1.5)
    supported_vertical_score = torch.clamp(
        (peak_clean_supported_vertical_speed - 0.05) / 0.45, 0.0, 1.5
    )
    takeoff_vertical_score = torch.clamp(
        takeoff_vertical_velocity / 0.40, 0.0, 1.5
    )
    takeoff_pitch_score = torch.clamp(takeoff_pitch_rate / 10.0, 0.0, 1.5)
    clean_flight_time = clean_flight_steps.float() * float(env.step_dt)
    clean_flight_time_score = torch.clamp(clean_flight_time / 0.50, 0.0, 1.5)
    fitness_env = (
        9.0 * clean_rotation_score + total_rotation_score
        + 3.0 * clean_clearance_score + takeoff.float()
        + 3.0 * (~pre_takeoff_body).float() + 5.0 * clean.float()
        + 5.0 * head_safety_score
        + 5.0 * contact_delay_score + 6.0 * clean_takeoff.float()
        + 2.0 * pitch_rate_score + 2.0 * vertical_speed_score
        + 5.0 * supported_vertical_score + 7.0 * takeoff_vertical_score
        + 3.0 * takeoff_pitch_score
        + 5.0 * clean_flight_time_score
        + 4.0 * landed.float() + 14.0 * stable_latch.float()
        + 3.0 * ((max_rotation >= MIN_LANDING_ROTATION) & clean).float()
        - 0.75 * torch.clamp(drift / 0.12, 0.0, 3.0)
        - 0.75 * torch.clamp(offaxis_rotation / math.radians(60.0), 0.0, 3.0)
        - 25.0 * (~finite).float()
    )
    shape = (population, rollouts)
    grouped = fitness_env.reshape(shape)
    robust_fitness = grouped.mean(dim=1) - 0.30 * grouped.std(dim=1, unbiased=False)

    launch_env = (
        10.0 * clean_rotation_score
        + 8.0 * clean_takeoff.float()
        + 4.0 * clean_clearance_score
        + 3.0 * pitch_rate_score
        + 2.0 * vertical_speed_score
        + 7.0 * supported_vertical_score
        + 10.0 * takeoff_vertical_score
        + 4.0 * takeoff_pitch_score
        + 5.0 * contact_delay_score
        + 7.0 * clean_flight_time_score
        + 3.0 * (~pre_takeoff_body).float()
        - 1.0 * torch.clamp(drift / 0.12, 0.0, 3.0)
        - 1.0 * torch.clamp(offaxis_rotation / math.radians(60.0), 0.0, 3.0)
        - 25.0 * (~finite).float()
    )
    launch_grouped = launch_env.reshape(shape)
    robust_launch_fitness = launch_grouped.mean(dim=1) - 0.50 * launch_grouped.std(
        dim=1, unbiased=False
    )
    jump_env = (
        20.0 * clean.float()
        + 15.0 * (~pre_takeoff_body).float()
        + 25.0 * clean_takeoff.float()
        + 25.0 * supported_vertical_score
        + 35.0 * takeoff_vertical_score
        + 5.0 * takeoff_pitch_score
        + 5.0 * contact_delay_score
        + 3.0 * clean_clearance_score
        + 30.0 * clean_flight_time_score
        - 40.0 * any_body.float()
        - 40.0 * pre_takeoff_body.float()
        - 0.5 * torch.clamp(drift / 0.12, 0.0, 3.0)
        - 25.0 * (~finite).float()
    )
    jump_grouped = jump_env.reshape(shape)
    robust_jump_fitness = jump_grouped.mean(dim=1) - 0.50 * jump_grouped.std(
        dim=1, unbiased=False
    )

    def mean(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(shape).float().mean(dim=1)

    def minimum(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(shape).float().amin(dim=1)

    def maximum(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(shape).float().amax(dim=1)

    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.trace_output, {
            "backend": "mjlab-mujoco-warp",
            "control_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.sim.model.opt.timestep.item()),
            "wheel_frictionloss": args.wheel_friction,
            "current_limit_a": args.current_limit_a,
            "frames": trace_rows,
        })

    return {
        "fitness": robust_fitness,
        "launch_fitness": robust_launch_fitness,
        "jump_fitness": robust_jump_fitness,
        "rotation_deg": mean(max_rotation) * (180.0 / math.pi),
        "minimum_rotation_deg": minimum(max_rotation) * (180.0 / math.pi),
        "clean_rotation_deg": mean(max_clean_rotation) * (180.0 / math.pi),
        "minimum_clean_rotation_deg": minimum(max_clean_rotation) * (180.0 / math.pi),
        "clearance_m": mean(peak_clearance),
        "clean_clearance_m": minimum(peak_clean_clearance),
        "clean_flight_time_s": minimum(clean_flight_time),
        "offaxis_deg": maximum(offaxis_rotation) * (180.0 / math.pi),
        "drift_m": maximum(drift),
        "takeoff_rate": mean(takeoff),
        "clean_takeoff_rate": mean(clean_takeoff),
        "minimum_clean_takeoff": minimum(clean_takeoff),
        "peak_clean_pitch_rate_rad_s": minimum(peak_clean_pitch_rate),
        "peak_clean_vertical_speed_mps": minimum(peak_clean_vertical_speed),
        "peak_clean_supported_vertical_speed_mps": minimum(
            peak_clean_supported_vertical_speed
        ),
        "takeoff_vertical_velocity_mps": minimum(takeoff_vertical_velocity),
        "takeoff_pitch_rate_rad_s": minimum(takeoff_pitch_rate),
        "landing_rate": mean(landed),
        "landing_signed_pitch_deg": mean(landing_signed_pitch_deg),
        "landing_tilt_deg": maximum(landing_tilt_deg),
        "landing_pitch_rate_rad_s": maximum(landing_pitch_rate),
        "landing_vertical_speed_mps": maximum(landing_vertical_speed),
        "landing_forward_speed_mps": minimum(landing_forward_speed),
        "landing_body_speed": maximum(landing_body_speed),
        "clean_rate": mean(clean),
        "pre_takeoff_clean_rate": mean(~pre_takeoff_body),
        "first_body_contact_time_s": minimum(first_body_time),
        "minimum_jaw_height_m": minimum(min_jaw_height),
        "minimum_head_reference_height_m": minimum(min_head_reference_height),
        "stable_rate": mean(stable_latch),
        "finite_rate": mean(finite),
    }


def candidate_record(generation: int, index: int, candidates: torch.Tensor, metrics: dict[str, torch.Tensor], default: torch.Tensor) -> dict:
    record = {"generation": generation, "candidate": index}
    for name, values in metrics.items():
        record[name] = float(values[index].item())
    params = candidates[index].detach()
    record["params"] = params.cpu().tolist()
    record["full_nodes"] = expand_nodes(params.unsqueeze(0), default)[0].cpu().tolist()
    return record


def main() -> None:
    args = parse_args()
    if not math.isclose(args.wheel_friction, 0.003, abs_tol=1.0e-12):
        raise SystemExit("This search is locked to exact wheel friction 0.003")
    if not math.isclose(args.current_limit_a, 1.75, abs_tol=1.0e-12):
        raise SystemExit("This search is locked to the 1.75 A current limit")
    simulation_batch = args.simulation_batch or args.population
    if simulation_batch <= 0 or simulation_batch > args.population:
        raise SystemExit("--simulation-batch must be in [1, population]")
    if args.population % simulation_batch != 0:
        raise SystemExit("population must be divisible by --simulation-batch")
    if simulation_batch * args.rollouts > args.max_envs:
        raise SystemExit(
            "simulation-batch * rollouts must be <= "
            f"--max-envs ({args.max_envs})"
        )
    if len(args.start_speeds) != args.rollouts:
        raise SystemExit("provide exactly one start speed per rollout")

    reference = json.loads(args.reference.read_text())
    if not math.isclose(float(reference["wheel_frictionloss"]), 0.003, abs_tol=1.0e-12):
        raise SystemExit("reference friction mismatch")
    if not math.isclose(float(reference["current_limit_a"]), 1.75, abs_tol=1.0e-12):
        raise SystemExit("reference current-limit mismatch")
    knot_times = torch.tensor(reference["knot_times_s"], device=args.device, dtype=torch.float32)
    source_nodes = torch.tensor(
        reference.get("max_rotation_full_nodes", reference["full_nodes"]),
        device=args.device,
        dtype=torch.float32,
    )
    default = source_nodes[0].clone()
    seed_nodes = [source_nodes]
    for seed_path in args.seed_reference:
        seed_payload = json.loads(seed_path.read_text())
        if not math.isclose(
            float(seed_payload["wheel_frictionloss"]), 0.003, abs_tol=1.0e-12
        ):
            raise SystemExit(f"{seed_path}: seed friction mismatch")
        if not math.isclose(
            float(seed_payload["current_limit_a"]), 1.75, abs_tol=1.0e-12
        ):
            raise SystemExit(f"{seed_path}: seed current-limit mismatch")
        if not torch.allclose(
            torch.as_tensor(
                seed_payload["knot_times_s"], device=args.device,
                dtype=torch.float32,
            ),
            knot_times,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise SystemExit(f"{seed_path}: seed knot-time mismatch")
        candidate_nodes = torch.as_tensor(
            seed_payload.get(
                "max_rotation_full_nodes", seed_payload["full_nodes"]
            ),
            device=args.device,
            dtype=torch.float32,
        )
        if not torch.allclose(candidate_nodes[0], default, atol=1.0e-6, rtol=0.0):
            raise SystemExit(f"{seed_path}: seed default-pose mismatch")
        seed_nodes.append(candidate_nodes)
    seed_params = torch.stack([compact_from_nodes(nodes) for nodes in seed_nodes])
    neutral_nodes = default.view(1, 14).repeat(INTERIOR_KNOTS + 2, 1)
    neutral_params = compact_from_nodes(neutral_nodes)
    mean = compact_from_nodes(source_nodes)
    std = torch.full_like(mean, 0.35).reshape(INTERIOR_KNOTS, COMPACT_DOF)
    std[:, 5:7] = 0.55
    std = std.reshape(-1)
    minimum_std = torch.full_like(std, 0.060)
    minimum_std.reshape(INTERIOR_KNOTS, COMPACT_DOF)[:, 5:7] = 0.080

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    start_generation = 0
    best_overall = None
    best_clean = None
    best_clean_launch = None
    best_jump = None
    incumbent_jump = None
    gpu_archive: list[dict] = []
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        mean = state["mean"].to(device)
        std = state["std"].to(device)
        start_generation = int(state["generation"]) + 1
        generator.set_state(state["generator_state"])
        best_overall = state.get("best_overall")
        best_clean = state.get("best_clean")
        best_clean_launch = state.get("best_clean_launch")
        incumbent_jump = state.get("best_jump")
        gpu_archive = state.get("gpu_archive", [])
        best_jump = incumbent_jump
        previous_duration = state.get("duration")
        if previous_duration is None:
            previous_duration = (
                incumbent_jump.get("first_body_contact_time_s")
                if incumbent_jump is not None else args.duration
            )
        if not math.isclose(
            float(previous_duration), args.duration, abs_tol=1.0e-6
        ):
            # Keep the former winner as a seed, but re-score it before calling
            # it the incumbent under a longer clean-contact horizon.
            best_jump = None
        if (
            best_clean_launch is not None
            and best_clean_launch.get("takeoff_vertical_velocity_mps", 0.0)
            < MIN_TAKEOFF_VERTICAL_SPEED
        ):
            best_clean_launch = None

    cfg = make_microduck_roller_frontflip_ballistic_env_cfg(play=True)
    cfg.scene.num_envs = simulation_batch * args.rollouts
    cfg.scene.env_spacing = 0.0
    cfg.episode_length_s = args.duration + 1.0
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("reset_backflip_state", None)
    cfg.events.pop("randomize_joint_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("expand_bam_friction_fields", None)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1385, 0.1385)
    cfg.terminations.pop("non_skate_ground_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.actions["joint_pos"].scale = 1.0
    robot_cfg = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = dataclasses.replace(
        robot_cfg,
        articulation=dataclasses.replace(
            robot_cfg.articulation,
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=(r"^(?!passive_).*$",),
                    delay_min_lag=0,
                    delay_max_lag=0,
                    delay_per_env_phase=False,
                ),
            ),
        ),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    torque_limit = apply_current_limit(env, args.current_limit_a)
    print(json.dumps({
        "event": "start", "device": args.device,
        "gpu": torch.cuda.get_device_name(device),
        "population": args.population, "rollouts": args.rollouts,
        "simulation_batch": simulation_batch,
        "simulation_environment_count": simulation_batch * args.rollouts,
        "rollout_episodes_per_generation": args.population * args.rollouts,
        "generations": args.generations, "wheel_frictionloss": args.wheel_friction,
        "current_limit_a": args.current_limit_a, "torque_limit_nm": torque_limit,
        "start_speeds_mps": args.start_speeds,
        "seed_references": [str(path.resolve()) for path in args.seed_reference],
        "environment_count": simulation_batch * args.rollouts,
        "max_envs": args.max_envs,
        "pre_takeoff_body_contact_is_invalid": True,
        "minimum_takeoff_vertical_speed_mps": MIN_TAKEOFF_VERTICAL_SPEED,
    }, sort_keys=True), flush=True)

    elite_count = max(8, int(round(args.population * args.elite_fraction)))
    try:
        for generation in range(start_generation, args.generations):
            started = time.monotonic()
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                args.population, PARAM_DIM, device=device, generator=generator
            )
            candidates = clamp_params(candidates)
            candidates[0] = compact_from_nodes(source_nodes)
            if args.population > 1:
                candidates[1] = clamp_params(mean.unsqueeze(0).clone())[0]
            if args.population > 2:
                # Always provide a known collision-free anchor.  Without it,
                # every sample can inherit the source primitive's 0.24 s jaw
                # plant and CEM has no feasible basin from which to improve.
                candidates[2] = neutral_params
            # Every known physically useful lineage is replayed each
            # generation.  This makes regressions visible and prevents a CEM
            # distribution update from deleting a rare launch or tuck family.
            seed_anchor_start = 3
            seed_anchor_count = min(
                len(seed_params), max(0, args.population - seed_anchor_start)
            )
            if seed_anchor_count:
                candidates[
                    seed_anchor_start:seed_anchor_start + seed_anchor_count
                ] = seed_params[:seed_anchor_count]

            source_compact = compact_from_nodes(source_nodes).reshape(
                INTERIOR_KNOTS, COMPACT_DOF
            )
            neutral_compact = neutral_params.reshape(
                INTERIOR_KNOTS, COMPACT_DOF
            )

            # Family A delays the head whip.  Half keeps a neutral preload;
            # half borrows only the old primitive's strong crouch/extension
            # leg prefix while removing the jaw-driving head prefix.
            family_a_start = min(
                args.population, seed_anchor_start + seed_anchor_count
            )
            family_a_end = min(
                args.population, family_a_start + max(32, args.population // 4)
            )
            if family_a_end > family_a_start:
                family_a = candidates[family_a_start:family_a_end].reshape(
                    -1, INTERIOR_KNOTS, COMPACT_DOF
                )
                prefix_noise = torch.randn(
                    family_a.shape[0], 2, COMPACT_DOF,
                    device=device, generator=generator,
                )
                prefix_scale = torch.tensor(
                    [0.06, 0.06, 0.18, 0.18, 0.18, 0.08, 0.08],
                    device=device,
                )
                family_a[:, 0:2, :] = (
                    neutral_compact[0].view(1, 1, COMPACT_DOF)
                    + prefix_noise * prefix_scale.view(1, 1, COMPACT_DOF)
                )
                structured_start = family_a.shape[0] // 2
                if structured_start < family_a.shape[0]:
                    family_a[structured_start:, 0:2, :5] = (
                        source_compact[0:2, :5].view(1, 2, 5)
                        + 0.16
                        * torch.randn(
                            family_a.shape[0] - structured_start, 2, 5,
                            device=device, generator=generator,
                        )
                    )
                    family_a[structured_start:, 0:3, 5:7] = (
                        default[5:7].view(1, 1, 2)
                        + 0.10
                        * torch.randn(
                            family_a.shape[0] - structured_start, 3, 2,
                            device=device, generator=generator,
                        )
                    )
                # Keep the head delayed through 0.46 s; legs may deliver the
                # impulse at knot 3 without driving the beak into the floor.
                family_a[:, 2, 5:7] = (
                    default[5:7].view(1, 2)
                    + 0.10
                    * torch.randn(
                        family_a.shape[0], 2,
                        device=device, generator=generator,
                    )
                )

            # Family B starts from an entirely neutral trajectory and widens
            # motion only after preload.  This protects a second, independent
            # route to a clean launch instead of merely repairing the exploit.
            family_b_start = family_a_end
            family_b_end = min(
                args.population, family_b_start + max(32, args.population // 4)
            )
            if family_b_end > family_b_start:
                count = family_b_end - family_b_start
                neutral_noise = torch.randn(
                    count, INTERIOR_KNOTS, COMPACT_DOF,
                    device=device, generator=generator,
                )
                time_scale = torch.tensor(
                    [0.06, 0.10, 0.28, 0.45, 0.45, 0.30], device=device
                ).view(1, INTERIOR_KNOTS, 1)
                dof_scale = torch.tensor(
                    [0.45, 0.45, 1.0, 1.0, 1.0, 1.1, 1.1], device=device
                ).view(1, 1, COMPACT_DOF)
                candidates[family_b_start:family_b_end] = (
                    neutral_params.view(1, INTERIOR_KNOTS, COMPACT_DOF)
                    + neutral_noise * time_scale * dof_scale
                ).reshape(count, -1)

            # Family C explicitly preserves and mutates the best physically
            # valid jump found so far.  On a fresh longer-horizon run the
            # packaged reference is the incumbent until a candidate proves
            # cleaner.  Mutate the post-preload knots much more strongly than
            # the launch prefix: we want to retain the known rolling jump while
            # discovering a jaw-clear tuck and forward rotation continuation.
            family_c_start = family_b_end
            family_c_end = min(
                args.population, family_c_start + max(32, args.population // 4)
            )
            if family_c_end > family_c_start:
                if incumbent_jump is not None:
                    incumbent_params = torch.as_tensor(
                        incumbent_jump["params"], device=device, dtype=torch.float32
                    )
                else:
                    incumbent_params = compact_from_nodes(source_nodes)
                count = family_c_end - family_c_start
                incumbent_noise = torch.randn(
                    count, INTERIOR_KNOTS, COMPACT_DOF,
                    device=device, generator=generator
                )
                suffix_scale = torch.tensor(
                    [0.025, 0.035, 0.070, 0.260, 0.420, 0.420],
                    device=device,
                ).view(1, INTERIOR_KNOTS, 1)
                dof_scale = torch.tensor(
                    [0.65, 0.65, 1.0, 1.0, 1.0, 1.15, 1.15],
                    device=device,
                ).view(1, 1, COMPACT_DOF)
                candidates[family_c_start:family_c_end] = (
                    incumbent_params.view(1, INTERIOR_KNOTS, COMPACT_DOF)
                    + incumbent_noise * suffix_scale * dof_scale
                ).reshape(count, -1)
                candidates[family_c_start] = incumbent_params
                if best_clean is not None and count >= 4:
                    # Keep a separate forward-rotation lineage alive.  Jump
                    # fitness alone strongly prefers vertical, zero-pitch
                    # motion and otherwise erases the rare rotating samples.
                    rotation_params = torch.as_tensor(
                        best_clean["params"], device=device, dtype=torch.float32
                    )
                    rotation_offset = count // 2
                    rotation_count = count - rotation_offset
                    rotation_noise = torch.randn(
                        rotation_count, INTERIOR_KNOTS, COMPACT_DOF,
                        device=device, generator=generator,
                    )
                    candidates[
                        family_c_start + rotation_offset:family_c_end
                    ] = (
                        rotation_params.view(1, INTERIOR_KNOTS, COMPACT_DOF)
                        + rotation_noise * suffix_scale * dof_scale
                    ).reshape(rotation_count, -1)
                    candidates[family_c_start + rotation_offset] = rotation_params

            # Family D is the deliberate phase-stitch population.  It takes
            # the preload/launch prefix from one protected primitive and the
            # airborne tuck/opening suffix from another.  A one-knot linear
            # bridge avoids discontinuous servo targets.  This is much higher
            # leverage than merely widening Gaussian noise around one gait.
            family_d_start = family_c_end
            family_d_end = args.population
            if family_d_end > family_d_start and len(seed_params) >= 2:
                count = family_d_end - family_d_start
                family_d = candidates[family_d_start:family_d_end].reshape(
                    count, INTERIOR_KNOTS, COMPACT_DOF
                )
                left_ids = torch.randint(
                    len(seed_params), (count,), device=device, generator=generator
                )
                right_ids = torch.randint(
                    len(seed_params), (count,), device=device, generator=generator
                )
                split_ids = torch.randint(
                    1, INTERIOR_KNOTS, (count,), device=device,
                    generator=generator,
                )
                seed_shaped = seed_params.reshape(
                    len(seed_params), INTERIOR_KNOTS, COMPACT_DOF
                )
                knot_index = torch.arange(
                    INTERIOR_KNOTS, device=device
                ).view(1, INTERIOR_KNOTS, 1)
                prefix = seed_shaped[left_ids]
                suffix = seed_shaped[right_ids]
                stitched = torch.where(
                    knot_index < split_ids.view(-1, 1, 1), prefix, suffix
                )
                bridge = split_ids.clamp(max=INTERIOR_KNOTS - 1)
                row_ids = torch.arange(count, device=device)
                stitched[row_ids, bridge] = (
                    0.45 * prefix[row_ids, bridge]
                    + 0.55 * suffix[row_ids, bridge]
                )
                phase_noise = torch.randn(
                    count, INTERIOR_KNOTS, COMPACT_DOF,
                    device=device, generator=generator,
                )
                phase_scale = torch.tensor(
                    [0.035, 0.055, 0.12, 0.20, 0.28, 0.22],
                    device=device,
                ).view(1, INTERIOR_KNOTS, 1)
                dof_scale = torch.tensor(
                    [0.55, 0.55, 1.0, 1.0, 1.0, 0.85, 0.85],
                    device=device,
                ).view(1, 1, COMPACT_DOF)
                family_d[:] = stitched + phase_noise * phase_scale * dof_scale
                # Deterministic pairwise splices make the full seed cross
                # product observable even when random sampling is unlucky.
                anchor = 0
                for left in range(len(seed_params)):
                    for right in range(len(seed_params)):
                        for split in (2, 3, 4):
                            if anchor >= count:
                                break
                            family_d[anchor, :split] = seed_shaped[left, :split]
                            family_d[anchor, split:] = seed_shaped[right, split:]
                            if split < INTERIOR_KNOTS:
                                family_d[anchor, split] = (
                                    0.45 * seed_shaped[left, split]
                                    + 0.55 * seed_shaped[right, split]
                                )
                            anchor += 1
            candidates = clamp_params(candidates)
            metric_chunks: dict[str, list[torch.Tensor]] = {}
            for batch_start in range(0, args.population, simulation_batch):
                batch_end = batch_start + simulation_batch
                batch_metrics = evaluate_population(
                    env,
                    candidates[batch_start:batch_end],
                    knot_times,
                    default,
                    args,
                )
                for name, values in batch_metrics.items():
                    metric_chunks.setdefault(name, []).append(values)
            metrics = {
                name: torch.cat(chunks, dim=0)
                for name, chunks in metric_chunks.items()
            }
            if args.trace_output is not None:
                record = candidate_record(
                    generation, 0, candidates, metrics, default
                )
                print(json.dumps({
                    "event": "trace",
                    **{key: value for key, value in record.items()
                       if key not in ("params", "full_nodes")},
                    "trace_output": str(args.trace_output.resolve()),
                }, sort_keys=True), flush=True)
                break
            primary_count = elite_count // 2
            safety_count = elite_count - primary_count
            if args.duration <= 0.65:
                clean_jump = metrics["clean_rate"] >= 1.0
                primary_metric = torch.where(
                    clean_jump,
                    metrics["jump_fitness"],
                    torch.full_like(metrics["jump_fitness"], -1.0e6),
                )
            elif args.duration <= 1.25:
                # This is the launch-to-rotation curriculum.  Once the batch
                # contains any robust, contact-free takeoff, it becomes a hard
                # feasibility constraint.  Rotation is credited only before
                # body contact; a fast jaw plant cannot win this stage.
                robust_clean_launch = (
                    (metrics["clean_rate"] >= 1.0)
                    & (metrics["minimum_clean_takeoff"] >= 1.0)
                    & (metrics["takeoff_vertical_velocity_mps"] >= 0.08)
                    & (metrics["clean_clearance_m"] >= 0.035)
                    & (metrics["clean_flight_time_s"] >= 0.08)
                )
                clean_launch_score = (
                    4.0 * metrics["minimum_clean_rotation_deg"]
                    + 1.0 * metrics["clean_rotation_deg"]
                    + 0.05 * metrics["jump_fitness"]
                    + 0.75 * metrics["peak_clean_pitch_rate_rad_s"]
                    - 0.15 * metrics["offaxis_deg"]
                    - 2.0 * torch.clamp(metrics["drift_m"] / 0.12, 0.0, 3.0)
                )
                if bool(robust_clean_launch.any()):
                    primary_metric = torch.where(
                        robust_clean_launch,
                        clean_launch_score,
                        torch.full_like(clean_launch_score, -1.0e6),
                    )
                else:
                    # Before the first fully feasible continuation exists,
                    # reward progressively longer clean flight while retaining
                    # the known positive-vertical-velocity rolling launch.
                    primary_metric = (
                        metrics["launch_fitness"]
                        + 20.0 * metrics["clean_rate"]
                        + 20.0 * metrics["minimum_clean_takeoff"]
                        + 20.0 * metrics["clean_flight_time_s"]
                    )
            else:
                primary_metric = metrics["fitness"]
            primary = torch.topk(primary_metric, primary_count).indices
            safety_fitness = (
                metrics["clean_rotation_deg"] / 30.0
                + metrics["rotation_deg"] / 120.0
                + 100.0
                * metrics["minimum_head_reference_height_m"].clamp(0.0, 0.12)
                + 8.0 * metrics["pre_takeoff_clean_rate"]
                + 4.0 * metrics["clean_rate"]
            )
            if args.duration <= 0.65:
                safety_fitness = (
                    metrics["jump_fitness"]
                    + 1.0e4 * (metrics["clean_rate"] >= 1.0).float()
                )
            elif args.duration <= 1.25 and bool(robust_clean_launch.any()):
                # The other half of the elite set protects launch authority;
                # this prevents the rotation lineage from collapsing vertical
                # impulse or clearance while the primary half chases pitch.
                safety_fitness = torch.where(
                    robust_clean_launch,
                    metrics["jump_fitness"]
                    + 0.30 * metrics["minimum_clean_rotation_deg"]
                    - 0.20 * metrics["offaxis_deg"],
                    torch.full_like(metrics["jump_fitness"], -1.0e6),
                )
            safety = torch.topk(safety_fitness, safety_count).indices
            elite_indices = torch.cat((primary, safety))
            elite = candidates[elite_indices]
            elite_mean = elite.mean(dim=0)
            elite_std = elite.std(dim=0, unbiased=False).clamp_min(minimum_std)
            mean = (1.0 - args.update_rate) * mean + args.update_rate * elite_mean
            std = ((1.0 - args.update_rate) * std + args.update_rate * elite_std).clamp_min(minimum_std)
            mean = clamp_params(mean.unsqueeze(0))[0]

            top_index = int(metrics["fitness"].argmax().item())
            top = candidate_record(generation, top_index, candidates, metrics, default)
            clean_mask = metrics["clean_rate"] >= 1.0
            if bool(clean_mask.any()):
                clean_scores = torch.where(clean_mask, metrics["rotation_deg"], torch.full_like(metrics["rotation_deg"], -1.0))
                clean_index = int(clean_scores.argmax().item())
                clean_record = candidate_record(generation, clean_index, candidates, metrics, default)
                if best_clean is None or clean_record["minimum_rotation_deg"] > best_clean["minimum_rotation_deg"]:
                    best_clean = clean_record
                    save_json(output_dir / "best-clean.json", best_clean)
            if bool(clean_mask.any()):
                jump_scores = torch.where(
                    clean_mask,
                    metrics["jump_fitness"],
                    torch.full_like(metrics["jump_fitness"], -1.0e6),
                )
                jump_index = int(jump_scores.argmax().item())
                jump_record = candidate_record(
                    generation, jump_index, candidates, metrics, default
                )
                if (
                    best_jump is None
                    or jump_record["jump_fitness"] > best_jump["jump_fitness"]
                ):
                    best_jump = jump_record
                    incumbent_jump = best_jump
                    save_json(output_dir / "best-jump.json", best_jump)
            robust_launch = metrics["minimum_clean_takeoff"] >= 1.0
            if bool(robust_launch.any()):
                launch_priority = torch.where(
                    robust_launch,
                    metrics["minimum_clean_rotation_deg"],
                    torch.full_like(metrics["minimum_clean_rotation_deg"], -1.0),
                )
                launch_index = int(launch_priority.argmax().item())
            else:
                launch_index = int(metrics["launch_fitness"].argmax().item())
            launch_record = candidate_record(
                generation, launch_index, candidates, metrics, default
            )
            if (
                best_clean_launch is None
                or launch_record["minimum_clean_rotation_deg"]
                > best_clean_launch["minimum_clean_rotation_deg"]
            ):
                best_clean_launch = launch_record
                save_json(
                    output_dir / "best-clean-launch.json", best_clean_launch
                )
            if best_overall is None or top["fitness"] > best_overall["fitness"]:
                best_overall = top
                save_json(output_dir / "best-overall.json", best_overall)

            # Preserve a broad proposal archive for later native-MuJoCo
            # rejection.  Warp is deliberately only the high-throughput
            # proposer here, so we retain distinct high-rotation, high-lift,
            # delayed-contact and line-holding candidates instead of trusting
            # a single simulator-specific scalar winner.
            proposal_score = (
                5.0 * metrics["minimum_clean_rotation_deg"]
                + 1800.0 * metrics["clean_clearance_m"]
                + 300.0 * metrics["takeoff_vertical_velocity_mps"]
                + 120.0 * metrics["first_body_contact_time_s"]
                + 2.0 * metrics["peak_clean_pitch_rate_rad_s"]
                - 400.0 * metrics["drift_m"]
                - 2.0 * metrics["offaxis_deg"]
                + 150.0 * metrics["minimum_clean_takeoff"]
                + 100.0 * metrics["pre_takeoff_clean_rate"]
            )
            proposal_score = torch.where(
                metrics["finite_rate"] >= 1.0,
                proposal_score,
                torch.full_like(proposal_score, -1.0e9),
            )
            per_reason = min(8, args.population)
            line_score = (
                metrics["minimum_clean_rotation_deg"]
                - 800.0 * metrics["drift_m"]
                - 3.0 * metrics["offaxis_deg"]
            )
            lift_score = (
                1800.0 * metrics["clean_clearance_m"]
                + 350.0 * metrics["takeoff_vertical_velocity_mps"]
                + 100.0 * metrics["clean_flight_time_s"]
            )
            delay_score = (
                180.0 * metrics["first_body_contact_time_s"]
                + 100.0 * metrics["clean_flight_time_s"]
                + metrics["minimum_clean_rotation_deg"]
            )
            impulse_score = (
                8.0 * metrics["peak_clean_pitch_rate_rad_s"]
                + 250.0 * metrics["takeoff_vertical_velocity_mps"]
                + metrics["minimum_clean_rotation_deg"]
            )
            selectors = {
                "balanced": (proposal_score, per_reason),
                "rotation": (metrics["minimum_clean_rotation_deg"], per_reason),
                "lift": (lift_score, per_reason),
                "contact_delay": (delay_score, per_reason),
                "line_hold": (line_score, per_reason),
                "angular_impulse": (impulse_score, per_reason),
            }
            for reason, (reason_score, take) in selectors.items():
                for proposal_index in torch.topk(reason_score, take).indices.tolist():
                    item = candidate_record(
                        generation, proposal_index, candidates, metrics, default
                    )
                    item["gpu_archive_reason"] = reason
                    item["gpu_archive_value"] = float(
                        reason_score[proposal_index].item()
                    )
                    item["gpu_proposal_score"] = float(
                        proposal_score[proposal_index].item()
                    )
                    gpu_archive.append(item)
            random_take = min(8, args.population)
            for proposal_index in torch.randperm(
                args.population, device=device, generator=generator
            )[:random_take].tolist():
                item = candidate_record(
                    generation, proposal_index, candidates, metrics, default
                )
                item["gpu_archive_reason"] = "random_diversity"
                item["gpu_archive_value"] = float(generation)
                item["gpu_proposal_score"] = float(
                    proposal_score[proposal_index].item()
                )
                gpu_archive.append(item)
            protected_indices = [0, 2] + list(
                range(seed_anchor_start, seed_anchor_start + seed_anchor_count)
            )
            for proposal_index in protected_indices:
                if proposal_index >= args.population:
                    continue
                item = candidate_record(
                    generation, proposal_index, candidates, metrics, default
                )
                item["gpu_archive_reason"] = "protected_anchor"
                item["gpu_archive_value"] = float(
                    metrics["minimum_clean_rotation_deg"][proposal_index].item()
                )
                item["gpu_proposal_score"] = float(
                    proposal_score[proposal_index].item()
                )
                gpu_archive.append(item)
            reason_groups: dict[str, list[dict]] = {}
            for item in gpu_archive:
                reason_groups.setdefault(
                    str(item.get("gpu_archive_reason", "balanced")), []
                ).append(item)
            gpu_archive = []
            for reason, items in reason_groups.items():
                items.sort(
                    key=lambda item: float(item.get("gpu_archive_value", -1.0e9)),
                    reverse=True,
                )
                gpu_archive.extend(items[:64])
            gpu_archive = gpu_archive[:512]
            save_json(output_dir / "gpu-elite-archive.json", {
                "backend": "mjlab-mujoco-warp-proposals-only",
                "wheel_frictionloss": args.wheel_friction,
                "current_limit_a": args.current_limit_a,
                "start_speeds_mps": args.start_speeds,
                "generation": generation,
                "candidates": gpu_archive,
            })

            state = {
                "generation": generation, "mean": mean.detach().cpu(),
                "std": std.detach().cpu(), "generator_state": generator.get_state(),
                "best_overall": best_overall, "best_clean": best_clean,
                "best_clean_launch": best_clean_launch,
                "best_jump": best_jump,
                "gpu_archive": gpu_archive,
                "duration": args.duration,
            }
            torch.save(state, output_dir / "search-state.pt")
            report = {
                "event": "generation", "generation": generation,
                "elapsed_s": time.monotonic() - started,
                "top": {key: value for key, value in top.items() if key not in ("params", "full_nodes")},
                "best_clean": (
                    {key: value for key, value in best_clean.items() if key not in ("params", "full_nodes")}
                    if best_clean else None
                ),
                "best_clean_launch": {
                    key: value for key, value in best_clean_launch.items()
                    if key not in ("params", "full_nodes")
                },
                "best_jump": (
                    {
                        key: value for key, value in best_jump.items()
                        if key not in ("params", "full_nodes")
                    }
                    if best_jump else None
                ),
                "anchors": [
                    {
                        key: value for key, value in candidate_record(
                            generation, anchor, candidates, metrics, default
                        ).items() if key not in ("params", "full_nodes")
                    }
                    for anchor in (0, 2) if anchor < args.population
                ],
            }
            print(json.dumps(report, sort_keys=True), flush=True)
            if (
                top["stable_rate"] >= 1.0 and top["clean_rate"] >= 1.0
                and top["minimum_rotation_deg"] >= 300.0
            ):
                save_json(output_dir / "success.json", top)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
