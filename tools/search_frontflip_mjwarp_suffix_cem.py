#!/usr/bin/env python3
"""GPU-first front-flip suffix search with mandatory native certification.

The detailed roller meshes are not contact-parity compatible during launch in
MuJoCo Warp 3.10.0.2.  This search therefore executes the protected prefix in
native MuJoCo once, copies the exact airborne state into thousands of MuJoCo
Warp worlds, and searches only the flight/landing suffix on the GPU.  Every
reported champion is replayed end-to-end in native MuJoCo before promotion.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from importlib.metadata import version as package_version
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from bam.model import load_model
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_roller_frontflip_ballistic_env_cfg import (
    make_microduck_roller_frontflip_ballistic_env_cfg,
)

import search_frontflip_gpu_cem as gpu
import search_frontflip_native_oc as oc


COMPACT_DOF = 7
ISLAND_NAMES = ("stable", "contact-free", "touchdown", "explore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--population", type=int, default=16384)
    parser.add_argument("--simulation-batch", type=int, default=8192)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--elite-fraction", type=float, default=0.0625)
    parser.add_argument("--update-rate", type=float, default=0.30)
    parser.add_argument("--duration", type=float, default=1.80)
    parser.add_argument("--handoff-time", type=float, default=0.82)
    parser.add_argument("--suffix-start-time", type=float, default=0.84)
    parser.add_argument("--start-speed", type=float, default=1.54)
    parser.add_argument("--wheel-friction", type=float, default=0.003)
    parser.add_argument("--current-limit-a", type=float, default=1.75)
    parser.add_argument("--native-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6500)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_fixed_physics(payload: dict, source: Path) -> None:
    if not math.isclose(
        float(payload["wheel_frictionloss"]), 0.003, abs_tol=1.0e-12
    ):
        raise SystemExit(f"{source}: wheel friction must be exactly 0.003")
    if not math.isclose(
        float(payload["current_limit_a"]), 1.75, abs_tol=1.0e-12
    ):
        raise SystemExit(f"{source}: current limit must be exactly 1.75 A")


def reset_native(context: dict, data: mujoco.MjData, speed: float) -> None:
    model: mujoco.MjModel = context["model"]
    qa, va = context["qpos_adr"], context["qvel_adr"]
    mujoco.mj_resetData(model, data)
    data.qpos[qa : qa + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    data.qvel[va] = speed
    for wheel_dof in context["wheel_dofs"]:
        data.qvel[wheel_dof] = speed / oc.WHEEL_RADIUS
    data.qpos[context["actuator_qpos"]] = context["default"]
    data.ctrl[:] = context["default"]
    mujoco.mj_forward(model, data)


def native_handoff(
    scene: Path,
    nodes: np.ndarray,
    speed: float,
    handoff_time: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    oc.make_context(str(scene.resolve()), nodes[0].tolist(), duration)
    context = oc._TLS.context
    model: mujoco.MjModel = context["model"]
    data = mujoco.MjData(model)
    reset_native(context, data, speed)
    stride = int(round(oc.CONTROL_DT / oc.PHYSICS_DT))
    target = nodes[0]
    previous_quat = data.qpos[
        context["qpos_adr"] + 3 : context["qpos_adr"] + 7
    ].copy()
    support_seen = len(gpu_contact_scan(context, data)[0]) == 2
    takeoff = False
    prefix_rotation = 0.0
    for step in range(int(round(handoff_time / oc.PHYSICS_DT))):
        if step % stride == 0:
            target = oc.target_at(step * oc.PHYSICS_DT, nodes)
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        sides, body, _ = gpu_contact_scan(context, data)
        both_grounded = len(sides) == 2
        both_airborne = len(sides) == 0
        support_seen |= both_grounded
        if (
            not takeoff
            and support_seen
            and both_airborne
            and data.qvel[context["qvel_adr"] + 2] > 0.02
            and not body
        ):
            takeoff = True
        current_quat = data.qpos[
            context["qpos_adr"] + 3 : context["qpos_adr"] + 7
        ].copy()
        if takeoff and both_airborne:
            prefix_rotation += max(
                0.0, float(oc.quat_delta_vector(previous_quat, current_quat)[1])
            )
        previous_quat = current_quat
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("native handoff state is non-finite")
    sides, body, names = gpu_contact_scan(context, data)
    if sides or body:
        raise RuntimeError(
            f"handoff at {handoff_time:.3f}s is not airborne: sides={sides}, "
            f"body={body}, names={names}"
        )
    return data.qpos.copy(), data.qvel.copy(), prefix_rotation


def gpu_contact_scan(
    context: dict, data: mujoco.MjData
) -> tuple[set[str], bool, list[str]]:
    view = dict(context)
    view["data"] = data
    return oc.scan_contacts(view)


def suffix_compact(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = nodes.index_select(0, indices)
    return torch.cat((selected[:, :5], selected[:, 5:7]), dim=-1).reshape(-1)


def expand_suffix(
    params: torch.Tensor, base_nodes: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    population = params.shape[0]
    compact = params.reshape(population, len(indices), COMPACT_DOF)
    full = base_nodes.unsqueeze(0).repeat(population, 1, 1)
    full[:, indices, :5] = compact[:, :, :5]
    full[:, indices, 5:7] = compact[:, :, 5:7]
    full[:, indices, 9:14] = -compact[:, :, :5]
    return full


def clamp_suffix(params: torch.Tensor, suffix_count: int) -> torch.Tensor:
    compact = params.reshape(-1, suffix_count, COMPACT_DOF)
    compact[:, :, 0].clamp_(-0.40, 0.40)
    compact[:, :, 1].clamp_(-0.38, 0.38)
    compact[:, :, 2:5].clamp_(-1.50, 1.50)
    compact[:, :, 5].clamp_(-1.50, 1.00)
    compact[:, :, 6].clamp_(-1.50, 1.50)
    return params


def build_env(args: argparse.Namespace, num_envs: int) -> ManagerBasedRlEnv:
    cfg = make_microduck_roller_frontflip_ballistic_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 0.0
    # Resolve tire touchdown and forbidden body contact at the native 5 ms
    # physics cadence.  evaluate_population still holds commands at 50 Hz.
    cfg.decimation = 1
    cfg.episode_length_s = args.duration + 1.0
    cfg.observations["actor"].enable_corruption = False
    for event in (
        "reset_backflip_state",
        "randomize_joint_friction",
        "encoder_bias",
        "base_com",
        "expand_bam_friction_fields",
    ):
        cfg.events.pop(event, None)
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
    motor = load_model(motor_name="xl330", model="m6")
    torque_limit = float(args.current_limit_a * motor.kt.value)
    env.sim.model.actuator_forcerange[..., 0] = -torque_limit
    env.sim.model.actuator_forcerange[..., 1] = torque_limit
    return env


def build_direct_warp_batch(
    args: argparse.Namespace,
    nodes: np.ndarray,
) -> dict:
    """Build the exact direct MuJoCo Warp path used by the parity gate."""
    oc.make_context(str(args.scene.resolve()), nodes[0].tolist(), args.duration)
    context = oc._TLS.context
    host_model: mujoco.MjModel = context["model"]
    host_data = mujoco.MjData(host_model)
    reset_native(context, host_data, args.start_speed)
    control_stride = int(round(oc.CONTROL_DT / oc.PHYSICS_DT))
    target = nodes[0]
    prefix_steps = int(round(args.handoff_time / oc.PHYSICS_DT))
    for step in range(prefix_steps):
        if step % control_stride == 0:
            target = oc.target_at(step * oc.PHYSICS_DT, nodes)
        host_data.ctrl[:] = target
        mujoco.mj_step(host_model, host_data)
    if not np.isfinite(host_data.qpos).all() or not np.isfinite(host_data.qvel).all():
        raise RuntimeError("direct Warp native prefix became non-finite")
    warp_model = mjw.put_model(host_model)
    warp_data = mjw.put_data(
        host_model,
        host_data,
        nworld=args.simulation_batch,
        # Match the independently passing parity harness.  Smaller contact or
        # constraint buffers clip the skate/body impact burst and change the
        # post-touchdown trajectory by several degrees.
        nconmax=256,
        nccdmax=256,
        njmax=512,
        njmax_nnz=8192,
        nvmax=host_model.nv,
    )
    device = torch.device(args.device)
    # MuJoCo Warp and PyTorch must share a CUDA stream.  Otherwise asynchronous
    # Torch state/control writes can race Warp's forward/step kernels.
    wp.set_stream(wp.stream_from_torch(device), device=args.device, sync=True)
    ngeom = host_model.ngeom
    ground = torch.zeros(ngeom, dtype=torch.bool, device=device)
    left = torch.zeros_like(ground)
    right = torch.zeros_like(ground)
    forbidden = torch.zeros_like(ground)
    ground[list(context["ground_geoms"])] = True
    for geom, side in context["wheel_geoms"].items():
        (left if side == "left" else right)[geom] = True
    for geom in range(ngeom):
        forbidden[geom] = (
            int(host_model.geom_bodyid[geom]) in context["forbidden_bodies"]
        )
    return {
        "model": warp_model,
        "data": warp_data,
        "qpos": wp.to_torch(warp_data.qpos),
        "qvel": wp.to_torch(warp_data.qvel),
        "ctrl": wp.to_torch(warp_data.ctrl),
        "nacon": wp.to_torch(warp_data.nacon),
        # Keep live int32 views.  Casting here would create stale snapshots
        # that never see contacts generated by later Warp steps.
        "contact_geom": wp.to_torch(warp_data.contact.geom),
        "contact_world": wp.to_torch(warp_data.contact.worldid),
        "contact_index": torch.arange(
            int(warp_data.contact.worldid.shape[0]), device=device
        ),
        "ground": ground,
        "left": left,
        "right": right,
        "forbidden": forbidden,
        "qpos_adr": int(context["qpos_adr"]),
        "qvel_adr": int(context["qvel_adr"]),
    }


def direct_contact_masks(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return left-tire, right-tire and forbidden-body ground contacts."""
    world_count = batch["qpos"].shape[0]
    geom = batch["contact_geom"]
    active = batch["contact_index"] < batch["nacon"][0]
    g1 = geom[:, 0].long().clamp(0, len(batch["ground"]) - 1)
    g2 = geom[:, 1].long().clamp(0, len(batch["ground"]) - 1)
    ground_1 = batch["ground"][g1]
    ground_2 = batch["ground"][g2]
    # Match the independently passing parity classifier exactly: a contact
    # contributes support for a side whenever one of that side's wheel geoms
    # appears in an active pair.  The scene's collision filters ensure these
    # are wheel/ground contacts during the airborne suffix.
    left_pair = active & (batch["left"][g1] | batch["left"][g2])
    right_pair = active & (batch["right"][g1] | batch["right"][g2])
    body_pair = active & (
        (ground_1 & batch["forbidden"][g2])
        | (ground_2 & batch["forbidden"][g1])
    )
    world = batch["contact_world"].long().clamp(0, world_count - 1)

    def reduce_any(mask: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(world_count, dtype=torch.int32, device=world.device)
        counts.scatter_add_(0, world, mask.to(torch.int32))
        return counts > 0

    return reduce_any(left_pair), reduce_any(right_pair), reduce_any(body_pair)


def evaluate_direct_warp(
    batch: dict,
    nodes: torch.Tensor,
    knot_times: torch.Tensor,
    handoff_time: float,
    duration: float,
    initial_rotation_rad: float,
) -> dict[str, torch.Tensor]:
    """Evaluate one candidate per world with direct 5 ms MuJoCo Warp steps."""
    count = nodes.shape[0]
    if count != batch["qpos"].shape[0]:
        raise ValueError("direct Warp batch must be completely filled")
    qa, va = batch["qpos_adr"], batch["qvel_adr"]
    previous_quat = batch["qpos"][:, qa + 3 : qa + 7].clone()
    start_xy = batch["qpos"][:, qa : qa + 2].clone()
    forward_rotation = torch.full(
        (count,), initial_rotation_rad, dtype=torch.float32, device=nodes.device
    )
    max_rotation = forward_rotation.clone()
    clean_rotation = forward_rotation.clone()
    max_clean_rotation = forward_rotation.clone()
    offaxis = torch.zeros_like(forward_rotation)
    peak_clearance = torch.clamp(
        batch["qpos"][:, qa + 2] - gpu.STAND_HEIGHT, min=0.0
    )
    landed = torch.zeros(count, dtype=torch.bool, device=nodes.device)
    any_body = torch.zeros_like(landed)
    stable_steps = torch.zeros(count, dtype=torch.int32, device=nodes.device)
    stable = torch.zeros_like(landed)
    finite = torch.ones_like(landed)
    first_body_step = torch.full(
        (count,), 10_000, dtype=torch.int32, device=nodes.device
    )
    landing_pitch = torch.full_like(forward_rotation, 90.0)
    landing_tilt = torch.full_like(forward_rotation, 180.0)
    landing_pitch_rate = torch.full_like(forward_rotation, 20.0)
    landing_vertical_speed = torch.full_like(forward_rotation, 20.0)
    landing_forward_speed = torch.zeros_like(forward_rotation)
    landing_body_speed = torch.full_like(forward_rotation, 20.0)
    dt = float(oc.PHYSICS_DT)
    steps = int(round((duration - handoff_time) / dt))
    control_stride = int(round(oc.CONTROL_DT / oc.PHYSICS_DT))

    for step in range(steps):
        absolute_time = handoff_time + step * dt
        if step % control_stride == 0:
            index = int(
                torch.searchsorted(
                    knot_times,
                    torch.tensor(absolute_time, device=nodes.device),
                    right=True,
                ).item()
            ) - 1
            index = max(0, min(index, len(knot_times) - 2))
            span = float(knot_times[index + 1] - knot_times[index])
            blend = max(
                0.0,
                min(
                    1.0,
                    (absolute_time - float(knot_times[index])) / max(span, 1.0e-9),
                ),
            )
            batch["ctrl"].copy_(
                (1.0 - blend) * nodes[:, index] + blend * nodes[:, index + 1]
            )
        mjw.step(batch["model"], batch["data"])
        left, right, body_now = direct_contact_masks(batch)
        both_contact = left & right
        both_airborne = ~left & ~right
        body_seen_before = any_body.clone()
        first_body_step = torch.where(
            body_now & ~any_body,
            torch.full_like(first_body_step, step + 1),
            first_body_step,
        )
        any_body |= body_now
        current_quat = batch["qpos"][:, qa + 3 : qa + 7].clone()
        active_flight = both_airborne & ~landed
        delta = gpu.quat_delta_vector(previous_quat, current_quat)
        positive_pitch = torch.clamp(delta[:, 1], min=0.0)
        forward_rotation += active_flight.float() * positive_pitch
        clean_rotation += (active_flight & ~any_body).float() * positive_pitch
        max_rotation = torch.maximum(max_rotation, forward_rotation)
        max_clean_rotation = torch.maximum(max_clean_rotation, clean_rotation)
        offaxis += active_flight.float() * torch.linalg.vector_norm(
            delta[:, (0, 2)], dim=-1
        )
        peak_clearance = torch.maximum(
            peak_clearance,
            torch.clamp(batch["qpos"][:, qa + 2] - gpu.STAND_HEIGHT, min=0.0),
        )
        previous_quat = current_quat

        landing_now = (
            both_contact
            & (max_rotation >= gpu.MIN_LANDING_ROTATION)
            & (max_rotation <= gpu.MAX_LANDING_ROTATION)
            & ~body_seen_before
        )
        new_landing = landing_now & ~landed
        quat = current_quat
        up_z = torch.clamp(
            1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()), -1.0, 1.0
        )
        tilt = torch.acos(up_z)
        root_vel = batch["qvel"][:, va : va + 6]
        lin, ang = root_vel[:, :3], root_vel[:, 3:]
        signed_pitch = torch.asin(
            torch.clamp(
                2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]),
                -1.0,
                1.0,
            )
        ) * (180.0 / math.pi)
        landing_pitch = torch.where(new_landing, signed_pitch, landing_pitch)
        landing_tilt = torch.where(
            new_landing, tilt * (180.0 / math.pi), landing_tilt
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
            torch.linalg.vector_norm(root_vel, dim=-1),
            landing_body_speed,
        )
        landed |= landing_now
        rolling_stable = (
            landed
            & both_contact
            & ~any_body
            & (tilt <= math.radians(15.0))
            & (lin[:, 0] >= 0.0)
            & (lin[:, 0] <= 1.50)
            & (lin[:, 1].abs() <= 0.15)
            & (lin[:, 2].abs() <= 0.15)
            & (torch.linalg.vector_norm(ang, dim=-1) <= 1.0)
        )
        stable_steps = torch.where(
            rolling_stable, stable_steps + 1, torch.zeros_like(stable_steps)
        )
        stable |= stable_steps >= int(round(0.50 / dt))
        finite &= torch.isfinite(batch["qpos"]).all(dim=-1)
        finite &= torch.isfinite(batch["qvel"]).all(dim=-1)

    first_body_time = first_body_step.clamp_max(steps).float() * dt
    drift = (batch["qpos"][:, qa + 1] - start_xy[:, 1]).abs()
    clean = ~any_body
    degrees = 180.0 / math.pi
    return {
        "minimum_rotation_deg": max_rotation * degrees,
        "minimum_clean_rotation_deg": max_clean_rotation * degrees,
        "first_body_contact_time_s": first_body_time,
        "drift_m": drift,
        "offaxis_deg": offaxis * degrees,
        "clean_rate": clean.float(),
        "landing_rate": landed.float(),
        "stable_rate": stable.float(),
        "finite_rate": finite.float(),
        "landing_signed_pitch_deg": landing_pitch,
        "landing_tilt_deg": landing_tilt,
        "landing_pitch_rate_rad_s": landing_pitch_rate,
        "landing_vertical_speed_mps": landing_vertical_speed,
        "landing_forward_speed_mps": landing_forward_speed,
        "landing_body_speed": landing_body_speed,
        "clearance_m": peak_clearance,
    }


def score_islands(metrics: dict[str, torch.Tensor], duration: float) -> dict[str, torch.Tensor]:
    rotation = metrics["minimum_clean_rotation_deg"]
    full_rotation = metrics["minimum_rotation_deg"]
    contact_fraction = torch.clamp(
        metrics["first_body_contact_time_s"] / max(duration, 1.0e-6), 0.0, 1.0
    )
    drift = metrics["drift_m"]
    offaxis = metrics["offaxis_deg"]
    clean = metrics["clean_rate"]
    landing = metrics["landing_rate"]
    stable = metrics["stable_rate"]
    finite = metrics["finite_rate"]
    line_cost = 650.0 * drift + 1.5 * offaxis
    touchdown_quality = landing * (
        340.0
        - 4.0 * torch.abs(metrics["landing_signed_pitch_deg"] + 10.0)
        - 8.0 * torch.abs(metrics["landing_pitch_rate_rad_s"])
        - 25.0 * torch.abs(metrics["landing_vertical_speed_mps"])
        - 4.0 * metrics["landing_body_speed"]
    )
    base = (
        2.0 * rotation
        + 120.0 * contact_fraction
        + 240.0 * landing
        + touchdown_quality
        + 700.0 * clean
        + 1600.0 * stable
        - line_cost
        - 2000.0 * (1.0 - finite)
    )
    return {
        "stable": base,
        "contact-free": (
            2.4 * rotation
            + 260.0 * contact_fraction
            + 1050.0 * clean
            + 300.0 * landing
            + touchdown_quality
            - line_cost
            - 2000.0 * (1.0 - finite)
        ),
        "touchdown": (
            1.4 * rotation
            - 1.2 * torch.abs(rotation - 335.0)
            + 600.0 * landing
            + 1.8 * touchdown_quality
            + 950.0 * stable
            + 300.0 * clean
            + 140.0 * contact_fraction
            - line_cost
            - 2000.0 * (1.0 - finite)
        ),
        "explore": (
            1.6 * torch.maximum(rotation, full_rotation)
            + 210.0 * contact_fraction
            + 420.0 * landing
            + 0.7 * touchdown_quality
            + 480.0 * clean
            - 0.55 * line_cost
            - 2000.0 * (1.0 - finite)
        ),
    }


def gpu_record(
    generation: int,
    candidate: int,
    nodes: torch.Tensor,
    metrics: dict[str, torch.Tensor],
) -> dict:
    record = {
        "backend": "mujoco-warp-airborne-proposal",
        "generation": generation,
        "candidate": candidate,
        "full_nodes": nodes[candidate].detach().cpu().tolist(),
        "knot_times_s": oc.KNOT_TIMES.tolist(),
        "wheel_frictionloss": 0.003,
        "current_limit_a": 1.75,
    }
    for name, values in metrics.items():
        record[name] = float(values[candidate].item())
    return record


def native_record(nodes: np.ndarray, speed: float) -> dict:
    result = oc.simulate(nodes, speed)
    result.update(
        {
            "backend": "native-mujoco-end-to-end-certification",
            "full_nodes": nodes.tolist(),
            "knot_times_s": oc.KNOT_TIMES.tolist(),
            "wheel_frictionloss": 0.003,
            "current_limit_a": 1.75,
            "completed_flip": bool(
                result["finite"]
                and result["takeoff"]
                and result["landing"]
                and result["clean_forward_rotation_deg"] >= 300.0
                and not result["body_contact"]
                and result["stable"]
            ),
        }
    )
    result["certification_score"] = float(
        result["clean_forward_rotation_deg"]
        + 200.0 * result["landing"]
        + 650.0 * (not result["body_contact"])
        + 1800.0 * result["stable"]
        + 4000.0 * result["completed_flip"]
        + 120.0 * min(result["first_body_contact_time_s"] / 1.8, 1.0)
        - 500.0 * result["drift_m"]
        - 1.5 * result["offaxis_deg"]
    )
    return result


def main() -> None:
    args = parse_args()
    if tuple(int(part) for part in package_version("mjlab").split(".")[:3]) < (
        1,
        5,
        0,
    ):
        raise SystemExit("this search requires MJLab >=1.5.0")
    if tuple(int(part) for part in mjw.__version__.split(".")[:4]) < (
        3,
        10,
        0,
        3,
    ):
        raise SystemExit("this search requires MuJoCo Warp >=3.10.0.3")
    if not math.isclose(args.wheel_friction, 0.003, abs_tol=1.0e-12):
        raise SystemExit("wheel friction is locked to exactly 0.003")
    if not math.isclose(args.current_limit_a, 1.75, abs_tol=1.0e-12):
        raise SystemExit("current limit is locked to exactly 1.75 A")
    if args.population % args.simulation_batch:
        raise SystemExit("population must be divisible by simulation batch")
    if args.population % len(ISLAND_NAMES):
        raise SystemExit("population must be divisible by four islands")
    if args.smoke:
        args.generations = min(args.generations, 2)

    reference = json.loads(args.reference.read_text())
    validate_fixed_physics(reference, args.reference)
    receipt = json.loads(args.parity_receipt.read_text())
    if not receipt.get("passed"):
        raise SystemExit("MuJoCo Warp parity receipt did not pass")
    if not math.isclose(
        float(receipt.get("handoff_time_s", -1.0)), args.handoff_time, abs_tol=1e-9
    ):
        raise SystemExit("parity receipt handoff does not match this run")
    nodes_np = np.asarray(reference["full_nodes"], dtype=np.float64)
    if nodes_np.shape != (len(oc.KNOT_TIMES), oc.SERVO_COUNT):
        raise SystemExit(f"unexpected reference shape {nodes_np.shape}")
    if not np.allclose(
        np.asarray(reference["knot_times_s"], dtype=np.float64),
        oc.KNOT_TIMES,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise SystemExit("reference knot times do not match authoritative evaluator")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    wp.set_device(args.device)
    base_nodes = torch.as_tensor(nodes_np, device=device, dtype=torch.float32)
    suffix_indices = torch.as_tensor(
        np.flatnonzero(oc.KNOT_TIMES >= args.suffix_start_time),
        device=device,
        dtype=torch.long,
    )
    suffix_count = int(suffix_indices.numel())
    if not suffix_count:
        raise SystemExit("suffix start time selects no knots")
    seed_params = suffix_compact(base_nodes, suffix_indices)
    handoff_qpos, handoff_qvel, handoff_rotation = native_handoff(
        args.scene, nodes_np, args.start_speed, args.handoff_time, args.duration
    )

    knot_times = torch.as_tensor(oc.KNOT_TIMES, device=device, dtype=torch.float32)

    island_size = args.population // len(ISLAND_NAMES)
    param_dim = suffix_count * COMPACT_DOF
    means = seed_params.repeat(len(ISLAND_NAMES), 1)
    time_scale = torch.linspace(0.035, 0.24, suffix_count, device=device)
    dof_scale = torch.tensor(
        [0.70, 0.70, 1.0, 1.0, 1.0, 0.90, 0.90], device=device
    )
    base_std = (time_scale[:, None] * dof_scale[None, :]).reshape(-1)
    std_multipliers = torch.tensor([0.65, 0.85, 1.0, 1.35], device=device)
    stds = std_multipliers[:, None] * base_std[None, :]
    minimum_std = 0.012 * torch.ones(param_dim, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    best_native: dict | None = None

    start_event = {
        "event": "start",
        "backend": "mujoco-warp-airborne-proposal/native-end-to-end-certification",
        "gpu": torch.cuda.get_device_name(device),
        "versions": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": mjw.__version__,
            "warp": wp.__version__,
            "torch": torch.__version__,
        },
        "population": args.population,
        "simulation_batch": args.simulation_batch,
        "generations": args.generations,
        "islands": list(ISLAND_NAMES),
        "parameter_count": param_dim,
        "handoff_time_s": args.handoff_time,
        "handoff_clean_rotation_deg": math.degrees(handoff_rotation),
        "suffix_start_time_s": args.suffix_start_time,
        "start_speed_mps": args.start_speed,
        "wheel_frictionloss": 0.003,
        "current_limit_a": 1.75,
        "parity_receipt": str(args.parity_receipt.resolve()),
    }
    print(json.dumps(start_event, sort_keys=True), flush=True)
    atomic_json(output_dir / "run-metadata.json", start_event)

    try:
        for generation in range(args.generations):
            started = time.monotonic()
            candidate_chunks: list[torch.Tensor] = []
            for island in range(len(ISLAND_NAMES)):
                noise = torch.randn(
                    island_size,
                    param_dim,
                    device=device,
                    generator=generator,
                )
                samples = means[island].unsqueeze(0) + stds[island].unsqueeze(0) * noise
                candidate_chunks.append(samples)
            params = clamp_suffix(torch.cat(candidate_chunks, dim=0), suffix_count)
            # Keep the parity-certified donor immutable in slot 0.  Slot 1
            # carries the evolving stable-island mean; the remaining island
            # anchors keep their first slot as before.
            params[0] = seed_params
            if island_size > 1:
                params[1] = means[0]
            for island in range(1, len(ISLAND_NAMES)):
                params[island * island_size] = means[island]
            nodes = expand_suffix(params, base_nodes, suffix_indices)

            metric_chunks: dict[str, list[torch.Tensor]] = {}
            for begin in range(0, args.population, args.simulation_batch):
                # put_data copies the complete native prefix state, including
                # solver warm-start and actuator state.  Rebuilding each batch
                # is the only reset path that is bit-for-bit aligned with the
                # independently validated parity harness.
                direct_batch = build_direct_warp_batch(args, nodes_np)
                batch = evaluate_direct_warp(
                    direct_batch,
                    nodes[begin : begin + args.simulation_batch],
                    knot_times,
                    args.handoff_time,
                    args.duration,
                    0.0,
                )
                for name, values in batch.items():
                    metric_chunks.setdefault(name, []).append(values)
            metrics = {
                name: torch.cat(values, dim=0)
                for name, values in metric_chunks.items()
            }
            scores = score_islands(metrics, args.duration - args.handoff_time)

            elite_count = min(
                island_size,
                max(8, int(round(island_size * args.elite_fraction))),
            )
            for island, name in enumerate(ISLAND_NAMES):
                begin = island * island_size
                end = begin + island_size
                local = torch.topk(scores[name][begin:end], elite_count).indices + begin
                elite = params[local]
                elite_mean = elite.mean(dim=0)
                elite_std = elite.std(dim=0, unbiased=False).clamp_min(minimum_std)
                means[island] = (
                    (1.0 - args.update_rate) * means[island]
                    + args.update_rate * elite_mean
                )
                stds[island] = (
                    (1.0 - args.update_rate) * stds[island]
                    + args.update_rate * elite_std
                ).clamp_min(minimum_std)
                means[island] = clamp_suffix(
                    means[island].unsqueeze(0), suffix_count
                )[0]

            combined = torch.stack(
                [
                    (value - value.mean()) / value.std().clamp_min(1.0e-6)
                    for value in scores.values()
                ],
                dim=0,
            ).amax(dim=0)
            native_pool = set()
            per_objective = max(2, args.native_candidates // len(ISLAND_NAMES))
            for value in scores.values():
                native_pool.update(torch.topk(value, per_objective).indices.tolist())
            native_pool.update(island * island_size for island in range(len(ISLAND_NAMES)))
            native_pool.update(
                torch.topk(
                    metrics["landing_rate"] * 1000.0
                    - torch.abs(metrics["landing_signed_pitch_deg"] + 10.0)
                    - 3.0 * torch.abs(metrics["landing_pitch_rate_rad_s"])
                    - 10.0 * torch.abs(metrics["landing_vertical_speed_mps"]),
                    min(per_objective, args.population),
                ).indices.tolist()
            )
            native_pool.update(
                torch.topk(combined, args.native_candidates).indices.tolist()
            )
            required_anchor_ids = [0]
            if island_size > 1:
                required_anchor_ids.append(1)
            required_anchor_ids.extend(
                island * island_size for island in range(1, len(ISLAND_NAMES))
            )
            ranked_ids = sorted(
                native_pool,
                key=lambda candidate_id: float(combined[candidate_id].item()),
                reverse=True,
            )
            candidate_ids = required_anchor_ids + [
                candidate_id
                for candidate_id in ranked_ids
                if candidate_id not in required_anchor_ids
            ]
            candidate_ids = candidate_ids[: max(args.native_candidates, 4)]
            oc.make_context(
                str(args.scene.resolve()), nodes_np[0].tolist(), args.duration
            )
            certified: list[dict] = []
            for candidate_id in candidate_ids:
                item = native_record(
                    nodes[candidate_id].detach().cpu().numpy().astype(np.float64),
                    args.start_speed,
                )
                item["generation"] = generation
                item["candidate"] = candidate_id
                certified.append(item)
                if (
                    best_native is None
                    or item["certification_score"] > best_native["certification_score"]
                ):
                    best_native = item
                    atomic_json(output_dir / "best-native-certified.json", best_native)
            certified.sort(key=lambda item: item["certification_score"], reverse=True)
            atomic_json(
                output_dir / f"native-screen-generation-{generation:04d}.json",
                certified,
            )

            best_gpu_id = int(scores["stable"].argmax().item())
            best_gpu = gpu_record(generation, best_gpu_id, nodes, metrics)
            atomic_json(output_dir / "best-gpu-proposal.json", best_gpu)
            elapsed = time.monotonic() - started
            event = {
                "event": "generation",
                "generation": generation,
                "seconds": elapsed,
                "gpu_best_clean_rotation_deg": best_gpu["minimum_clean_rotation_deg"],
                "gpu_best_contact_time_s": best_gpu["first_body_contact_time_s"],
                "gpu_best_landing_rate": best_gpu["landing_rate"],
                "gpu_best_clean_rate": best_gpu["clean_rate"],
                "gpu_best_stable_rate": best_gpu["stable_rate"],
                "gpu_best_landing_signed_pitch_deg": best_gpu[
                    "landing_signed_pitch_deg"
                ],
                "gpu_best_landing_pitch_rate_rad_s": best_gpu[
                    "landing_pitch_rate_rad_s"
                ],
                "gpu_seed_clean_rotation_deg": float(
                    metrics["minimum_clean_rotation_deg"][0].item()
                ),
                "gpu_seed_landing_rate": float(metrics["landing_rate"][0].item()),
                "gpu_seed_body_clean_rate": float(metrics["clean_rate"][0].item()),
                "native_best_clean_rotation_deg": best_native[
                    "clean_forward_rotation_deg"
                ],
                "native_best_landing": best_native["landing"],
                "native_best_body_contact": best_native["body_contact"],
                "native_best_stable": best_native["stable"],
                "native_best_first_body_contact_time_s": best_native[
                    "first_body_contact_time_s"
                ],
                "native_completed_flip": best_native["completed_flip"],
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            atomic_json(output_dir / "latest-generation.json", event)
            torch.save(
                {
                    "generation": generation,
                    "means": means.detach().cpu(),
                    "stds": stds.detach().cpu(),
                    "generator_state": generator.get_state(),
                    "best_native": best_native,
                },
                output_dir / "optimizer-private.pt",
            )
            if best_native["completed_flip"]:
                atomic_json(output_dir / "completed-frontflip.json", best_native)
                print(
                    json.dumps(
                        {
                            "event": "completed-frontflip",
                            "generation": generation,
                            "clean_rotation_deg": best_native[
                                "clean_forward_rotation_deg"
                            ],
                            "landing_time_s": best_native["landing_time_s"],
                            "drift_m": best_native["drift_m"],
                            "offaxis_deg": best_native["offaxis_deg"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
    finally:
        # MuJoCo Warp allocations are released when the process exits.  Keeping
        # a single batch alive avoids allocator churn between generations.
        pass


if __name__ == "__main__":
    main()
