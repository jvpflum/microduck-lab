#!/usr/bin/env python3
"""Compare native MuJoCo and MuJoCo Warp on one exact front-flip replay.

This is an event-level parity gate, not a claim that two contact solvers should
remain bit-identical after impact.  It requires the same pre-contact motion,
takeoff, clean rotation, tire touchdown, and first forbidden contact within
tight tolerances before MuJoCo Warp may become the search backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import search_frontflip_native_oc as oc


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def quat_delta_vector(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous = previous / max(float(np.linalg.norm(previous)), 1.0e-12)
    current = current / max(float(np.linalg.norm(current)), 1.0e-12)
    pw, px, py, pz = previous
    cw, cx, cy, cz = current
    delta = np.asarray(
        [
            pw * cw + px * cx + py * cy + pz * cz,
            pw * cx - px * cw - py * cz + pz * cy,
            pw * cy + px * cz - py * cw - pz * cx,
            pw * cz - px * cy + py * cx - pz * cw,
        ],
        dtype=np.float64,
    )
    delta /= max(float(np.linalg.norm(delta)), 1.0e-12)
    if delta[0] < 0.0:
        delta *= -1.0
    vector = delta[1:]
    length = float(np.linalg.norm(vector))
    angle = 2.0 * math.atan2(length, max(float(delta[0]), 0.0))
    return vector / max(length, 1.0e-12) * angle


def reset_data(context: dict, data: mujoco.MjData, start_speed: float) -> None:
    model: mujoco.MjModel = context["model"]
    qa, va = context["qpos_adr"], context["qvel_adr"]
    mujoco.mj_resetData(model, data)
    data.qpos[qa : qa + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    data.qvel[va] = start_speed
    for wheel_dof in context["wheel_dofs"]:
        data.qvel[wheel_dof] = start_speed / oc.WHEEL_RADIUS
    data.qpos[context["actuator_qpos"]] = context["default"]
    data.ctrl[:] = context["default"]
    mujoco.mj_forward(model, data)


def scan(context: dict, data: mujoco.MjData) -> tuple[set[str], bool, list[str]]:
    view = dict(context)
    view["data"] = data
    return oc.scan_contacts(view)


def scan_geom_pairs(
    context: dict, geom_pairs: np.ndarray
) -> tuple[set[str], bool, list[str]]:
    """Classify direct MuJoCo Warp contact pairs without host MjData copies."""
    model: mujoco.MjModel = context["model"]
    sides: set[str] = set()
    body_hit = False
    names: list[str] = []
    for pair in np.asarray(geom_pairs, dtype=np.int32).reshape(-1, 2):
        geom1, geom2 = int(pair[0]), int(pair[1])
        if not (0 <= geom1 < model.ngeom and 0 <= geom2 < model.ngeom):
            raise RuntimeError(f"MuJoCo Warp returned invalid geom pair {pair.tolist()}")
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


def replay(
    context: dict,
    nodes: np.ndarray,
    start_speed: float,
    duration: float,
    backend: str,
    warp_model: mjw.Model | None = None,
    handoff_time: float = 0.0,
) -> dict:
    model: mujoco.MjModel = context["model"]
    qa, va = context["qpos_adr"], context["qvel_adr"]
    data = mujoco.MjData(model)
    reset_data(context, data, start_speed)
    control_stride = int(round(oc.CONTROL_DT / oc.PHYSICS_DT))
    prefix_steps = int(round(handoff_time / oc.PHYSICS_DT))
    target = nodes[0].copy()
    for prefix_step in range(prefix_steps):
        time_s = prefix_step * oc.PHYSICS_DT
        if prefix_step % control_stride == 0:
            target = oc.target_at(time_s, nodes)
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("native handoff prefix became non-finite")

    warp_data = None
    if backend == "mujoco-warp":
        if warp_model is None:
            raise ValueError("warp_model is required")
        # The default allocation is derived from the reset pose and is too
        # small for the constraint burst at skate touchdown.  Under-allocation
        # makes the copied contact table invalid, so parity must use explicit
        # conservative capacities rather than silently dropping constraints.
        warp_data = mjw.put_data(
            model,
            data,
            nworld=1,
            nconmax=256,
            nccdmax=256,
            njmax=512,
            njmax_nnz=8192,
            naconmax=256,
            naccdmax=256,
            nvmax=int(model.nv),
        )

    start_z = float(data.qpos[qa + 2])
    previous_quat = data.qpos[qa + 3 : qa + 7].copy()
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    initial_sides, _, _ = scan(context, data)
    support_seen = len(initial_sides) == 2 or handoff_time > 0.0
    takeoff = handoff_time > 0.0
    landed = False
    body_contact = False
    takeoff_time = handoff_time if takeoff else None
    landing_time = None
    first_body_contact_time = None
    first_body_contact_geoms: list[str] = []
    clean_rotation = 0.0
    total_rotation = 0.0
    max_clean_rotation = 0.0
    max_total_rotation = 0.0
    peak_clearance = 0.0
    precontact_qpos: list[list[float]] = []
    precontact_qvel: list[list[float]] = []
    steps = int(round((duration - handoff_time) / oc.PHYSICS_DT))

    for step in range(steps):
        absolute_step = prefix_steps + step
        time_s = absolute_step * oc.PHYSICS_DT
        if absolute_step % control_stride == 0:
            target = oc.target_at(time_s, nodes)
        if backend == "native-mujoco":
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            sides, body_now, body_names = scan(context, data)
        else:
            assert warp_data is not None and warp_model is not None
            warp_data.ctrl.assign(np.asarray(target, dtype=np.float32)[None, :])
            mjw.step(warp_model, warp_data)
            wp.synchronize_device("cuda:0")
            # mujoco_warp.get_data_into() does not reliably copy explicitly
            # over-allocated contact buffers in 3.10.0.2.  Read the official
            # device arrays directly: nacon is the valid point-contact count
            # and active pairs occupy contact.geom[:nacon].
            qpos = np.asarray(warp_data.qpos.numpy()[0], dtype=np.float64)
            qvel = np.asarray(warp_data.qvel.numpy()[0], dtype=np.float64)
            nacon = int(np.asarray(warp_data.nacon.numpy()).reshape(-1)[0])
            overflow = int(np.asarray(warp_data.overflow.numpy()).reshape(-1)[0])
            if overflow:
                raise RuntimeError(f"MuJoCo Warp buffer overflow bitmask={overflow}")
            geom_pairs = np.asarray(warp_data.contact.geom.numpy())[:nacon]
            sides, body_now, body_names = scan_geom_pairs(context, geom_pairs)

        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            break
        both_grounded = len(sides) == 2
        both_airborne = len(sides) == 0
        support_seen |= both_grounded
        if body_now and first_body_contact_time is None:
            first_body_contact_time = (absolute_step + 1) * oc.PHYSICS_DT
            first_body_contact_geoms = body_names
        if first_body_contact_time is None:
            precontact_qpos.append(qpos.tolist())
            precontact_qvel.append(qvel.tolist())
        body_contact |= body_now

        vz = float(qvel[va + 2])
        if not takeoff and support_seen and both_airborne and vz > 0.02 and not body_now:
            takeoff = True
            takeoff_time = (absolute_step + 1) * oc.PHYSICS_DT

        current_quat = qpos[qa + 3 : qa + 7].copy()
        if takeoff and not landed and both_airborne:
            delta = quat_delta_vector(previous_quat, current_quat)
            positive_pitch = max(0.0, float(delta[1]))
            total_rotation += positive_pitch
            max_total_rotation = max(max_total_rotation, total_rotation)
            if not body_contact:
                clean_rotation += positive_pitch
                max_clean_rotation = max(max_clean_rotation, clean_rotation)
            peak_clearance = max(peak_clearance, float(qpos[qa + 2]) - start_z)
        previous_quat = current_quat

        if (
            takeoff
            and not landed
            and both_grounded
            and not body_contact
            and math.radians(oc.MIN_FLIP_DEG) <= max_total_rotation
            <= math.radians(oc.MAX_FLIP_DEG)
        ):
            landed = True
            landing_time = (absolute_step + 1) * oc.PHYSICS_DT

    return {
        "backend": backend,
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        "takeoff": takeoff,
        "takeoff_time_s": takeoff_time if takeoff_time is not None else duration,
        "landing": landed,
        "landing_time_s": landing_time if landing_time is not None else duration,
        "body_contact": body_contact,
        "first_body_contact_time_s": (
            first_body_contact_time if first_body_contact_time is not None else duration
        ),
        "first_body_contact_geoms": first_body_contact_geoms,
        "clean_rotation_deg": math.degrees(max_clean_rotation),
        "full_rotation_deg": math.degrees(max_total_rotation),
        "peak_clearance_m": max(0.0, peak_clearance),
        "precontact_qpos": precontact_qpos,
        "precontact_qvel": precontact_qvel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.80)
    parser.add_argument("--start-speed", type=float, default=1.54)
    parser.add_argument(
        "--handoff-time",
        type=float,
        default=0.0,
        help=(
            "Run an exact native prefix to this time, then compare the suffix. "
            "Use an airborne handoff to gate Warp flight/landing proposals when "
            "the source wheel mesh is not contact-parity compatible."
        ),
    )
    parser.add_argument(
        "--disable-multiccd",
        action="store_true",
        help=(
            "Disable MultiCCD in both pristine models. This is useful when the "
            "CPU path already emits one plane/mesh point but MuJoCo Warp emits "
            "a four-point manifold for the same wheel pair."
        ),
    )
    args = parser.parse_args()
    if args.handoff_time < 0.0 or args.handoff_time >= args.duration:
        raise SystemExit("handoff time must be in [0, duration)")

    reference = json.loads(args.reference.read_text())
    if not math.isclose(float(reference["wheel_frictionloss"]), 0.003, abs_tol=1e-12):
        raise SystemExit("reference must use exact wheel frictionloss 0.003")
    if not math.isclose(float(reference["current_limit_a"]), 1.75, abs_tol=1e-12):
        raise SystemExit("reference must use exact current limit 1.75 A")
    knot_times = np.asarray(reference["knot_times_s"], dtype=np.float64)
    if not np.allclose(knot_times, oc.KNOT_TIMES, atol=1e-12, rtol=0.0):
        raise SystemExit("reference knot times do not match the authoritative evaluator")
    nodes = np.asarray(reference["full_nodes"], dtype=np.float64)
    if nodes.shape != (len(oc.KNOT_TIMES), oc.SERVO_COUNT):
        raise SystemExit(f"unexpected full_nodes shape {nodes.shape}")

    wp.set_device("cuda:0")
    # Keep independent host models.  mujoco_warp.put_model() prepares model
    # state for the device backend; the native reference must be evaluated on
    # a pristine MjModel before that conversion occurs.
    oc.make_context(str(args.scene.resolve()), nodes[0].tolist(), args.duration)
    native_context = oc._TLS.context
    if args.disable_multiccd:
        native_context["model"].opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_MULTICCD
        )
    native = replay(
        native_context,
        nodes,
        args.start_speed,
        args.duration,
        "native-mujoco",
        handoff_time=args.handoff_time,
    )

    oc.make_context(str(args.scene.resolve()), nodes[0].tolist(), args.duration)
    gpu_context = oc._TLS.context
    gpu_model: mujoco.MjModel = gpu_context["model"]
    if args.disable_multiccd:
        gpu_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    warp_model = mjw.put_model(gpu_model)
    gpu = replay(
        gpu_context,
        nodes,
        args.start_speed,
        args.duration,
        "mujoco-warp",
        warp_model,
        handoff_time=args.handoff_time,
    )

    shared = min(len(native["precontact_qpos"]), len(gpu["precontact_qpos"]))
    if shared:
        native_qpos = np.asarray(native["precontact_qpos"][:shared])
        gpu_qpos = np.asarray(gpu["precontact_qpos"][:shared])
        native_qvel = np.asarray(native["precontact_qvel"][:shared])
        gpu_qvel = np.asarray(gpu["precontact_qvel"][:shared])
        qpos_error = float(np.max(np.abs(native_qpos - gpu_qpos)))
        qvel_error = float(np.max(np.abs(native_qvel - gpu_qvel)))
    else:
        qpos_error = float("inf")
        qvel_error = float("inf")

    deltas = {
        "takeoff_time_s": abs(native["takeoff_time_s"] - gpu["takeoff_time_s"]),
        "first_body_contact_time_s": abs(
            native["first_body_contact_time_s"] - gpu["first_body_contact_time_s"]
        ),
        "clean_rotation_deg": abs(
            native["clean_rotation_deg"] - gpu["clean_rotation_deg"]
        ),
        "peak_clearance_m": abs(native["peak_clearance_m"] - gpu["peak_clearance_m"]),
        "precontact_qpos_max_abs": qpos_error,
        "precontact_qvel_max_abs": qvel_error,
    }
    passed = bool(
        native["finite"]
        and gpu["finite"]
        and native["takeoff"] == gpu["takeoff"]
        and native["landing"] == gpu["landing"]
        and native["body_contact"] == gpu["body_contact"]
        and native["first_body_contact_geoms"] == gpu["first_body_contact_geoms"]
        and deltas["takeoff_time_s"] <= 0.015
        and deltas["first_body_contact_time_s"] <= 0.020
        and deltas["clean_rotation_deg"] <= 5.0
        and deltas["peak_clearance_m"] <= 0.010
        and qpos_error <= 0.050
        and qvel_error <= 5.0
    )
    result = {
        "passed": passed,
        "standard_candidate": "mujoco-warp" if passed else "native-mujoco",
        "versions": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": mjw.__version__,
            "warp": wp.__version__,
            "device": str(wp.get_device()),
        },
        "physics": {"wheel_frictionloss": 0.003, "current_limit_a": 1.75},
        "collision_options": {"multiccd_disabled": args.disable_multiccd},
        "handoff_time_s": args.handoff_time,
        "native": {k: v for k, v in native.items() if not k.startswith("precontact_")},
        "gpu": {k: v for k, v in gpu.items() if not k.startswith("precontact_")},
        "absolute_deltas": deltas,
        "tolerances": {
            "takeoff_time_s": 0.015,
            "first_body_contact_time_s": 0.020,
            "clean_rotation_deg": 5.0,
            "peak_clearance_m": 0.010,
            "precontact_qpos_max_abs": 0.050,
            "precontact_qvel_max_abs": 5.0,
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
