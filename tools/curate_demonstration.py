#!/usr/bin/env python3
"""Validate and normalize a browser-arena motion demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    w, x, y, z = a
    W, X, Y, Z = b
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )


def normalized(values: list[float]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


def upright_z(quaternion: tuple[float, ...]) -> float:
    _, x, y, _ = quaternion
    return -(1.0 - 2.0 * (x * x + y * y))


def curate(source: Path, destination: Path) -> dict:
    raw_bytes = source.read_bytes()
    document = json.loads(raw_bytes)
    frames = document["frames"]
    hz = int(document.get("control_hz", 50))
    resets = [index for index in range(1, len(frames)) if frames[index]["t"] < frames[index - 1]["t"]]
    start = resets[-1] if resets else 0
    frames = frames[start:]

    quaternions: list[tuple[float, ...]] = []
    for frame in frames:
        quaternion = normalized(frame["qpos"][3:7])
        if quaternions and sum(a * b for a, b in zip(quaternion, quaternions[-1])) < 0:
            quaternion = tuple(-value for value in quaternion)
        quaternions.append(quaternion)

    rotation = [0.0, 0.0, 0.0]
    for previous, current in zip(quaternions, quaternions[1:]):
        delta = normalized(list(quat_mul((previous[0], -previous[1], -previous[2], -previous[3]), current)))
        if delta[0] < 0:
            delta = tuple(-value for value in delta)
        vector_length = math.sqrt(sum(value * value for value in delta[1:]))
        angle = 2.0 * math.atan2(vector_length, max(0.0, delta[0]))
        if vector_length > 1e-9:
            for axis in range(3):
                rotation[axis] += delta[axis + 1] / vector_length * angle

    heights = [frame["qpos"][2] for frame in frames]
    upright = [upright_z(quaternion) for quaternion in quaternions]
    apex_index = max(range(len(heights)), key=heights.__getitem__)
    baseline_height = statistics.median(heights[: min(25, len(heights))])
    tail = frames[-min(25, len(frames)) :]
    tail_upright = upright[-len(tail) :]
    mean_tail_angular_speed = statistics.mean(
        math.sqrt(sum(value * value for value in frame["qvel"][3:6])) for frame in tail
    )
    dx = frames[-1]["qpos"][0] - frames[0]["qpos"][0]
    dy = frames[-1]["qpos"][1] - frames[0]["qpos"][1]
    displacement = math.hypot(dx, dy)
    primary_axis = max(range(3), key=lambda axis: abs(rotation[axis]))
    turns = rotation[primary_axis] / (2.0 * math.pi)
    resets_after_apex = any(index - start > apex_index for index in resets)

    metrics = {
        "primary_rotation_axis": "xyz"[primary_axis],
        "primary_rotation_turns": round(turns, 4),
        "apex_height_m": round(heights[apex_index], 4),
        "apex_lift_m": round(heights[apex_index] - baseline_height, 4),
        "max_inversion": round(max(upright), 4),
        "landing_upright_mean": round(statistics.mean(tail_upright), 4),
        "landing_angular_speed_rad_s": round(mean_tail_angular_speed, 4),
        "horizontal_displacement_m": round(displacement, 4),
        "reset_after_apex": resets_after_apex,
    }
    accepted = (
        0.8 <= abs(turns) <= 1.2
        and metrics["apex_lift_m"] >= 0.08
        and metrics["max_inversion"] >= 0.8
        and metrics["landing_upright_mean"] <= -0.9
        and metrics["landing_angular_speed_rad_s"] <= 0.3
        and not resets_after_apex
    )
    if not accepted:
        raise ValueError(f"Demonstration failed acceptance gates: {metrics}")

    for index, frame in enumerate(frames):
        frame["sim_time"] = frame["t"]
        frame["t"] = index / hz
    document["frames"] = frames
    document["curation"] = {
        "accepted": True,
        "motion_class": "rolling-backflip" if displacement >= 0.25 else "stationary-backflip",
        "source": str(source),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_start_frame": start,
        "metrics": metrics,
        "usage": "State-reference trajectory; recorded policy actions are not behavior-cloning labels because the operator supplied external perturbation force.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, separators=(",", ":")))
    return document["curation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(curate(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
