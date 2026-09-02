#!/usr/bin/env python3
"""Fuse specialist and incumbent actions with per-joint authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper


OBSERVATION_DIM = 61
ACTION_DIM = 14
COMMAND_X_INDEX = 48
COMMAND_YAW_INDEX = 50
STEERING_INDICES = (0, 1, 9, 10)
PROPULSION_INDICES = (2, 3, 4, 11, 12, 13)
HEAD_INDICES = (5, 6, 7, 8)


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def _validate_actor(model: onnx.ModelProto, label: str) -> None:
    if len(model.graph.input) != 1 or _shape(model.graph.input[0]) != (1, OBSERVATION_DIM):
        raise ValueError(f"{label} must accept one [1, {OBSERVATION_DIM}] observation")
    if len(model.graph.output) != 1 or _shape(model.graph.output[0]) != (1, ACTION_DIM):
        raise ValueError(f"{label} must produce one [1, {ACTION_DIM}] action")


def build_joint_fusion(
    incumbent_path: Path,
    specialist_path: Path,
    output_path: Path,
    *,
    steering_authority: float,
    propulsion_authority: float,
    head_authority: float,
    hip_yaw_authority: float | None = None,
    hip_roll_authority: float | None = None,
    hip_pitch_authority: float | None = None,
    knee_authority: float | None = None,
    ankle_authority: float | None = None,
    speed_command_threshold: float = 0.5,
    smooth_turn_start: float = 0.08,
    smooth_turn_end: float = 0.25,
) -> onnx.ModelProto:
    if not 0.0 <= smooth_turn_start < smooth_turn_end:
        raise ValueError("smooth turn band must satisfy 0 <= start < end")
    explicit_authorities = (
        hip_yaw_authority,
        hip_roll_authority,
        hip_pitch_authority,
        knee_authority,
        ankle_authority,
    )
    if min(steering_authority, propulsion_authority, head_authority) < 0.0 or any(
        value is not None and value < 0.0 for value in explicit_authorities
    ):
        raise ValueError("joint authorities must be non-negative")

    incumbent = onnx.load(incumbent_path)
    specialist = onnx.load(specialist_path)
    _validate_actor(incumbent, "incumbent policy")
    _validate_actor(specialist, "specialist policy")
    if incumbent.opset_import[0].version != specialist.opset_import[0].version:
        raise ValueError("Policies use different ONNX opset versions")

    incumbent_input = incumbent.graph.input[0].name
    incumbent_output = incumbent.graph.output[0].name
    specialist_input = specialist.graph.input[0].name
    specialist_output = specialist.graph.output[0].name
    incumbent = compose.add_prefix(incumbent, "incumbent_")
    specialist = compose.add_prefix(specialist, "specialist_")
    incumbent_input = f"incumbent_{incumbent_input}"
    incumbent_output = f"incumbent_{incumbent_output}"
    specialist_input = f"specialist_{specialist_input}"
    specialist_output = f"specialist_{specialist_output}"

    nodes = list(incumbent.graph.node) + list(specialist.graph.node)
    for node in nodes:
        node.input[:] = [
            "obs" if name in {incumbent_input, specialist_input} else name
            for name in node.input
        ]
    nodes.extend(
        [
            helper.make_node("Gather", ["obs", "fusion_command_x_index"], ["fusion_command_x"], axis=1),
            helper.make_node("Greater", ["fusion_command_x", "fusion_speed_threshold"], ["fusion_is_speed"]),
            helper.make_node("Cast", ["fusion_is_speed"], ["fusion_speed_gate"], to=TensorProto.FLOAT),
            helper.make_node("Gather", ["obs", "fusion_command_yaw_index"], ["fusion_command_yaw"], axis=1),
            helper.make_node("Abs", ["fusion_command_yaw"], ["fusion_abs_command_yaw"]),
            helper.make_node("Sub", ["fusion_turn_end", "fusion_abs_command_yaw"], ["fusion_turn_headroom"]),
            helper.make_node("Div", ["fusion_turn_headroom", "fusion_turn_span"], ["fusion_turn_gate_raw"]),
            helper.make_node("Clip", ["fusion_turn_gate_raw", "fusion_zero", "fusion_one"], ["fusion_turn_gate"]),
            helper.make_node("Mul", ["fusion_speed_gate", "fusion_turn_gate"], ["fusion_route_gate"]),
            helper.make_node("Sub", [specialist_output, incumbent_output], ["fusion_specialist_delta"]),
            helper.make_node("Mul", ["fusion_specialist_delta", "fusion_joint_authority"], ["fusion_joint_delta"]),
            helper.make_node("Mul", ["fusion_joint_delta", "fusion_route_gate"], ["fusion_gated_delta"]),
            helper.make_node("Add", [incumbent_output, "fusion_gated_delta"], ["actions"]),
        ]
    )

    authority = np.zeros((1, ACTION_DIM), dtype=np.float32)
    authority[:, STEERING_INDICES] = steering_authority
    authority[:, PROPULSION_INDICES] = propulsion_authority
    authority[:, HEAD_INDICES] = head_authority
    for indices, value in (
        ((0, 9), hip_yaw_authority),
        ((1, 10), hip_roll_authority),
        ((2, 11), hip_pitch_authority),
        ((3, 12), knee_authority),
        ((4, 13), ankle_authority),
    ):
        if value is not None:
            authority[:, indices] = value
    initializers = list(incumbent.graph.initializer) + list(specialist.graph.initializer)
    initializers.extend(
        [
            numpy_helper.from_array(np.asarray([COMMAND_X_INDEX], dtype=np.int64), "fusion_command_x_index"),
            numpy_helper.from_array(np.asarray([COMMAND_YAW_INDEX], dtype=np.int64), "fusion_command_yaw_index"),
            numpy_helper.from_array(np.asarray([speed_command_threshold], dtype=np.float32), "fusion_speed_threshold"),
            numpy_helper.from_array(np.asarray([smooth_turn_end], dtype=np.float32), "fusion_turn_end"),
            numpy_helper.from_array(np.asarray([smooth_turn_end - smooth_turn_start], dtype=np.float32), "fusion_turn_span"),
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "fusion_zero"),
            numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "fusion_one"),
            numpy_helper.from_array(authority, "fusion_joint_authority"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "microduck_joint_specialist_fusion",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBSERVATION_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
        initializer=initializers,
        value_info=list(incumbent.graph.value_info) + list(specialist.graph.value_info),
    )
    model = helper.make_model(
        graph,
        producer_name="ducklab-joint-fusion",
        opset_imports=list(incumbent.opset_import),
        ir_version=max(incumbent.ir_version, specialist.ir_version),
    )
    helper.set_model_props(
        model,
        {
            **{prop.key: prop.value for prop in incumbent.metadata_props},
            "joint_fusion_incumbent": str(incumbent_path.resolve()),
            "joint_fusion_specialist": str(specialist_path.resolve()),
            "joint_fusion_authority": (
                f"steering={steering_authority:g},propulsion={propulsion_authority:g},"
                f"head={head_authority:g};vector="
                + ",".join(f"{value:g}" for value in authority[0])
            ),
            "joint_fusion_route": (
                f"command_x>{speed_command_threshold:g}; yaw taper "
                f"{smooth_turn_start:g}..{smooth_turn_end:g} rad/s"
            ),
        },
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("specialist", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steering-authority", type=float, required=True)
    parser.add_argument("--propulsion-authority", type=float, required=True)
    parser.add_argument("--head-authority", type=float, default=0.0)
    parser.add_argument("--hip-yaw-authority", type=float)
    parser.add_argument("--hip-roll-authority", type=float)
    parser.add_argument("--hip-pitch-authority", type=float)
    parser.add_argument("--knee-authority", type=float)
    parser.add_argument("--ankle-authority", type=float)
    parser.add_argument("--speed-command-threshold", type=float, default=0.5)
    parser.add_argument("--smooth-turn-start", type=float, default=0.08)
    parser.add_argument("--smooth-turn-end", type=float, default=0.25)
    args = parser.parse_args()
    build_joint_fusion(
        args.incumbent,
        args.specialist,
        args.output,
        steering_authority=args.steering_authority,
        propulsion_authority=args.propulsion_authority,
        head_authority=args.head_authority,
        hip_yaw_authority=args.hip_yaw_authority,
        hip_roll_authority=args.hip_roll_authority,
        hip_pitch_authority=args.hip_pitch_authority,
        knee_authority=args.knee_authority,
        ankle_authority=args.ankle_authority,
        speed_command_threshold=args.speed_command_threshold,
        smooth_turn_start=args.smooth_turn_start,
        smooth_turn_end=args.smooth_turn_end,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
