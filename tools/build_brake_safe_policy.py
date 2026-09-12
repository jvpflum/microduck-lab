#!/usr/bin/env python3
"""Route zero-command idle/braking to a safe transition policy.

The input drive policy can itself be a command-routed V66-style composition.
This outer gate uses V65 only at effectively zero forward command, where its
native low/mid/high stack provides the stable deceleration transition that the
V66 high-to-V11 hard switch lacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper


OBSERVATION_DIM = 61
ACTION_DIM = 14
COMMAND_X_INDEX = 48
JOINT_VELOCITY_START = 20
JOINT_VELOCITY_END = 34
LAST_ACTION_START = 34
LAST_ACTION_END = 48
BASE_ANGULAR_VELOCITY_START = 0
BASE_ANGULAR_VELOCITY_END = 3


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def _validate_actor(model: onnx.ModelProto, label: str) -> None:
    if len(model.graph.input) != 1 or _shape(model.graph.input[0]) != (1, OBSERVATION_DIM):
        raise ValueError(f"{label} must accept one [1, {OBSERVATION_DIM}] observation")
    if len(model.graph.output) != 1 or _shape(model.graph.output[0]) != (1, ACTION_DIM):
        raise ValueError(f"{label} must produce one [1, {ACTION_DIM}] action")


def build_brake_safe_policy(
    drive_path: Path,
    brake_path: Path,
    output_path: Path,
    *,
    zero_command_threshold: float = 0.02,
    joint_velocity_threshold: float = 0.25,
    gate_mode: str = "joint_velocity",
    high_action_threshold: float = 0.42,
    low_action_threshold: float = 0.10,
    angular_velocity_threshold: float = 0.08,
) -> onnx.ModelProto:
    if zero_command_threshold <= 0.0:
        raise ValueError("zero command threshold must be positive")
    if joint_velocity_threshold < 0.0:
        raise ValueError("joint velocity threshold must be non-negative")
    if gate_mode not in {"joint_velocity", "brake_signature"}:
        raise ValueError("gate mode must be joint_velocity or brake_signature")
    if not 0.0 <= low_action_threshold < high_action_threshold:
        raise ValueError("action thresholds must satisfy 0 <= low < high")

    drive = onnx.load(drive_path)
    brake = onnx.load(brake_path)
    _validate_actor(drive, "drive policy")
    _validate_actor(brake, "brake policy")
    if drive.opset_import[0].version != brake.opset_import[0].version:
        raise ValueError("Policies use different ONNX opset versions")

    drive_input = drive.graph.input[0].name
    drive_output = drive.graph.output[0].name
    brake_input = brake.graph.input[0].name
    brake_output = brake.graph.output[0].name
    drive = compose.add_prefix(drive, "drive_")
    brake = compose.add_prefix(brake, "brake_")
    drive_prefixed_input = f"drive_{drive_input}"
    brake_prefixed_input = f"brake_{brake_input}"
    drive_prefixed_output = f"drive_{drive_output}"
    brake_prefixed_output = f"brake_{brake_output}"

    nodes = list(drive.graph.node) + list(brake.graph.node)
    for node in nodes:
        node.input[:] = [
            "obs" if name in {drive_prefixed_input, brake_prefixed_input} else name
            for name in node.input
        ]
    nodes.extend(
        [
            helper.make_node("Gather", ["obs", "brake_router_command_x_index"], ["brake_router_command_x"], axis=1),
            helper.make_node("Abs", ["brake_router_command_x"], ["brake_router_abs_command_x"]),
            helper.make_node(
                "Less",
                ["brake_router_abs_command_x", "brake_router_zero_command_threshold"],
                ["brake_router_is_zero_command"],
            ),
        ]
    )
    if gate_mode == "joint_velocity":
        nodes.extend(
            [
            helper.make_node(
                "Slice",
                [
                    "obs",
                    "brake_router_joint_velocity_starts",
                    "brake_router_joint_velocity_ends",
                    "brake_router_slice_axes",
                    "brake_router_slice_steps",
                ],
                ["brake_router_joint_velocity"],
            ),
            helper.make_node(
                "Abs",
                ["brake_router_joint_velocity"],
                ["brake_router_abs_joint_velocity"],
            ),
            helper.make_node(
                "ReduceMean",
                ["brake_router_abs_joint_velocity", "brake_router_reduce_axes"],
                ["brake_router_mean_abs_joint_velocity"],
                keepdims=1,
            ),
            helper.make_node(
                "Greater",
                ["brake_router_mean_abs_joint_velocity", "brake_router_joint_velocity_threshold"],
                ["brake_router_is_moving"],
            ),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [
                        "obs",
                        "brake_router_last_action_starts",
                        "brake_router_last_action_ends",
                        "brake_router_slice_axes",
                        "brake_router_slice_steps",
                    ],
                    ["brake_router_last_action"],
                ),
                helper.make_node(
                    "Abs",
                    ["brake_router_last_action"],
                    ["brake_router_abs_last_action"],
                ),
                helper.make_node(
                    "ReduceMean",
                    ["brake_router_abs_last_action", "brake_router_reduce_axes"],
                    ["brake_router_mean_abs_last_action"],
                    keepdims=1,
                ),
                helper.make_node(
                    "Slice",
                    [
                        "obs",
                        "brake_router_base_angular_velocity_starts",
                        "brake_router_base_angular_velocity_ends",
                        "brake_router_slice_axes",
                        "brake_router_slice_steps",
                    ],
                    ["brake_router_base_angular_velocity"],
                ),
                helper.make_node(
                    "Abs",
                    ["brake_router_base_angular_velocity"],
                    ["brake_router_abs_base_angular_velocity"],
                ),
                helper.make_node(
                    "ReduceMean",
                    ["brake_router_abs_base_angular_velocity", "brake_router_reduce_axes"],
                    ["brake_router_mean_abs_base_angular_velocity"],
                    keepdims=1,
                ),
                helper.make_node(
                    "Greater",
                    ["brake_router_mean_abs_last_action", "brake_router_high_action_threshold"],
                    ["brake_router_high_action"],
                ),
                helper.make_node(
                    "Less",
                    ["brake_router_mean_abs_last_action", "brake_router_low_action_threshold"],
                    ["brake_router_low_action"],
                ),
                helper.make_node(
                    "Greater",
                    ["brake_router_mean_abs_base_angular_velocity", "brake_router_angular_velocity_threshold"],
                    ["brake_router_coasting_rotation"],
                ),
                helper.make_node(
                    "And",
                    ["brake_router_low_action", "brake_router_coasting_rotation"],
                    ["brake_router_low_action_coasting"],
                ),
                helper.make_node(
                    "Or",
                    ["brake_router_high_action", "brake_router_low_action_coasting"],
                    ["brake_router_is_moving"],
                ),
            ]
        )
    nodes.extend(
        [
            helper.make_node(
                "And",
                ["brake_router_is_zero_command", "brake_router_is_moving"],
                ["brake_router_use_transition_policy"],
            ),
            helper.make_node(
                "Where",
                ["brake_router_use_transition_policy", brake_prefixed_output, drive_prefixed_output],
                ["actions"],
            ),
        ]
    )
    initializers = list(drive.graph.initializer) + list(brake.graph.initializer)
    initializers.extend(
        [
            numpy_helper.from_array(
                np.asarray([COMMAND_X_INDEX], dtype=np.int64), "brake_router_command_x_index"
            ),
            numpy_helper.from_array(
                np.asarray([zero_command_threshold], dtype=np.float32),
                "brake_router_zero_command_threshold",
            ),
            numpy_helper.from_array(
                np.asarray([JOINT_VELOCITY_START], dtype=np.int64),
                "brake_router_joint_velocity_starts",
            ),
            numpy_helper.from_array(
                np.asarray([JOINT_VELOCITY_END], dtype=np.int64),
                "brake_router_joint_velocity_ends",
            ),
            numpy_helper.from_array(
                np.asarray([1], dtype=np.int64), "brake_router_slice_axes"
            ),
            numpy_helper.from_array(
                np.asarray([1], dtype=np.int64), "brake_router_slice_steps"
            ),
            numpy_helper.from_array(
                np.asarray([1], dtype=np.int64), "brake_router_reduce_axes"
            ),
            numpy_helper.from_array(
                np.asarray([joint_velocity_threshold], dtype=np.float32),
                "brake_router_joint_velocity_threshold",
            ),
            numpy_helper.from_array(
                np.asarray([LAST_ACTION_START], dtype=np.int64),
                "brake_router_last_action_starts",
            ),
            numpy_helper.from_array(
                np.asarray([LAST_ACTION_END], dtype=np.int64),
                "brake_router_last_action_ends",
            ),
            numpy_helper.from_array(
                np.asarray([BASE_ANGULAR_VELOCITY_START], dtype=np.int64),
                "brake_router_base_angular_velocity_starts",
            ),
            numpy_helper.from_array(
                np.asarray([BASE_ANGULAR_VELOCITY_END], dtype=np.int64),
                "brake_router_base_angular_velocity_ends",
            ),
            numpy_helper.from_array(
                np.asarray([high_action_threshold], dtype=np.float32),
                "brake_router_high_action_threshold",
            ),
            numpy_helper.from_array(
                np.asarray([low_action_threshold], dtype=np.float32),
                "brake_router_low_action_threshold",
            ),
            numpy_helper.from_array(
                np.asarray([angular_velocity_threshold], dtype=np.float32),
                "brake_router_angular_velocity_threshold",
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "microduck_zero_command_brake_router",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBSERVATION_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
        initializer=initializers,
        value_info=list(drive.graph.value_info) + list(brake.graph.value_info),
    )
    model = helper.make_model(
        graph,
        producer_name="ducklab-brake-router",
        opset_imports=list(drive.opset_import),
        ir_version=max(drive.ir_version, brake.ir_version),
    )
    helper.set_model_props(
        model,
        {
            **{prop.key: prop.value for prop in drive.metadata_props},
            "brake_safe_drive_policy": str(drive_path.resolve()),
            "brake_safe_transition_policy": str(brake_path.resolve()),
            "brake_safe_route": (
                f"brake policy when abs(command_x)<{zero_command_threshold:g} and "
                + (
                    f"mean(abs(joint_velocity))>{joint_velocity_threshold:g} rad/s"
                    if gate_mode == "joint_velocity"
                    else (
                        f"mean_abs(last_action)>{high_action_threshold:g} or "
                        f"(mean_abs(last_action)<{low_action_threshold:g} and "
                        f"mean_abs(base_ang_vel)>{angular_velocity_threshold:g})"
                    )
                )
            ),
        },
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drive_policy", type=Path)
    parser.add_argument("brake_policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zero-command-threshold", type=float, default=0.02)
    parser.add_argument("--joint-velocity-threshold", type=float, default=0.25)
    parser.add_argument(
        "--gate-mode",
        choices=("joint_velocity", "brake_signature"),
        default="joint_velocity",
    )
    parser.add_argument("--high-action-threshold", type=float, default=0.42)
    parser.add_argument("--low-action-threshold", type=float, default=0.10)
    parser.add_argument("--angular-velocity-threshold", type=float, default=0.08)
    args = parser.parse_args()
    build_brake_safe_policy(
        args.drive_policy,
        args.brake_policy,
        args.output,
        zero_command_threshold=args.zero_command_threshold,
        joint_velocity_threshold=args.joint_velocity_threshold,
        gate_mode=args.gate_mode,
        high_action_threshold=args.high_action_threshold,
        low_action_threshold=args.low_action_threshold,
        angular_velocity_threshold=args.angular_velocity_threshold,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
