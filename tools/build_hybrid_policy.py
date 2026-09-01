#!/usr/bin/env python3
"""Compose two MicroDuck ONNX actors into one command-routed policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, compose, helper, numpy_helper
import numpy as np


OBSERVATION_DIM = 61
ACTION_DIM = 14
COMMAND_X_INDEX = 48
COMMAND_YAW_INDEX = 50


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def _validate_actor(model: onnx.ModelProto, label: str) -> None:
    if len(model.graph.input) != 1 or _shape(model.graph.input[0]) != (1, OBSERVATION_DIM):
        raise ValueError(f"{label} must accept one [1, {OBSERVATION_DIM}] observation")
    if len(model.graph.output) != 1 or _shape(model.graph.output[0]) != (1, ACTION_DIM):
        raise ValueError(f"{label} must produce one [1, {ACTION_DIM}] action")


def build_hybrid(
    control_path: Path,
    speed_path: Path,
    output_path: Path,
    *,
    speed_command_threshold: float = 0.5,
    turn_command_threshold: float = 0.25,
    speed_blend: float = 1.0,
    smooth_turn_start: float | None = None,
    smooth_turn_end: float | None = None,
) -> onnx.ModelProto:
    if not 0.0 <= speed_blend <= 1.0:
        raise ValueError("speed_blend must be between 0 and 1")
    if (smooth_turn_start is None) != (smooth_turn_end is None):
        raise ValueError("smooth turn start and end must be provided together")
    if (
        smooth_turn_start is not None
        and smooth_turn_end is not None
        and not 0.0 <= smooth_turn_start < smooth_turn_end
    ):
        raise ValueError("smooth turn band must satisfy 0 <= start < end")
    control = onnx.load(control_path)
    speed = onnx.load(speed_path)
    _validate_actor(control, "control policy")
    _validate_actor(speed, "speed policy")
    if control.opset_import[0].version != speed.opset_import[0].version:
        raise ValueError("Policies use different ONNX opset versions")

    control_input = control.graph.input[0].name
    control_output = control.graph.output[0].name
    speed_input = speed.graph.input[0].name
    speed_output = speed.graph.output[0].name
    control = compose.add_prefix(control, "control_")
    speed = compose.add_prefix(speed, "speed_")
    control_prefixed_input = f"control_{control_input}"
    speed_prefixed_input = f"speed_{speed_input}"
    control_prefixed_output = f"control_{control_output}"
    speed_prefixed_output = f"speed_{speed_output}"

    nodes = list(control.graph.node) + list(speed.graph.node)
    for node in nodes:
        node.input[:] = [
            "obs" if name in {control_prefixed_input, speed_prefixed_input} else name
            for name in node.input
        ]

    nodes.extend(
        [
            helper.make_node("Gather", ["obs", "command_x_index"], ["hybrid_command_x"], axis=1),
            helper.make_node("Gather", ["obs", "command_yaw_index"], ["hybrid_command_yaw"], axis=1),
            helper.make_node("Abs", ["hybrid_command_yaw"], ["hybrid_abs_command_yaw"]),
        ]
    )
    if smooth_turn_start is not None and smooth_turn_end is not None:
        nodes.extend(
            [
                helper.make_node(
                    "Greater",
                    ["hybrid_command_x", "speed_command_threshold"],
                    ["hybrid_is_speed_command"],
                ),
                helper.make_node(
                    "Cast",
                    ["hybrid_is_speed_command"],
                    ["hybrid_speed_command_gate"],
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Sub",
                    ["smooth_turn_end", "hybrid_abs_command_yaw"],
                    ["hybrid_turn_headroom"],
                ),
                helper.make_node(
                    "Div",
                    ["hybrid_turn_headroom", "smooth_turn_span"],
                    ["hybrid_turn_gate_unclipped"],
                ),
                helper.make_node(
                    "Clip",
                    ["hybrid_turn_gate_unclipped", "router_zero", "router_one"],
                    ["hybrid_turn_gate"],
                ),
                helper.make_node(
                    "Mul",
                    ["hybrid_speed_command_gate", "hybrid_turn_gate"],
                    ["hybrid_route_gate"],
                ),
                helper.make_node(
                    "Mul",
                    ["hybrid_route_gate", "speed_blend"],
                    ["hybrid_speed_weight"],
                ),
                helper.make_node(
                    "Sub",
                    [speed_prefixed_output, control_prefixed_output],
                    ["hybrid_action_delta"],
                ),
                helper.make_node(
                    "Mul",
                    ["hybrid_action_delta", "hybrid_speed_weight"],
                    ["hybrid_weighted_delta"],
                ),
                helper.make_node(
                    "Add",
                    [control_prefixed_output, "hybrid_weighted_delta"],
                    ["actions"],
                ),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node(
                    "Greater",
                    ["hybrid_command_x", "speed_command_threshold"],
                    ["hybrid_is_speed_command"],
                ),
                helper.make_node(
                    "Less",
                    ["hybrid_abs_command_yaw", "turn_command_threshold"],
                    ["hybrid_is_straight_command"],
                ),
                helper.make_node(
                    "And",
                    ["hybrid_is_speed_command", "hybrid_is_straight_command"],
                    ["hybrid_use_speed_policy"],
                ),
                helper.make_node(
                    "Mul",
                    [speed_prefixed_output, "speed_blend"],
                    ["hybrid_speed_actions"],
                ),
                helper.make_node(
                    "Mul",
                    [control_prefixed_output, "control_blend"],
                    ["hybrid_control_actions"],
                ),
                helper.make_node(
                    "Add",
                    ["hybrid_speed_actions", "hybrid_control_actions"],
                    ["hybrid_blended_actions"],
                ),
                helper.make_node(
                    "Where",
                    [
                        "hybrid_use_speed_policy",
                        "hybrid_blended_actions",
                        control_prefixed_output,
                    ],
                    ["actions"],
                ),
            ]
        )

    initializers = list(control.graph.initializer) + list(speed.graph.initializer)
    initializers.extend(
        [
            numpy_helper.from_array(np.asarray([COMMAND_X_INDEX], dtype=np.int64), "command_x_index"),
            numpy_helper.from_array(np.asarray([COMMAND_YAW_INDEX], dtype=np.int64), "command_yaw_index"),
            numpy_helper.from_array(
                np.asarray([speed_command_threshold], dtype=np.float32),
                "speed_command_threshold",
            ),
            numpy_helper.from_array(np.asarray([speed_blend], dtype=np.float32), "speed_blend"),
        ]
    )
    if smooth_turn_start is not None and smooth_turn_end is not None:
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.asarray([smooth_turn_end], dtype=np.float32), "smooth_turn_end"
                ),
                numpy_helper.from_array(
                    np.asarray([smooth_turn_end - smooth_turn_start], dtype=np.float32),
                    "smooth_turn_span",
                ),
                numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "router_zero"),
                numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "router_one"),
            ]
        )
    else:
        initializers.extend(
            [
                numpy_helper.from_array(
                    np.asarray([turn_command_threshold], dtype=np.float32),
                    "turn_command_threshold",
                ),
                numpy_helper.from_array(
                    np.asarray([1.0 - speed_blend], dtype=np.float32), "control_blend"
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "microduck_command_routed_hybrid",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBSERVATION_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
        initializer=initializers,
        value_info=list(control.graph.value_info) + list(speed.graph.value_info),
    )
    hybrid = helper.make_model(
        graph,
        producer_name="ducklab-command-router",
        opset_imports=list(control.opset_import),
        ir_version=max(control.ir_version, speed.ir_version),
    )
    for prop in speed.metadata_props:
        metadata = hybrid.metadata_props.add()
        metadata.key = prop.key
        metadata.value = prop.value
    helper.set_model_props(
        hybrid,
        {
            **{prop.key: prop.value for prop in hybrid.metadata_props},
            "hybrid_control_policy": str(control_path.resolve()),
            "hybrid_speed_policy": str(speed_path.resolve()),
            "hybrid_route": (
                f"speed authority tapers from abs(command_yaw)<={smooth_turn_start:g} "
                f"to control at abs(command_yaw)>={smooth_turn_end:g}; "
                f"command_x>{speed_command_threshold:g}; speed blend {speed_blend:g}"
                if smooth_turn_start is not None and smooth_turn_end is not None
                else f"speed when command_x>{speed_command_threshold:g} and "
                f"abs(command_yaw)<{turn_command_threshold:g}; control otherwise; "
                f"speed blend {speed_blend:g}"
            ),
        },
    )
    onnx.checker.check_model(hybrid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(hybrid, output_path)
    return hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_policy", type=Path)
    parser.add_argument("speed_policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed-command-threshold", type=float, default=0.5)
    parser.add_argument("--turn-command-threshold", type=float, default=0.25)
    parser.add_argument("--speed-blend", type=float, default=1.0)
    parser.add_argument("--smooth-turn-start", type=float)
    parser.add_argument("--smooth-turn-end", type=float)
    args = parser.parse_args()
    build_hybrid(
        args.control_policy,
        args.speed_policy,
        args.output,
        speed_command_threshold=args.speed_command_threshold,
        turn_command_threshold=args.turn_command_threshold,
        speed_blend=args.speed_blend,
        smooth_turn_start=args.smooth_turn_start,
        smooth_turn_end=args.smooth_turn_end,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
