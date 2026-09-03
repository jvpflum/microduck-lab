#!/usr/bin/env python3
"""Add a high-command-only joint-state residual to a MicroDuck actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ACTION_DIM = 14
PROPULSION = (2, 3, 4, 11, 12, 13)


def gain_vector(value: float) -> np.ndarray:
    result = np.zeros((1, ACTION_DIM), dtype=np.float32)
    result[0, list(PROPULSION)] = value
    return result


def build(
    source: Path,
    output: Path,
    *,
    pos_gain: float,
    vel_gain: float,
    last_gain: float,
    command_threshold: float,
) -> None:
    model = onnx.load(source)
    if len(model.graph.output) != 1:
        raise ValueError("source actor must have one output")
    output_info = model.graph.output[0]
    raw_name = output_info.name
    output_info.name = "command_gated_dynamic_actions"
    for node in model.graph.node:
        node.output[:] = ["command_gated_base_actions" if name == raw_name else name for name in node.output]
    model.graph.node.extend([
        helper.make_node("Slice", ["obs", "cg_pos_start", "cg_pos_end", "cg_axis"], ["cg_joint_pos"]),
        helper.make_node("Slice", ["obs", "cg_vel_start", "cg_vel_end", "cg_axis"], ["cg_joint_vel"]),
        helper.make_node("Slice", ["obs", "cg_last_start", "cg_last_end", "cg_axis"], ["cg_last_action"]),
        helper.make_node("Slice", ["obs", "cg_cmd_start", "cg_cmd_end", "cg_axis"], ["cg_command_x"]),
        helper.make_node("Greater", ["cg_command_x", "cg_command_threshold"], ["cg_high_command_bool"]),
        helper.make_node("Cast", ["cg_high_command_bool"], ["cg_high_command_gate"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["cg_joint_pos", "cg_pos_gain"], ["cg_pos_delta"]),
        helper.make_node("Mul", ["cg_joint_vel", "cg_vel_gain"], ["cg_vel_delta"]),
        helper.make_node("Mul", ["cg_last_action", "cg_last_gain"], ["cg_last_delta"]),
        helper.make_node("Add", ["cg_pos_delta", "cg_vel_delta"], ["cg_state_delta"]),
        helper.make_node("Add", ["cg_state_delta", "cg_last_delta"], ["cg_raw_delta"]),
        helper.make_node("Mul", ["cg_raw_delta", "cg_high_command_gate"], ["cg_gated_delta"]),
        helper.make_node("Add", ["command_gated_base_actions", "cg_gated_delta"], ["command_gated_dynamic_actions"]),
    ])
    model.graph.initializer.extend([
        numpy_helper.from_array(np.asarray([6], dtype=np.int64), "cg_pos_start"),
        numpy_helper.from_array(np.asarray([20], dtype=np.int64), "cg_pos_end"),
        numpy_helper.from_array(np.asarray([20], dtype=np.int64), "cg_vel_start"),
        numpy_helper.from_array(np.asarray([34], dtype=np.int64), "cg_vel_end"),
        numpy_helper.from_array(np.asarray([34], dtype=np.int64), "cg_last_start"),
        numpy_helper.from_array(np.asarray([48], dtype=np.int64), "cg_last_end"),
        numpy_helper.from_array(np.asarray([48], dtype=np.int64), "cg_cmd_start"),
        numpy_helper.from_array(np.asarray([49], dtype=np.int64), "cg_cmd_end"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "cg_axis"),
        numpy_helper.from_array(np.asarray([command_threshold], dtype=np.float32), "cg_command_threshold"),
        numpy_helper.from_array(gain_vector(pos_gain), "cg_pos_gain"),
        numpy_helper.from_array(gain_vector(vel_gain), "cg_vel_gain"),
        numpy_helper.from_array(gain_vector(last_gain), "cg_last_gain"),
    ])
    properties = {prop.key: prop.value for prop in model.metadata_props}
    properties["ducklab_command_gated_dynamic_residual"] = (
        f"command_x>{command_threshold:g}; propulsion pos={pos_gain:g}, "
        f"vel={vel_gain:g}, last={last_gain:g}"
    )
    del model.metadata_props[:]
    for key, value in properties.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pos-gain", type=float, required=True)
    parser.add_argument("--vel-gain", type=float, required=True)
    parser.add_argument("--last-gain", type=float, required=True)
    parser.add_argument("--command-threshold", type=float, default=0.5)
    args = parser.parse_args()
    build(
        args.source,
        args.output,
        pos_gain=args.pos_gain,
        vel_gain=args.vel_gain,
        last_gain=args.last_gain,
        command_threshold=args.command_threshold,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
