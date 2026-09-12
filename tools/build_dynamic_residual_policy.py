#!/usr/bin/env python3
"""Add a bounded joint-state residual to a compatible MicroDuck actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ACTION_DIM = 14
PROPULSION = (2, 3, 4, 11, 12, 13)


def _vector(value: float) -> np.ndarray:
    result = np.zeros(ACTION_DIM, dtype=np.float32)
    result[list(PROPULSION)] = value
    return result


def build(source: Path, output: Path, pos_gain: float, vel_gain: float, last_gain: float) -> None:
    model = onnx.load(source)
    if len(model.graph.output) != 1:
        raise ValueError("source actor must have one output")
    output_info = model.graph.output[0]
    raw_name = output_info.name
    output_info.name = "dynamic_residual_actions"
    nodes = model.graph.node
    nodes.extend(
        [
            helper.make_node("Slice", ["obs", "slice_pos_start", "slice_pos_end", "slice_axis"], ["joint_pos"], name="residual_joint_pos"),
            helper.make_node("Slice", ["obs", "slice_vel_start", "slice_vel_end", "slice_axis"], ["joint_vel"], name="residual_joint_vel"),
            helper.make_node("Slice", ["obs", "slice_last_start", "slice_last_end", "slice_axis"], ["last_action"], name="residual_last_action"),
            helper.make_node("Mul", ["joint_pos", "dynamic_pos_gain"], ["pos_delta"]),
            helper.make_node("Mul", ["joint_vel", "dynamic_vel_gain"], ["vel_delta"]),
            helper.make_node("Mul", ["last_action", "dynamic_last_gain"], ["last_delta"]),
            helper.make_node("Add", ["pos_delta", "vel_delta"], ["state_delta"]),
            helper.make_node("Add", ["state_delta", "last_delta"], ["dynamic_delta"]),
            helper.make_node("Add", [raw_name, "dynamic_delta"], ["dynamic_residual_actions"]),
        ]
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([6], dtype=np.int64), "slice_pos_start"),
            numpy_helper.from_array(np.asarray([20], dtype=np.int64), "slice_pos_end"),
            numpy_helper.from_array(np.asarray([20], dtype=np.int64), "slice_vel_start"),
            numpy_helper.from_array(np.asarray([34], dtype=np.int64), "slice_vel_end"),
            numpy_helper.from_array(np.asarray([34], dtype=np.int64), "slice_last_start"),
            numpy_helper.from_array(np.asarray([48], dtype=np.int64), "slice_last_end"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "slice_axis"),
            numpy_helper.from_array(_vector(pos_gain).reshape(1, ACTION_DIM), "dynamic_pos_gain"),
            numpy_helper.from_array(_vector(vel_gain).reshape(1, ACTION_DIM), "dynamic_vel_gain"),
            numpy_helper.from_array(_vector(last_gain).reshape(1, ACTION_DIM), "dynamic_last_gain"),
        ]
    )
    props = {prop.key: prop.value for prop in model.metadata_props}
    props["ducklab_dynamic_residual"] = f"propulsion: pos={pos_gain:g}, vel={vel_gain:g}, last={last_gain:g}"
    del model.metadata_props[:]
    for key, value in props.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pos-gain", type=float, default=0.0)
    parser.add_argument("--vel-gain", type=float, default=0.0)
    parser.add_argument("--last-gain", type=float, default=0.0)
    args = parser.parse_args()
    build(args.source, args.output, args.pos_gain, args.vel_gain, args.last_gain)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
