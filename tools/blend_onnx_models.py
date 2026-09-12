#!/usr/bin/env python3
"""Linearly blend compatible ONNX policy parameters."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def blend_models(left: onnx.ModelProto, right: onnx.ModelProto, alpha: float) -> onnx.ModelProto:
    if [node.SerializeToString() for node in left.graph.node] != [
        node.SerializeToString() for node in right.graph.node
    ]:
        raise ValueError("ONNX graphs differ; only parameter-compatible models can be blended")

    right_initializers = {tensor.name: tensor for tensor in right.graph.initializer}
    if {tensor.name for tensor in left.graph.initializer} != set(right_initializers):
        raise ValueError("ONNX initializer names differ")

    result = copy.deepcopy(left)
    del result.graph.initializer[:]
    for left_tensor in left.graph.initializer:
        right_tensor = right_initializers[left_tensor.name]
        left_array = numpy_helper.to_array(left_tensor)
        right_array = numpy_helper.to_array(right_tensor)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            raise ValueError(f"Incompatible initializer {left_tensor.name!r}")
        if np.issubdtype(left_array.dtype, np.floating):
            blended = (1.0 - alpha) * left_array + alpha * right_array
            blended = blended.astype(left_array.dtype, copy=False)
        else:
            if not np.array_equal(left_array, right_array):
                raise ValueError(f"Non-floating initializer differs: {left_tensor.name!r}")
            blended = left_array
        result.graph.initializer.append(numpy_helper.from_array(blended, left_tensor.name))

    del result.metadata_props[:]
    for prop in left.metadata_props:
        metadata = result.metadata_props.add()
        metadata.key = prop.key
        metadata.value = prop.value
    for key, value in {
        "ducklab_blend_left": "parameter model",
        "ducklab_blend_alpha": f"{alpha:.9g}",
    }.items():
        metadata = result.metadata_props.add()
        metadata.key = key
        metadata.value = value
    onnx.checker.check_model(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args()
    if not args.allow_extrapolation and not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    left = onnx.load(args.left)
    right = onnx.load(args.right)
    result = blend_models(left, right, args.alpha)
    metadata = result.metadata_props.add()
    metadata.key = "ducklab_blend_sources"
    metadata.value = f"{args.left.resolve()} -> {args.right.resolve()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
