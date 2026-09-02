#!/usr/bin/env python3
"""Blend a faster candidate into a baseline only while body yaw-rate is calm."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper


OBSERVATION_DIM = 61
ACTION_DIM = 14
BASE_ANGULAR_VELOCITY_Z_INDEX = 2


def _shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)


def _validate_actor(model: onnx.ModelProto, label: str) -> None:
    if len(model.graph.input) != 1 or _shape(model.graph.input[0]) != (1, OBSERVATION_DIM):
        raise ValueError(f"{label} must accept one [1, {OBSERVATION_DIM}] observation")
    if len(model.graph.output) != 1 or _shape(model.graph.output[0]) != (1, ACTION_DIM):
        raise ValueError(f"{label} must produce one [1, {ACTION_DIM}] action")


def build_state_guarded_policy(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    yaw_start: float,
    yaw_end: float,
    candidate_authority: float = 1.0,
) -> onnx.ModelProto:
    """Return baseline + authority * yaw_gate * (candidate - baseline).

    The candidate has full configured authority at or below ``yaw_start``.
    Authority tapers linearly to zero at ``yaw_end`` and the baseline is exact
    above that point.
    """
    if not 0.0 <= yaw_start < yaw_end:
        raise ValueError("yaw band must satisfy 0 <= start < end")
    if not 0.0 <= candidate_authority <= 1.25:
        raise ValueError("candidate authority must be between 0 and 1.25")

    baseline = onnx.load(baseline_path)
    candidate = onnx.load(candidate_path)
    _validate_actor(baseline, "baseline policy")
    _validate_actor(candidate, "candidate policy")
    if list(baseline.opset_import) != list(candidate.opset_import):
        raise ValueError("Policies use different ONNX opset versions")

    baseline_input = baseline.graph.input[0].name
    baseline_output = baseline.graph.output[0].name
    candidate_input = candidate.graph.input[0].name
    candidate_output = candidate.graph.output[0].name
    baseline = compose.add_prefix(baseline, "baseline_")
    candidate = compose.add_prefix(candidate, "candidate_")
    baseline_input = f"baseline_{baseline_input}"
    baseline_output = f"baseline_{baseline_output}"
    candidate_input = f"candidate_{candidate_input}"
    candidate_output = f"candidate_{candidate_output}"

    nodes = list(baseline.graph.node) + list(candidate.graph.node)
    for node in nodes:
        node.input[:] = [
            "obs" if name in {baseline_input, candidate_input} else name
            for name in node.input
        ]
    nodes.extend(
        [
            helper.make_node(
                "Gather",
                ["obs", "state_guard_yaw_index"],
                ["state_guard_yaw_rate"],
                axis=1,
            ),
            helper.make_node("Abs", ["state_guard_yaw_rate"], ["state_guard_abs_yaw_rate"]),
            helper.make_node(
                "Sub",
                ["state_guard_yaw_end", "state_guard_abs_yaw_rate"],
                ["state_guard_headroom"],
            ),
            helper.make_node(
                "Div",
                ["state_guard_headroom", "state_guard_yaw_span"],
                ["state_guard_raw_gate"],
            ),
            helper.make_node(
                "Clip",
                ["state_guard_raw_gate", "state_guard_zero", "state_guard_one"],
                ["state_guard_gate"],
            ),
            helper.make_node(
                "Mul",
                ["state_guard_gate", "state_guard_candidate_authority"],
                ["state_guard_scaled_gate"],
            ),
            helper.make_node(
                "Sub",
                [candidate_output, baseline_output],
                ["state_guard_candidate_delta"],
            ),
            helper.make_node(
                "Mul",
                ["state_guard_candidate_delta", "state_guard_scaled_gate"],
                ["state_guard_gated_delta"],
            ),
            helper.make_node(
                "Add",
                [baseline_output, "state_guard_gated_delta"],
                ["actions"],
            ),
        ]
    )
    initializers = list(baseline.graph.initializer) + list(candidate.graph.initializer)
    initializers.extend(
        [
            numpy_helper.from_array(
                np.asarray([BASE_ANGULAR_VELOCITY_Z_INDEX], dtype=np.int64),
                "state_guard_yaw_index",
            ),
            numpy_helper.from_array(
                np.asarray([yaw_end], dtype=np.float32), "state_guard_yaw_end"
            ),
            numpy_helper.from_array(
                np.asarray([yaw_end - yaw_start], dtype=np.float32),
                "state_guard_yaw_span",
            ),
            numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), "state_guard_zero"),
            numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), "state_guard_one"),
            numpy_helper.from_array(
                np.asarray([candidate_authority], dtype=np.float32),
                "state_guard_candidate_authority",
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "microduck_body_yaw_state_guard",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBSERVATION_DIM])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
        initializer=initializers,
        value_info=list(baseline.graph.value_info) + list(candidate.graph.value_info),
    )
    model = helper.make_model(
        graph,
        producer_name="ducklab-state-guard",
        opset_imports=list(baseline.opset_import),
        ir_version=max(baseline.ir_version, candidate.ir_version),
    )
    helper.set_model_props(
        model,
        {
            **{prop.key: prop.value for prop in baseline.metadata_props},
            "state_guard_baseline": baseline_path.name,
            "state_guard_candidate": candidate_path.name,
            "state_guard_route": (
                f"candidate authority {candidate_authority:g} at abs(base_ang_vel_z)<="
                f"{yaw_start:g}; linear fallback to baseline by {yaw_end:g} rad/s"
            ),
        },
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--yaw-start", type=float, required=True)
    parser.add_argument("--yaw-end", type=float, required=True)
    parser.add_argument("--candidate-authority", type=float, default=1.0)
    args = parser.parse_args()
    build_state_guarded_policy(
        args.baseline,
        args.candidate,
        args.output,
        yaw_start=args.yaw_start,
        yaw_end=args.yaw_end,
        candidate_authority=args.candidate_authority,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
