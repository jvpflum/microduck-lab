from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "tools"))

from build_state_guarded_policy import build_state_guarded_policy  # noqa: E402


def _actor(path: Path, bias: float) -> None:
    nodes = [
        helper.make_node(
            "Slice", ["obs", "starts", "ends", "axes", "steps"], ["slice"]
        ),
        helper.make_node("Add", ["slice", "bias"], ["actions"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.asarray([14], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "steps"),
            numpy_helper.from_array(np.full((1, 14), bias, dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


class StateGuardedPolicyTest(unittest.TestCase):
    def test_gate_endpoints_and_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate, guarded = (
                root / "baseline.onnx",
                root / "candidate.onnx",
                root / "guarded.onnx",
            )
            _actor(baseline, 0.0)
            _actor(candidate, 2.0)
            build_state_guarded_policy(
                baseline,
                candidate,
                guarded,
                yaw_start=0.2,
                yaw_end=1.0,
                candidate_authority=0.75,
            )
            session = ort.InferenceSession(str(guarded), providers=["CPUExecutionProvider"])
            observation = np.zeros((1, 61), dtype=np.float32)
            for yaw, expected_offset in ((0.0, 1.5), (0.2, 1.5), (0.6, 0.75), (1.0, 0.0), (2.0, 0.0)):
                observation[0, 2] = yaw
                actions = session.run(None, {"obs": observation})[0]
                expected = observation[:, :14] + expected_offset
                np.testing.assert_allclose(actions, expected, rtol=0.0, atol=1e-6)

    def test_rejects_invalid_gate_band(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.onnx", root / "candidate.onnx"
            _actor(baseline, 0.0)
            _actor(candidate, 1.0)
            with self.assertRaisesRegex(ValueError, "0 <= start < end"):
                build_state_guarded_policy(
                    baseline,
                    candidate,
                    root / "guarded.onnx",
                    yaw_start=1.0,
                    yaw_end=1.0,
                )


if __name__ == "__main__":
    unittest.main()
