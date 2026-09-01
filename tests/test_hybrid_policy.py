from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


LAB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = LAB_ROOT / "tools" / "build_hybrid_policy.py"
SPEC = importlib.util.spec_from_file_location("build_hybrid_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
hybrid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hybrid
SPEC.loader.exec_module(hybrid)


def test_command_router_selects_speed_and_control_policies(tmp_path: Path) -> None:
    control_path = (
        LAB_ROOT
        / "policy-bench/runs/race5-2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42-i10/artifacts/ducklab-race5-v11-drag-launch-i10-s42.onnx"
    )
    speed_path = (
        LAB_ROOT
        / "policy-bench/runs/race5-2026-08-31_09-06-07_ducklab-speed-retention-v3-straight-e4096-i6000-s42-i6159/artifacts/2026-08-31_09-06-07_ducklab-speed-retention-v3-straight-e4096-i6000-s42.onnx"
    )
    if not control_path.is_file() or not speed_path.is_file():
        return
    output = tmp_path / "hybrid.onnx"
    hybrid.build_hybrid(control_path, speed_path, output)

    control = ort.InferenceSession(str(control_path), providers=["CPUExecutionProvider"])
    speed = ort.InferenceSession(str(speed_path), providers=["CPUExecutionProvider"])
    routed = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    observation = np.linspace(-0.2, 0.2, 61, dtype=np.float32)[None, :]

    def infer(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
        return session.run(None, {session.get_inputs()[0].name: obs})[0]

    for command_x, command_yaw, expected in (
        (0.30, 0.0, control),
        (0.80, 0.0, speed),
        (0.80, 0.18, speed),
        (0.80, 0.30, control),
        (0.20, -0.30, control),
        (0.0, 0.0, control),
    ):
        obs = observation.copy()
        obs[0, hybrid.COMMAND_X_INDEX] = command_x
        obs[0, hybrid.COMMAND_YAW_INDEX] = command_yaw
        np.testing.assert_allclose(infer(routed, obs), infer(expected, obs), rtol=1e-6, atol=1e-6)
