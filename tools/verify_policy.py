#!/usr/bin/env python3
"""Validate the deployable policy contract and CPU MuJoCo model."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort


parser = argparse.ArgumentParser()
parser.add_argument("policy", type=Path)
parser.add_argument("--roller", action="store_true")
args = parser.parse_args()

policy_path = args.policy.resolve()
onnx_model = onnx.load(policy_path)
onnx.checker.check_model(onnx_model)

session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
output_meta = session.get_outputs()[0]
if input_meta.shape != [1, 61]:
    raise SystemExit(f"Unexpected observation shape: {input_meta.shape}")
if output_meta.shape != [1, 14]:
    raise SystemExit(f"Unexpected action shape: {output_meta.shape}")

rng = np.random.default_rng(42)
for observation in (
    np.zeros((1, 61), dtype=np.float32),
    rng.normal(0.0, 0.1, (1, 61)).astype(np.float32),
):
    action = session.run([output_meta.name], {input_meta.name: observation})[0]
    if action.shape != (1, 14) or not np.isfinite(action).all():
        raise SystemExit("ONNX inference returned invalid actions")

scene_name = "scene_rollers.xml" if args.roller else "scene.xml"
scene_path = Path("src/mjlab_microduck/robot/microduck") / scene_name
model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
    raise SystemExit("CPU MuJoCo rehearsal produced invalid state")

metadata = session.get_modelmeta().custom_metadata_map
required_metadata = {"joint_names", "default_joint_pos", "action_scale", "observation_names"}
missing_metadata = sorted(required_metadata - metadata.keys())
if missing_metadata:
    raise SystemExit(f"Missing deployment metadata: {missing_metadata}")

print(json.dumps({
    "policy": str(policy_path),
    "onnx_valid": True,
    "input_shape": input_meta.shape,
    "output_shape": output_meta.shape,
    "finite_inference": True,
    "cpu_mujoco_steps": 100,
    "robot_model": "rollers" if args.roller else "feet",
    "metadata_keys": sorted(metadata),
}, indent=2, sort_keys=True))
