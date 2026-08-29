#!/usr/bin/env python3
"""Compare a Pollen RSL-RL checkpoint with its exported ONNX policy.

Both policies receive the exact same 61D actor observations collected from the
official mjlab environment.  This isolates export/normalizer mistakes from
differences between mjlab training physics and the CPU deployment rehearsal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--task", default="Mjlab-Velocity-Swizzle-MicroDuck")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    if not checkpoint.is_file() or not onnx_path.is_file():
        raise SystemExit("Checkpoint and ONNX paths must both exist")

    configure_torch_backends()
    import mjlab.tasks  # noqa: F401  # Populate the task registry.

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(args.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device="cpu")
        runner.load(str(checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        max_errors: list[float] = []
        rms_errors: list[float] = []
        obs = env.get_observations().to("cpu")
        with torch.inference_mode():
            for _ in range(args.samples):
                torch_action = policy(obs)
                flat_obs = torch.cat([obs[name] for name in policy.obs_groups], dim=-1)
                onnx_action = session.run(
                    [output_name], {input_name: flat_obs.numpy().astype(np.float32)}
                )[0]
                delta = torch_action.numpy() - onnx_action
                max_errors.append(float(np.max(np.abs(delta))))
                rms_errors.append(float(np.sqrt(np.mean(np.square(delta)))))
                obs, _, _, _ = env.step(torch_action)
                obs = obs.to("cpu")

        max_abs_error = max(max_errors, default=float("inf"))
        result = {
            "schema_version": 1,
            "task": args.task,
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "samples": args.samples,
            "observation_dim": int(flat_obs.shape[-1]),
            "action_dim": int(torch_action.shape[-1]),
            "max_abs_action_error": max_abs_error,
            "mean_rms_action_error": float(np.mean(rms_errors)),
            "tolerance": args.tolerance,
            "passed": max_abs_error <= args.tolerance,
        }
    finally:
        env.close()

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
