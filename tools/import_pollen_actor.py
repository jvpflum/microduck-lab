#!/usr/bin/env python3
"""Import a deployed Pollen ONNX actor into an RSL-RL checkpoint scaffold.

The ONNX export contains the deterministic actor and observation normalizer,
but not the PPO critic, exploration standard deviation, or optimizer.  A local
checkpoint supplies that container shape; every policy-defining tensor is then
replaced from ONNX and optimizer history is cleared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


ACTOR_TENSORS = (
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory_onnx", type=Path)
    parser.add_argument("checkpoint_scaffold", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exploration-std", type=float, default=0.10)
    args = parser.parse_args()

    model = onnx.load(args.factory_onnx)
    initializers = _initializers(model)
    missing = [name for name in ("obs_normalizer._mean", "onnx::Div_24", *ACTOR_TENSORS) if name not in initializers]
    if missing:
        raise SystemExit(f"Factory ONNX is missing expected tensors: {missing}")

    checkpoint = torch.load(args.checkpoint_scaffold, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    for name in ACTOR_TENSORS:
        value = torch.from_numpy(initializers[name]).to(dtype=actor[name].dtype)
        if value.shape != actor[name].shape:
            raise SystemExit(f"Shape mismatch for {name}: ONNX {tuple(value.shape)}, checkpoint {tuple(actor[name].shape)}")
        actor[name] = value

    mean = torch.from_numpy(initializers["obs_normalizer._mean"]).to(dtype=actor["obs_normalizer._mean"].dtype)
    std = torch.from_numpy(initializers["onnx::Div_24"]).to(dtype=actor["obs_normalizer._std"].dtype)
    if mean.shape != actor["obs_normalizer._mean"].shape or std.shape != actor["obs_normalizer._std"].shape:
        raise SystemExit("Factory normalizer is incompatible with the 61D Sprint actor")
    actor["obs_normalizer._mean"] = mean
    actor["obs_normalizer._std"] = std
    actor["obs_normalizer._var"] = std.square()
    # The deployed ONNX does not contain the training sample count.  A large
    # count preserves its known-good normalization during a short fine-tune;
    # Sprint-v1 accidentally retained the tiny smoke count and immediately
    # rewrote the actor's input distribution.
    actor["obs_normalizer.count"] = torch.full_like(
        actor["obs_normalizer.count"], 1_000_000_000
    )
    actor["distribution.std_param"] = torch.full_like(
        actor["distribution.std_param"], args.exploration_std
    )

    # The scaffold contributes only a critic and checkpoint schema.  PPO starts
    # at iteration zero with fresh Adam moments under the Sprint objective.
    checkpoint["optimizer_state_dict"]["state"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "warmstart": "pollen-factory-roller-onnx",
        "factory_onnx": str(args.factory_onnx.resolve()),
        "checkpoint_scaffold": str(args.checkpoint_scaffold.resolve()),
        "optimizer_reset": True,
        "actor_normalizer_count": 1_000_000_000,
    }

    # Independent fixed-vector check of every imported deterministic actor
    # tensor before writing the checkpoint.
    rng = np.random.default_rng(20260830)
    obs = initializers["obs_normalizer._mean"] + initializers["onnx::Div_24"] * rng.normal(
        scale=0.5, size=(1, 61)
    ).astype(np.float32)
    session = ort.InferenceSession(str(args.factory_onnx), providers=["CPUExecutionProvider"])
    expected = session.run(None, {"obs": obs.astype(np.float32)})[0]
    value = (torch.from_numpy(obs) - mean) / std
    for index, prefix in enumerate(("mlp.0", "mlp.2", "mlp.4", "mlp.6")):
        value = torch.nn.functional.linear(value, actor[f"{prefix}.weight"], actor[f"{prefix}.bias"])
        if index < 3:
            value = torch.nn.functional.elu(value)
    max_error = float(np.max(np.abs(value.detach().numpy() - expected)))
    if max_error > 1e-5:
        raise SystemExit(f"Imported actor failed equivalence check: max error {max_error:.3g}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Imported Pollen actor: {args.output}")
    print(f"Deterministic output max error: {max_error:.3g}")


if __name__ == "__main__":
    main()
