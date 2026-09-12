#!/usr/bin/env python3
"""Prepare an actor/critic checkpoint for a fresh PPO fine-tuning stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--actor-std",
        type=float,
        default=None,
        help="Optionally reopen Gaussian exploration with this per-action std.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"Checkpoint not found: {source}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise SystemExit(f"Checkpoint is missing keys: {sorted(missing)}")

    # Preserve the trained actor, observation normalizer, and critic, but do not
    # carry Adam momentum from the previous reward function into a new stage.
    optimizer = checkpoint["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer.get("param_groups", []):
        group["lr"] = args.learning_rate
        group["initial_lr"] = args.learning_rate
    if args.actor_std is not None:
        if args.actor_std <= 0.0:
            raise SystemExit("--actor-std must be positive")
        std_name = "distribution.std_param"
        if std_name not in checkpoint["actor_state_dict"]:
            raise SystemExit(f"Checkpoint actor is missing {std_name}")
        checkpoint["actor_state_dict"][std_name].fill_(args.actor_std)
    checkpoint["iter"] = 0
    checkpoint["infos"] = {
        "warmstart_source": str(source),
        "optimizer_reset": True,
        "learning_rate": args.learning_rate,
        "actor_std": args.actor_std,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(output)


if __name__ == "__main__":
    main()
