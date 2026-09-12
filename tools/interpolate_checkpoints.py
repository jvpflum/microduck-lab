#!/usr/bin/env python3
"""Linearly interpolate compatible RSL-RL checkpoints from one training path."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


def blend_state_dict(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if left.keys() != right.keys():
        raise ValueError("Checkpoint state dictionaries have different keys")
    blended: dict[str, torch.Tensor] = {}
    for name, left_value in left.items():
        right_value = right[name]
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            raise ValueError(f"Incompatible tensor {name!r}")
        if left_value.is_floating_point():
            blended[name] = torch.lerp(left_value, right_value, alpha)
        else:
            # Normalizer sample counts are integer metadata. Preserve the
            # interpolated effective count without coercing model parameters.
            blended[name] = torch.round(
                torch.lerp(left_value.float(), right_value.float(), alpha)
            ).to(dtype=left_value.dtype)
    return blended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    left: dict[str, Any] = torch.load(args.left, map_location="cpu", weights_only=False)
    right: dict[str, Any] = torch.load(args.right, map_location="cpu", weights_only=False)
    result = copy.deepcopy(left)
    for state_name in ("actor_state_dict", "critic_state_dict"):
        result[state_name] = blend_state_dict(
            left[state_name], right[state_name], args.alpha
        )
    result["iter"] = round(
        (1.0 - args.alpha) * int(left.get("iter", 0))
        + args.alpha * int(right.get("iter", 0))
    )
    result["infos"] = {
        **copy.deepcopy(left.get("infos", {})),
        "ducklab_interpolation": {
            "left": str(args.left.resolve()),
            "right": str(args.right.resolve()),
            "alpha": args.alpha,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
