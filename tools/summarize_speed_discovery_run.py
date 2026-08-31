#!/usr/bin/env python3
"""Summarize learning throughput and GPU efficiency for an env-count A/B run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def numbers(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.training_log.read_text(encoding="utf-8", errors="replace")
    reported_sps = numbers(r"Steps per second:\s*([0-9.]+)", text)
    collection = numbers(r"Collection time:\s*([0-9.]+)s", text)
    learning = numbers(r"Learning time:\s*([0-9.]+)s", text)
    batch = args.envs * args.rollout
    policy_fps = [batch / value for value in collection if value > 0.0]
    update_throughput = [
        batch / (collect + learn)
        for collect, learn in zip(collection, learning, strict=False)
        if collect + learn > 0.0
    ]

    gpu_rows = []
    if args.gpu_log.is_file():
        for line in args.gpu_log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in row:
                gpu_rows.append(row)

    best_path = args.run_dir / "best_speed_discovery.json"
    best = None
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8")).get("best")
    report = {
        "run_dir": str(args.run_dir.resolve()),
        "num_envs": args.envs,
        "rollout_steps": args.rollout,
        "transitions_per_update": batch,
        "iterations_observed": len(reported_sps),
        "reported_sim_steps_per_second": stats(reported_sps),
        "policy_collection_fps": stats(policy_fps),
        "ppo_transitions_per_wall_second": stats(update_throughput),
        "gpu_utilization_percent": stats(
            [float(row["gpu_utilization_percent"]) for row in gpu_rows]
        ),
        "vram_used_mib": stats(
            [float(row["vram_used_mib"]) for row in gpu_rows if row.get("vram_used_mib") is not None]
        ),
        "system_memory_available_mib": stats(
            [
                float(row["system_memory_available_mib"])
                for row in gpu_rows
                if row.get("system_memory_available_mib") is not None
            ]
        ),
        "temperature_c": stats([float(row["temperature_c"]) for row in gpu_rows]),
        "best_checkpoint": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
