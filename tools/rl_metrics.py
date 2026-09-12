#!/usr/bin/env python3
"""Read TensorBoard scalar events into a portable, local JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def downsample(points: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
    if len(points) <= limit:
        return points
    indexes = [round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)]
    return [points[index] for index in indexes]


def collect_metrics(run_dir: Path, limit: int = 600) -> dict[str, Any]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required to ingest training curves") from error
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 100_000}).Reload()
    scalars: dict[str, list[dict[str, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = downsample(
            [{"step": float(item.step), "wall_time": float(item.wall_time), "value": float(item.value)} for item in accumulator.Scalars(tag)],
            limit,
        )
    return {
        "schema_version": 1,
        "source_run_dir": str(run_dir.resolve()),
        "event_files": [str(path.resolve()) for path in event_files],
        "scalar_count": len(scalars),
        "scalars": scalars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    result = collect_metrics(args.run_dir, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scalar_count": result["scalar_count"]}))


if __name__ == "__main__":
    main()
