#!/usr/bin/env python3
"""Create a Pareto-ranked, dashboard-ready report from an official speed sweep.

This deliberately reads only deterministic evaluator manifests.  PPO rollout
reward and training-log speed are useful diagnostics, but never decide a winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    """True when left is at least as good on every race metric and better once."""
    keys_up = ("world_x_sustained_mph", "peak_mph", "survival_fraction")
    keys_down = ("max_lateral_deviation_m", "mean_abs_heading_deg")
    at_least = all(float(left[key]) >= float(right[key]) for key in keys_up)
    at_least &= all(float(left[key]) <= float(right[key]) for key in keys_down)
    strictly = any(float(left[key]) > float(right[key]) for key in keys_up)
    strictly |= any(float(left[key]) < float(right[key]) for key in keys_down)
    return at_least and strictly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    candidates: list[dict[str, object]] = []
    for manifest_path in sorted(sweep_dir.glob("*/best_speed_discovery.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best = dict(manifest["best"])
        best["run_name"] = manifest_path.parent.name
        best["selection_manifest"] = str(manifest_path)
        candidates.append(best)
    if not candidates:
        raise SystemExit(f"No best_speed_discovery.json files below {sweep_dir}")

    for candidate in candidates:
        candidate["pareto"] = not any(
            other is not candidate and dominates(other, candidate)
            for other in candidates
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            not bool(row["pareto"]),
            -float(row["world_x_sustained_mph"]),
            -float(row["survival_fraction"]),
            float(row["max_lateral_deviation_m"]),
            float(row["mean_abs_heading_deg"]),
            -float(row["peak_mph"]),
        ),
    )
    result = {
        "protocol": "Race5 official bearing friction 0.003, line-hold evaluation",
        "winner_rule": "Pareto front first; then sustained world-X mph, survival, drift, heading, peak mph",
        "winner": ranked[0],
        "pareto_front": [row for row in ranked if row["pareto"]],
        "candidates": ranked,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    output = args.output or sweep_dir / "official_sweep_scoreboard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
