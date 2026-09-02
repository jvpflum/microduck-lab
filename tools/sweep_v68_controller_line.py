#!/usr/bin/env python3
"""Sweep a controller interpolation line across a V68 promotion boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
TOOLS = LAB_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from optimize_v68_controller import evaluate_candidate, load_json  # noqa: E402
from optimize_v68_joint_fusion import extract_metrics, score_metrics  # noqa: E402


PARAMETER_NAMES = ("yaw_kp", "lateral_kp", "yaw_kd", "max_wz")


def parse_vector(value: str) -> np.ndarray:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != len(PARAMETER_NAMES):
        raise argparse.ArgumentTypeError("controller vector must contain yaw_kp,lateral_kp,yaw_kd,max_wz")
    return np.asarray(parts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--end", type=parse_vector, required=True)
    parser.add_argument("--start", type=parse_vector, default=parse_vector("0.70,0.22,0.07,0.15"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("steps must be at least 2")
    policy = args.policy.resolve()
    if not policy.is_file():
        raise SystemExit(f"Candidate policy missing: {policy}")
    baseline_path = LAB_ROOT / "releases" / "v67" / "leader-metrics.json"
    baseline = load_json(baseline_path)
    output_dir = args.output_dir.resolve()
    for child in ("evaluations", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    records = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, args.steps + 1)):
        values = args.start + fraction * (args.end - args.start)
        evaluation_path, evaluation = evaluate_candidate(policy, values, output_dir)
        metrics = extract_metrics(evaluation)
        fitness, improvements, strict = score_metrics(metrics, baseline)
        record = {
            "index": index,
            "fraction": float(fraction),
            "parameters": dict(zip(PARAMETER_NAMES, values.tolist(), strict=True)),
            "policy": str(policy),
            "evaluation": str(evaluation_path),
            "fitness": fitness,
            "strict_promotable": strict,
            "metrics": metrics,
            "improvement_ratios": improvements,
        }
        records.append(record)
        print(
            f"fraction={fraction:.3f} fitness={fitness:.6f} strict={strict}",
            flush=True,
        )

    records.sort(key=lambda item: item["fitness"], reverse=True)
    strict_records = [record for record in records if record["strict_promotable"]]
    (output_dir / "scorecard.json").write_text(json.dumps(records, indent=2) + "\n")
    (output_dir / "promotion-candidates.json").write_text(json.dumps(strict_records, indent=2) + "\n")
    print(f"Controller line sweep complete: {len(strict_records)} strict candidates")


if __name__ == "__main__":
    main()
