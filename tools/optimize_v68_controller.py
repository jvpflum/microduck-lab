#!/usr/bin/env python3
"""Recover line tracking for a fast V68 joint-fusion candidate using CEM."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
TOOLS = LAB_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from optimize_v68_joint_fusion import extract_metrics, score_metrics  # noqa: E402


PARAMETER_NAMES = ("yaw_kp", "lateral_kp", "yaw_kd", "max_wz")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate_id(values: np.ndarray) -> str:
    payload = ",".join(f"{value:.7f}" for value in values)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def evaluate_candidate(policy: Path, values: np.ndarray, output_dir: Path) -> tuple[Path, dict]:
    identifier = candidate_id(values)
    evaluation_path = output_dir / "evaluations" / f"{identifier}.json"
    log_path = output_dir / "logs" / f"{identifier}.log"
    if not evaluation_path.exists():
        command = [
            sys.executable,
            str(TOOLS / "evaluate_swizzle.py"),
            str(policy),
            "--profile", "race-5mph",
            "--current-limit", "1.75",
            "--wheel-friction", "0.003",
            "--line-hold",
            "--line-yaw-kp", f"{values[0]:.9g}",
            "--line-lateral-kp", f"{values[1]:.9g}",
            "--line-yaw-kd", f"{values[2]:.9g}",
            "--line-max-wz", f"{values[3]:.9g}",
            "--output", str(evaluation_path),
        ]
        with log_path.open("w") as log:
            subprocess.run(command, cwd=UPSTREAM, stdout=log, stderr=subprocess.STDOUT, check=True)
    return evaluation_path, load_json(evaluation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=LAB_ROOT / "reports" / "duckwing-v68-cem" / "policies" / "3532e081bcc6.onnx",
    )
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=7)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6811)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAB_ROOT / "reports" / "duckwing-v68-controller-cem",
    )
    args = parser.parse_args()
    if not 2 <= args.elite < args.population:
        raise SystemExit("elite must be at least 2 and smaller than population")
    policy = args.policy.resolve()
    if not policy.is_file():
        raise SystemExit(f"Candidate policy missing: {policy}")
    baseline_path = LAB_ROOT / "releases" / "v67" / "leader-metrics.json"
    if not baseline_path.is_file():
        raise SystemExit(f"V67 baseline missing: {baseline_path}")

    output_dir = args.output_dir.resolve()
    for child in ("evaluations", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    baseline = load_json(baseline_path)
    rng = np.random.default_rng(args.seed)
    mean = np.asarray([0.70, 0.22, 0.07, 0.15], dtype=np.float64)
    sigma = np.asarray([0.10, 0.055, 0.025, 0.035], dtype=np.float64)
    lower = np.asarray([0.40, 0.05, 0.01, 0.08], dtype=np.float64)
    upper = np.asarray([1.10, 0.45, 0.18, 0.28], dtype=np.float64)
    all_records: list[dict] = []

    for generation in range(args.generations):
        samples = np.clip(rng.normal(mean, sigma, size=(args.population, len(mean))), lower, upper)
        samples[0] = mean
        if generation == 0:
            samples[1] = np.asarray([0.70, 0.22, 0.07, 0.15])
        generation_records = []
        for index, values in enumerate(samples):
            evaluation_path, evaluation = evaluate_candidate(policy, values, output_dir)
            metrics = extract_metrics(evaluation)
            fitness, improvements, strict = score_metrics(metrics, baseline)
            record = {
                "generation": generation,
                "index": index,
                "candidate_id": candidate_id(values),
                "parameters": dict(zip(PARAMETER_NAMES, (float(value) for value in values), strict=True)),
                "policy": str(policy),
                "evaluation": str(evaluation_path),
                "fitness": fitness,
                "strict_promotable": strict,
                "metrics": metrics,
                "improvement_ratios": improvements,
            }
            generation_records.append(record)
            all_records.append(record)
        generation_records.sort(key=lambda item: item["fitness"], reverse=True)
        elites = generation_records[: args.elite]
        elite_values = np.asarray(
            [[record["parameters"][name] for name in PARAMETER_NAMES] for record in elites]
        )
        mean = 0.25 * mean + 0.75 * elite_values.mean(axis=0)
        sigma = np.maximum(
            np.asarray([0.004, 0.004, 0.002, 0.002]),
            0.25 * sigma + 0.75 * elite_values.std(axis=0),
        )
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "generation": generation,
                    "mean": dict(zip(PARAMETER_NAMES, mean.tolist(), strict=True)),
                    "sigma": dict(zip(PARAMETER_NAMES, sigma.tolist(), strict=True)),
                    "best": generation_records[0],
                    "strict_count": sum(record["strict_promotable"] for record in all_records),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"generation {generation}: best={generation_records[0]['fitness']:.6f} "
            f"strict_total={sum(record['strict_promotable'] for record in all_records)}",
            flush=True,
        )

    all_records.sort(key=lambda item: item["fitness"], reverse=True)
    strict_records = [record for record in all_records if record["strict_promotable"]]
    (output_dir / "scorecard.json").write_text(json.dumps(all_records, indent=2) + "\n")
    (output_dir / "promotion-candidates.json").write_text(json.dumps(strict_records, indent=2) + "\n")
    print(
        f"V68 controller CEM complete: {len(strict_records)} strict candidates; "
        f"scorecard={output_dir / 'scorecard.json'}"
    )


if __name__ == "__main__":
    main()
