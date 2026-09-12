#!/usr/bin/env python3
"""CEM refinement of V67's five symmetric joint-pair authorities."""

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

from build_brake_safe_policy import build_brake_safe_policy  # noqa: E402
from build_joint_fusion_policy import build_joint_fusion  # noqa: E402


MPS_TO_MPH = 2.2369362920544
PARAMETER_NAMES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_metrics(evaluation: dict) -> dict[str, float | bool]:
    race = evaluation["phases"]["max_speed"]
    stop = evaluation["phases"]["stop_cruise"]
    return {
        "wheel_frictionloss": float(evaluation["wheel_frictionloss"]),
        "current_limit_a": float(evaluation["current_limit_a"]),
        "finished_100ft": bool(race["finished_100ft"]),
        "finish_time_100ft_s": float(race.get("finish_time_100ft_s") or 999.0),
        "sustained_speed_mph": float(race["steady_mean_world_forward_speed_mps"]) * MPS_TO_MPH,
        "verified_top_speed_mph": float(race["verified_top_speed_0_5s_mph"]),
        "acceleration_mps2": float(race["acceleration_first_second_mps2"]),
        "max_drift_ft": float(race["max_lateral_drift_ft"]),
        "max_heading_error_deg": float(race["max_heading_error_deg"]),
        "max_tilt_deg": float(race["tilt_max_deg"]),
        "grounded_fraction": float(race["steady_both_blades_grounded_fraction"]),
        "stop_time_s": float(stop.get("stop_time_below_0_05_mps_s") or 999.0),
    }


def ratios(metrics: dict[str, float | bool], baseline: dict[str, float | bool]) -> dict[str, float]:
    return {
        "finish": float(baseline["finish_time_100ft_s"]) / float(metrics["finish_time_100ft_s"]) - 1.0,
        "sustained": float(metrics["sustained_speed_mph"]) / float(baseline["sustained_speed_mph"]) - 1.0,
        "top": float(metrics["verified_top_speed_mph"]) / float(baseline["verified_top_speed_mph"]) - 1.0,
        "acceleration": float(metrics["acceleration_mps2"]) / float(baseline["acceleration_mps2"]) - 1.0,
        "drift": float(baseline["max_drift_ft"]) / max(0.01, float(metrics["max_drift_ft"])) - 1.0,
        "heading": float(baseline["max_heading_error_deg"]) / max(0.1, float(metrics["max_heading_error_deg"])) - 1.0,
        "tilt": float(baseline["max_tilt_deg"]) / max(0.1, float(metrics["max_tilt_deg"])) - 1.0,
        "grounded": float(metrics["grounded_fraction"]) / float(baseline["grounded_fraction"]) - 1.0,
        "stop": float(baseline["stop_time_s"]) / float(metrics["stop_time_s"]) - 1.0,
    }


def score_metrics(metrics: dict[str, float | bool], baseline: dict[str, float | bool]) -> tuple[float, dict[str, float], bool]:
    improvements = ratios(metrics, baseline)
    weights = {
        "finish": 4.0,
        "sustained": 3.0,
        "top": 1.5,
        "acceleration": 1.0,
        "drift": 0.75,
        "heading": 0.75,
        "tilt": 0.5,
        "grounded": 0.5,
        "stop": 0.25,
    }
    regressions = [max(0.0, -value) for value in improvements.values()]
    fitness = sum(weights[name] * improvements[name] for name in weights)
    fitness -= 5.0 * sum(regressions) + 50.0 * sum(value * value for value in regressions)
    if not metrics["finished_100ft"]:
        fitness -= 100.0
    strict = (
        metrics["wheel_frictionloss"] == 0.003
        and metrics["current_limit_a"] == 1.75
        and bool(metrics["finished_100ft"])
        and all(value >= 0.0 for value in improvements.values())
        and (
        improvements["finish"] > 0.0 and improvements["sustained"] > 0.0
        )
    )
    return fitness, improvements, strict


def candidate_id(values: np.ndarray) -> str:
    payload = ",".join(f"{value:.6f}" for value in values)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def evaluate_candidate(
    values: np.ndarray,
    *,
    incumbent: Path,
    specialist: Path,
    brake: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict]:
    identifier = candidate_id(values)
    drive_path = output_dir / "drive" / f"{identifier}.onnx"
    policy_path = output_dir / "policies" / f"{identifier}.onnx"
    evaluation_path = output_dir / "evaluations" / f"{identifier}.json"
    log_path = output_dir / "logs" / f"{identifier}.log"
    if not drive_path.exists():
        build_joint_fusion(
            incumbent,
            specialist,
            drive_path,
            steering_authority=0.25,
            propulsion_authority=1.05,
            head_authority=0.0,
            hip_yaw_authority=float(values[0]),
            hip_roll_authority=float(values[1]),
            hip_pitch_authority=float(values[2]),
            knee_authority=float(values[3]),
            ankle_authority=float(values[4]),
            speed_command_threshold=0.5,
            smooth_turn_start=0.08,
            smooth_turn_end=0.25,
        )
    if not policy_path.exists():
        build_brake_safe_policy(
            drive_path,
            brake,
            policy_path,
            zero_command_threshold=0.02,
            joint_velocity_threshold=0.20,
            gate_mode="joint_velocity",
        )
    if not evaluation_path.exists():
        command = [
            sys.executable,
            str(TOOLS / "evaluate_swizzle.py"),
            str(policy_path),
            "--profile", "race-5mph",
            "--current-limit", "1.75",
            "--wheel-friction", "0.003",
            "--line-hold",
            "--line-yaw-kp", "0.70",
            "--line-lateral-kp", "0.22",
            "--line-yaw-kd", "0.07",
            "--line-max-wz", "0.15",
            "--output", str(evaluation_path),
        ]
        with log_path.open("w") as log:
            subprocess.run(command, cwd=UPSTREAM, stdout=log, stderr=subprocess.STDOUT, check=True)
    return policy_path, evaluation_path, load_json(evaluation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=7)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6801)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAB_ROOT / "reports" / "duckwing-v68-cem",
    )
    args = parser.parse_args()
    if not 2 <= args.elite < args.population:
        raise SystemExit("elite must be at least 2 and smaller than population")

    incumbent = LAB_ROOT / "releases" / "v66" / "duckwing-v66-v65-control-fusion.onnx"
    specialist = LAB_ROOT / "incoming" / "rtx5090" / "v47-official-friction-speed-specialist" / "policy.onnx"
    brake = LAB_ROOT / "incoming" / "rtx5090" / "v65-v63-immediate-switch-2026-09-01" / "policy.onnx"
    baseline_path = LAB_ROOT / "releases" / "v67" / "leader-metrics.json"
    for path in (incumbent, specialist, brake, baseline_path):
        if not path.is_file():
            raise SystemExit(f"Required input missing: {path}")

    output_dir = args.output_dir.resolve()
    for child in ("drive", "policies", "evaluations", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    baseline = load_json(baseline_path)
    rng = np.random.default_rng(args.seed)
    mean = np.asarray([0.25, 0.25, 1.05, 1.05, 1.05], dtype=np.float64)
    sigma = np.asarray([0.055, 0.055, 0.045, 0.045, 0.045], dtype=np.float64)
    lower = np.asarray([0.05, 0.05, 0.85, 0.85, 0.85], dtype=np.float64)
    upper = np.asarray([0.55, 0.55, 1.25, 1.25, 1.25], dtype=np.float64)
    all_records: list[dict] = []

    for generation in range(args.generations):
        samples = rng.normal(mean, sigma, size=(args.population, len(mean)))
        samples = np.clip(samples, lower, upper)
        samples[0] = mean
        if generation == 0:
            samples[1] = np.asarray([0.25, 0.25, 1.05, 1.05, 1.05])
        generation_records = []
        for index, values in enumerate(samples):
            policy_path, evaluation_path, evaluation = evaluate_candidate(
                values,
                incumbent=incumbent,
                specialist=specialist,
                brake=brake,
                output_dir=output_dir,
            )
            metrics = extract_metrics(evaluation)
            fitness, improvements, strict = score_metrics(metrics, baseline)
            record = {
                "generation": generation,
                "index": index,
                "candidate_id": candidate_id(values),
                "parameters": dict(zip(PARAMETER_NAMES, (float(value) for value in values), strict=True)),
                "policy": str(policy_path),
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
        sigma = np.maximum(0.004, 0.25 * sigma + 0.75 * elite_values.std(axis=0))
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
        f"V68 CEM complete: {len(strict_records)} strict candidates; "
        f"scorecard={output_dir / 'scorecard.json'}"
    )


if __name__ == "__main__":
    main()
