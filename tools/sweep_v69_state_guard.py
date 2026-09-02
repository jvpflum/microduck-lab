#!/usr/bin/env python3
"""Search a body-yaw state guard around the best rejected V68 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
TOOLS = LAB_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from build_brake_safe_policy import build_brake_safe_policy  # noqa: E402
from build_joint_fusion_policy import build_joint_fusion  # noqa: E402
from build_state_guarded_policy import build_state_guarded_policy  # noqa: E402
from optimize_v68_joint_fusion import extract_metrics, load_json, score_metrics  # noqa: E402


V68_AUTHORITIES = (0.2473519900, 0.3476408048, 1.0503262431, 1.0399181835, 1.1813245881)
DEFAULT_STARTS = (0.0, 0.15, 0.30, 0.50)
DEFAULT_ENDS = (0.40, 0.70, 1.10, 1.60)
DEFAULT_AUTHORITIES = (0.01, 0.025, 0.05, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)


def _candidate_id(start: float, end: float, authority: float) -> str:
    payload = f"{start:.6f},{end:.6f},{authority:.6f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def _build_v68_challenger(output_dir: Path) -> Path:
    incumbent = LAB_ROOT / "releases" / "v66" / "duckwing-v66-v65-control-fusion.onnx"
    specialist = LAB_ROOT / "incoming" / "rtx5090" / "v47-official-friction-speed-specialist" / "policy.onnx"
    brake = LAB_ROOT / "incoming" / "rtx5090" / "v65-v63-immediate-switch-2026-09-01" / "policy.onnx"
    for path in (incumbent, specialist, brake):
        if not path.is_file():
            raise SystemExit(f"Required public input is missing: {path}")
    drive_path = output_dir / "v68-challenger-drive.onnx"
    policy_path = output_dir / "v68-challenger-policy.onnx"
    if not drive_path.exists():
        build_joint_fusion(
            incumbent,
            specialist,
            drive_path,
            steering_authority=0.25,
            propulsion_authority=1.05,
            head_authority=0.0,
            hip_yaw_authority=V68_AUTHORITIES[0],
            hip_roll_authority=V68_AUTHORITIES[1],
            hip_pitch_authority=V68_AUTHORITIES[2],
            knee_authority=V68_AUTHORITIES[3],
            ankle_authority=V68_AUTHORITIES[4],
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
    return policy_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=_float_list, default=DEFAULT_STARTS)
    parser.add_argument("--ends", type=_float_list, default=DEFAULT_ENDS)
    parser.add_argument("--authorities", type=_float_list, default=DEFAULT_AUTHORITIES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAB_ROOT / "reports" / "duckwing-v69-state-guard",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = LAB_ROOT / output_dir
    output_dir = output_dir.resolve()
    for child in ("policies", "evaluations", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    baseline_path = LAB_ROOT / "releases" / "v67" / "leader-metrics.json"
    baseline_policy = LAB_ROOT / "releases" / "v67" / "duckwing-v67-joint-specialist-fusion.onnx"
    for path in (baseline_path, baseline_policy):
        if not path.is_file():
            raise SystemExit(f"Required V67 release input is missing: {path}")
    baseline = load_json(baseline_path)
    challenger = _build_v68_challenger(output_dir)

    configurations = [
        values
        for values in product(args.starts, args.ends, args.authorities)
        if 0.0 <= values[0] < values[1]
    ]
    records: list[dict] = []
    for index, (start, end, authority) in enumerate(configurations, start=1):
        identifier = _candidate_id(start, end, authority)
        policy_path = output_dir / "policies" / f"{identifier}.onnx"
        evaluation_path = output_dir / "evaluations" / f"{identifier}.json"
        log_path = output_dir / "logs" / f"{identifier}.log"
        if not policy_path.exists():
            build_state_guarded_policy(
                baseline_policy,
                challenger,
                policy_path,
                yaw_start=start,
                yaw_end=end,
                candidate_authority=authority,
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
        metrics = extract_metrics(load_json(evaluation_path))
        fitness, improvements, strict = score_metrics(metrics, baseline)
        record = {
            "candidate_id": identifier,
            "parameters": {
                "yaw_start_rad_s": start,
                "yaw_end_rad_s": end,
                "candidate_authority": authority,
            },
            "policy": str(policy_path.relative_to(LAB_ROOT)),
            "evaluation": str(evaluation_path.relative_to(LAB_ROOT)),
            "fitness": fitness,
            "strict_promotable": strict,
            "metrics": metrics,
            "improvement_ratios": improvements,
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(configurations):02d}] {identifier} "
            f"fitness={fitness:+.5f} strict={strict}",
            flush=True,
        )
        ranked = sorted(records, key=lambda item: item["fitness"], reverse=True)
        (output_dir / "progress.json").write_text(json.dumps(ranked, indent=2) + "\n")

    records.sort(key=lambda item: item["fitness"], reverse=True)
    strict_records = [record for record in records if record["strict_promotable"]]
    (output_dir / "scorecard.json").write_text(json.dumps(records, indent=2) + "\n")
    (output_dir / "promotion-candidates.json").write_text(
        json.dumps(strict_records, indent=2) + "\n"
    )
    print(
        f"V69 state-guard search complete: {len(strict_records)} strict candidates; "
        f"scorecard={output_dir / 'scorecard.json'}"
    )


if __name__ == "__main__":
    main()
