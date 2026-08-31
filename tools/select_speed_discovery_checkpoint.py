#!/usr/bin/env python3
"""Export, evaluate, and preserve the best speed-discovery checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
UV = LAB_ROOT / ".tools" / "uv" / "bin" / "uv"
EXPORT = UPSTREAM / "scripts" / "export.py"
EVALUATE = LAB_ROOT / "tools" / "evaluate_speed_discovery.py"
TASK = "Mjlab-SpeedDiscovery-Flat-MicroDuck-Rollers"


def iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def run_checked(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed ({result.returncode}):\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--command-mps", type=float, default=0.8)
    parser.add_argument("--wheel-friction", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Re-export and re-evaluate cached candidates.")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--race-line-control", action="store_true")
    parser.add_argument("--yaw-kp", type=float, default=0.55)
    parser.add_argument("--lateral-kp", type=float, default=0.10)
    parser.add_argument("--yaw-kd", type=float, default=0.08)
    parser.add_argument("--max-correction", type=float, default=0.18)
    parser.add_argument("--rank-world-x", action="store_true")
    parser.add_argument("--min-body-sustained-mps", type=float, default=0.0)
    parser.add_argument("--min-survival", type=float, default=0.0)
    parser.add_argument("--max-lateral-deviation-m", type=float, default=float("inf"))
    parser.add_argument("--max-mean-heading-deg", type=float, default=float("inf"))
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help="Evaluate every Nth saved checkpoint (1 evaluates every saved file).",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=iteration)
    checkpoints = checkpoints[:: max(args.checkpoint_stride, 1)]
    if not checkpoints:
        raise SystemExit(f"No model_*.pt checkpoints in {run_dir}")

    evaluation_dir = run_dir / "speed_discovery_evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        step = iteration(checkpoint)
        onnx_path = evaluation_dir / f"model_{step}.onnx"
        report_path = evaluation_dir / f"model_{step}.json"
        if args.force or not onnx_path.is_file():
            run_checked(
                [
                    str(UV),
                    "run",
                    "python",
                    str(EXPORT),
                    args.task,
                    "--checkpoint-file",
                    str(checkpoint),
                    "--onnx-file",
                    str(onnx_path),
                    "--num-envs",
                    "1",
                ],
                UPSTREAM,
            )
        report_matches_protocol = False
        if report_path.is_file() and not args.force:
            try:
                cached = json.loads(report_path.read_text(encoding="utf-8"))
                report_matches_protocol = (
                    int(cached.get("episodes", -1)) == args.episodes
                    and float(cached.get("duration_s", -1.0)) == args.duration
                    and float(cached.get("command_mps", -1.0)) == args.command_mps
                    and float(cached.get("wheel_frictionloss", -1.0))
                    == args.wheel_friction
                    and bool(cached.get("race_line_control", False))
                    == args.race_line_control
                    and float(cached.get("line_hold", {}).get("yaw_kp", 0.55))
                    == args.yaw_kp
                    and float(cached.get("line_hold", {}).get("lateral_kp", 0.10))
                    == args.lateral_kp
                    and float(cached.get("line_hold", {}).get("yaw_kd", 0.08))
                    == args.yaw_kd
                    and float(cached.get("line_hold", {}).get("max_correction", 0.18))
                    == args.max_correction
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                report_matches_protocol = False
        if args.force or not report_matches_protocol:
            evaluate_command = [
                    str(UV),
                    "run",
                    "python",
                    str(EVALUATE),
                    str(onnx_path),
                    "--output",
                    str(report_path),
                    "--episodes",
                    str(args.episodes),
                    "--duration",
                    str(args.duration),
                    "--command-mps",
                    str(args.command_mps),
                    "--wheel-friction",
                    str(args.wheel_friction),
                ]
            if args.race_line_control:
                evaluate_command.extend(
                    [
                        "--race-line-control",
                        "--yaw-kp", str(args.yaw_kp),
                        "--lateral-kp", str(args.lateral_kp),
                        "--yaw-kd", str(args.yaw_kd),
                        "--max-correction", str(args.max_correction),
                    ]
                )
            run_checked(evaluate_command, UPSTREAM)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        row = {
            "iteration": step,
            "checkpoint": str(checkpoint),
            "policy": str(onnx_path),
            "evaluation": str(report_path),
            "sustained_mean_mps": float(summary["sustained_mean_forward_speed_mps"]),
            "sustained_mean_mph": float(summary["sustained_mean_forward_speed_mph"]),
            "survival_fraction": float(summary["survival_fraction"]),
            "peak_mps": float(summary["maximum_forward_speed_mps"]),
            "peak_mph": float(summary["maximum_forward_speed_mph"]),
            "best_1s_mps": float(summary["best_1s_forward_speed_mps"]),
            "best_1s_mph": float(summary["best_1s_forward_speed_mph"]),
            "world_x_sustained_mps": float(summary["sustained_world_x_speed_mps"]),
            "world_x_sustained_mph": float(summary["sustained_world_x_speed_mph"]),
            "max_lateral_deviation_m": float(
                summary["mean_max_abs_lateral_deviation_m"]
            ),
            "mean_abs_heading_deg": float(summary["mean_abs_heading_deg"]),
        }
        row["qualified"] = bool(
            row["sustained_mean_mps"] >= args.min_body_sustained_mps
            and row["survival_fraction"] >= args.min_survival
            and row["max_lateral_deviation_m"] <= args.max_lateral_deviation_m
            and row["mean_abs_heading_deg"] <= args.max_mean_heading_deg
        )
        rows.append(row)
        print(
            f"i{step}: mean={row['sustained_mean_mps']:.3f} m/s "
            f"({row['sustained_mean_mph']:.2f} mph), "
            f"survival={100.0 * row['survival_fraction']:.1f}%, "
            f"peak={row['peak_mps']:.3f} m/s, "
            f"world-x={row['world_x_sustained_mps']:.3f} m/s, "
            f"qualified={row['qualified']}",
            flush=True,
        )

    qualified_rows = [row for row in rows if row["qualified"]]
    selection_pool = qualified_rows or rows
    if args.rank_world_x:
        best = max(
            selection_pool,
            key=lambda row: (
                row["world_x_sustained_mps"],
                row["sustained_mean_mps"],
                row["survival_fraction"],
                row["peak_mps"],
            ),
        )
    else:
        best = max(
            selection_pool,
            key=lambda row: (
                row["sustained_mean_mps"],
                row["survival_fraction"],
                row["peak_mps"],
            ),
        )
    best_checkpoint = run_dir / "best_speed_discovery.pt"
    best_policy = run_dir / "best_speed_discovery.onnx"
    shutil.copy2(Path(str(best["checkpoint"])), best_checkpoint)
    shutil.copy2(Path(str(best["policy"])), best_policy)
    manifest = {
        "selection_order": (
            [
                "world_x_sustained_velocity",
                "body_sustained_velocity",
                "survival_fraction",
                "peak_forward_velocity",
            ]
            if args.rank_world_x
            else [
                "sustained_mean_forward_velocity",
                "survival_fraction",
                "peak_forward_velocity",
            ]
        ),
        "qualification": {
            "min_body_sustained_mps": args.min_body_sustained_mps,
            "min_survival": args.min_survival,
            "max_lateral_deviation_m": args.max_lateral_deviation_m,
            "max_mean_heading_deg": args.max_mean_heading_deg,
            "any_qualified": bool(qualified_rows),
        },
        "evaluation_protocol": {
            "episodes": args.episodes,
            "duration_s": args.duration,
            "command_mps": args.command_mps,
            "wheel_frictionloss": args.wheel_friction,
            "race_line_control": args.race_line_control,
            "line_hold": {
                "yaw_kp": args.yaw_kp,
                "lateral_kp": args.lateral_kp,
                "yaw_kd": args.yaw_kd,
                "max_correction": args.max_correction,
            },
        },
        "best": best,
        "best_checkpoint": str(best_checkpoint),
        "best_policy": str(best_policy),
        "candidates": rows,
    }
    manifest_path = run_dir / "best_speed_discovery.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Best speed-discovery checkpoint: i{best['iteration']}")
    print(manifest_path)


if __name__ == "__main__":
    main()
