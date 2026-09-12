#!/usr/bin/env python3
"""Direct-evaluator local search for a small V67 propulsion residual."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
BUILD = LAB_ROOT / "tools" / "build_dynamic_residual_policy.py"
EVAL = LAB_ROOT / "tools" / "evaluate_speed_discovery.py"


def evaluate(policy: Path, output: Path, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(EVAL),
        str(policy),
        "--episodes", str(args.episodes),
        "--duration", str(args.duration),
        "--command-mps", "0.8",
        "--current-limit", "1.75",
        "--wheel-friction", "0.003",
        "--seed", str(args.seed),
        "--race-line-control",
        "--yaw-kp", "0.70",
        "--lateral-kp", "0.22",
        "--yaw-kd", "0.07",
        "--max-correction", "0.15",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=UPSTREAM, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=LAB_ROOT / "releases/v67/duckwing-v67-joint-specialist-fusion.onnx")
    parser.add_argument("--output-dir", type=Path, default=LAB_ROOT / "reports/v76-v67-dynamic-residual-search")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=8800)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    values = {
        "pos": (-0.10, -0.075, -0.05, -0.025, 0.0),
        "vel": (-0.02, -0.015, -0.01, -0.005, 0.0),
        "last": (-0.16, -0.13, -0.10, -0.07, -0.04),
    }
    records = []
    for index, (pos, vel, last) in enumerate(itertools.product(values["pos"], values["vel"], values["last"])):
        stem = f"p{pos:+.3f}_v{vel:+.3f}_l{last:+.3f}".replace("+", "plus").replace("-", "minus").replace(".", "d")
        policy = output_dir / f"{stem}.onnx"
        result_path = output_dir / f"{stem}.json"
        subprocess.run(
            [sys.executable, str(BUILD), str(source), str(policy), "--pos-gain", str(pos), "--vel-gain", str(vel), "--last-gain", str(last)],
            cwd=UPSTREAM, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        result = evaluate(policy, result_path, args)
        summary = result["summary"]
        survival = float(summary["survival_fraction"])
        speed = float(summary["sustained_mean_forward_speed_mps"])
        heading = float(summary["mean_abs_heading_deg"])
        lateral = float(summary["mean_max_abs_lateral_deviation_m"])
        score = speed - 0.015 * heading - 0.04 * lateral - 1.5 * (1.0 - survival)
        records.append({
            "index": index,
            "pos_gain": pos,
            "vel_gain": vel,
            "last_gain": last,
            "score": score,
            "sustained_mps": speed,
            "sustained_mph": float(summary["sustained_mean_forward_speed_mph"]),
            "survival": survival,
            "heading_deg": heading,
            "lateral_m": lateral,
            "policy": str(policy),
        })
        print(f"[{index + 1:03d}/125] score={score:+.4f} speed={speed * 2.2369362921:.3f}mph survival={survival:.2f} {pos:+.3f}/{vel:+.3f}/{last:+.3f}", flush=True)

    records.sort(key=lambda item: item["score"], reverse=True)
    (output_dir / "scorecard.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("best=" + json.dumps(records[0]))
    print("policy=" + records[0]["policy"])


if __name__ == "__main__":
    main()
