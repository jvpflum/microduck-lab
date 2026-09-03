#!/usr/bin/env python3
"""Re-rank an existing residual sweep on clean and randomized starts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
EVALUATOR = LAB_ROOT / "tools" / "evaluate_swizzle.py"


def evaluate_clean(policy: Path, output: Path) -> dict:
    command = [
        sys.executable, str(EVALUATOR), str(policy),
        "--profile", "race-screen",
        "--current-limit", "1.75",
        "--wheel-friction", "0.003",
        "--line-hold",
        "--line-yaw-kp", "0.70",
        "--line-lateral-kp", "0.22",
        "--line-yaw-kd", "0.07",
        "--line-max-wz", "0.15",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=UPSTREAM, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorecard", type=Path,
        default=LAB_ROOT / "reports/v76-v67-dynamic-residual-search/scorecard.json",
    )
    parser.add_argument(
        "--baseline-noisy", type=Path,
        default=LAB_ROOT / "reports/v80-baseline-noisy-3x12.json",
    )
    parser.add_argument(
        "--baseline-clean", type=Path,
        default=LAB_ROOT / "reports/v78-screen-baseline.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=LAB_ROOT / "reports/v80-v67-composite-rank",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    candidates = json.loads(args.scorecard.read_text(encoding="utf-8"))
    noisy_base = json.loads(args.baseline_noisy.read_text(encoding="utf-8"))["summary"]
    clean_base = json.loads(args.baseline_clean.read_text(encoding="utf-8"))["phases"]["max_speed"]
    noisy_base_mph = float(noisy_base["sustained_mean_forward_speed_mph"])
    clean_base_mph = float(clean_base["mean_world_forward_speed_mph"])
    clean_base_peak = float(clean_base["peak_world_forward_speed_mph"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def task(row: dict) -> tuple[dict, dict]:
        policy = Path(row["policy"])
        output = output_dir / f"candidate-{int(row['index']):03d}.json"
        result = evaluate_clean(policy, output)
        return row, result["phases"]["max_speed"]

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(task, row) for row in candidates]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, clean = future.result()
            noisy_mph = float(row["sustained_mph"])
            clean_mph = float(clean["mean_world_forward_speed_mph"])
            peak_mph = float(clean["peak_world_forward_speed_mph"])
            noisy_gain = noisy_mph / noisy_base_mph - 1.0
            clean_gain = clean_mph / clean_base_mph - 1.0
            peak_gain = peak_mph / clean_base_peak - 1.0
            survival = float(row["survival"])
            tilt = float(clean["tilt_max_deg"])
            hard_penalty = 2.0 * (1.0 - survival) + max(0.0, tilt - 18.0) * 0.05
            composite = 3.0 * min(noisy_gain, clean_gain) + 0.5 * (noisy_gain + clean_gain) + 0.15 * peak_gain - hard_penalty
            results.append({
                **row,
                "noisy_gain_fraction": noisy_gain,
                "clean_mph": clean_mph,
                "clean_gain_fraction": clean_gain,
                "clean_peak_mph": peak_mph,
                "clean_tilt_deg": tilt,
                "composite_score": composite,
                "beats_both_sustained": survival == 1.0 and noisy_gain > 0.0 and clean_gain > 0.0,
            })
            print(f"[{completed:03d}/{len(candidates):03d}]", flush=True)

    results.sort(key=lambda row: row["composite_score"], reverse=True)
    (output_dir / "scorecard.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pareto = [row for row in results if row["beats_both_sustained"]]
    (output_dir / "beats-both.json").write_text(json.dumps(pareto, indent=2) + "\n", encoding="utf-8")
    print("baseline=" + json.dumps({"noisy_mph": noisy_base_mph, "clean_mph": clean_base_mph, "clean_peak_mph": clean_base_peak}))
    print(f"beats_both={len(pareto)}")
    print("best=" + json.dumps(results[0]))


if __name__ == "__main__":
    main()
