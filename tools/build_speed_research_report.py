#!/usr/bin/env python3
"""Build a compact, evidence-only MicroDuck speed research scoreboard."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "speed-research-scoreboard.json"
OUT_MD = REPORTS / "speed-research-scoreboard.md"


def summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if "best" in data:
        best = data["best"]
        return {
            "label": path.parent.name,
            "source": str(path),
            "world_x_mps": best.get("world_x_sustained_mps"),
            "body_mps": best.get("sustained_mean_mps"),
            "survival": best.get("survival_fraction"),
            "lateral_m": best.get("max_lateral_deviation_m"),
            "heading_deg": best.get("mean_abs_heading_deg"),
            "peak_mps": best.get("peak_mps"),
        }
    values = data.get("summary", data.get("checkpoint_rank", {}))
    return {
        "label": path.stem,
        "source": str(path),
        "world_x_mps": values.get("sustained_world_x_speed_mps", values.get("world_x_sustained_mps")),
        "body_mps": values.get("sustained_mean_forward_speed_mps", values.get("primary_sustained_mean_mps")),
        "survival": values.get("survival_fraction", values.get("secondary_survival_fraction")),
        "lateral_m": values.get("mean_max_abs_lateral_deviation_m"),
        "heading_deg": values.get("mean_abs_heading_deg"),
        "peak_mps": values.get("maximum_forward_speed_mps", values.get("tertiary_peak_mps")),
    }


def scout_transfer_audit() -> dict[str, object]:
    log = REPORTS / "train-ducklab-speed-scout-transfer-e4096-i4000-s541.log"
    if not log.is_file():
        return {"available": False}
    text = log.read_text(errors="replace")
    stages = [int(value) for value in re.findall(r"Curriculum/speed_scout_friction_transfer/stage: ([0-9]+)", text)]
    frictions = [float(value) for value in re.findall(r"wheel_frictionloss: ([0-9.]+)", text)]
    speeds = [float(value) for value in re.findall(r"world_forward_velocity_mps: ([0-9.]+)", text)]
    return {
        "available": True,
        "last_stage": stages[-1] if stages else None,
        "highest_stage": max(stages) if stages else None,
        "last_friction": frictions[-1] if frictions else None,
        "last_training_world_x_mps": speeds[-1] if speeds else None,
        "interpretation": (
            "Prior transfer plateaued before official 0.003 friction; use low-plasticity "
            "gating rather than another direct full-drag fine-tune."
        ),
    }


def main() -> None:
    source_paths = [
        REPORTS / "speed-retention-v3-final-official.json",
        REPORTS / "speed-scout-controller-baseline.json",
        REPORTS / "speed-scout-transfer-current-friction-0025.json",
        REPORTS / "speed-scout-transfer-i160-linehold-frictionless.json",
    ]
    source_paths.extend(sorted(REPORTS.glob("official-speed-sweep-*/*/best_speed_discovery.json")))
    rows = [summary(path) for path in source_paths if path.is_file()]
    rows.sort(key=lambda row: (row["world_x_mps"] is not None, row["world_x_mps"] or -1), reverse=True)
    report = {"scoreboard": rows, "historical_audit": scout_transfer_audit()}
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# MicroDuck speed research scoreboard", "", "| Policy | World-X m/s | Body m/s | Survival | Lateral m | Heading ° |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| {label} | {world_x_mps!s} | {body_mps!s} | {survival!s} | {lateral_m!s} | {heading_deg!s} |".format(**row)
        )
    audit = report["historical_audit"]
    lines.extend(["", "## Historical transfer audit", "", json.dumps(audit, indent=2, sort_keys=True)])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)


if __name__ == "__main__":
    main()
