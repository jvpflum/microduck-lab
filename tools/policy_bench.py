#!/usr/bin/env python3
"""Offline-first experiment registry and policy promotion workflow."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from capability_catalog import task_has_evaluator, task_spec
from agent_runs import load_agent_runs


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
POLLEN_RUNTIME = LAB_ROOT / "upstream" / "microduck"
POLLEN_SIMULATOR = LAB_ROOT / "upstream" / "microduck-simulator"
DEFAULT_STATE = LAB_ROOT / "policy-bench"
DASHBOARD_PORT = int(os.environ.get("DUCKLAB_BENCH_PORT", "8091"))
FACTORY_ARENA_URL = f"http://localhost:{DASHBOARD_PORT}/factory/?boot=1"
SCHEMA_VERSION = 1
STAGES = ("experimental", "evaluated", "sim-qualified", "hardware-candidate", "production")
SPRINT_BASELINE_REPORT = LAB_ROOT / "reports" / "sprint-pollen-factory-baseline.json"
SPRINT_SPEED_PHASES = ("speed_030", "speed_040", "speed_050", "speed_055")
RACE_POLLEN_BASELINE_REPORT = LAB_ROOT / "reports" / "race-pollen-factory-baseline.json"
RACE_TRAINED_INCUMBENT_REPORT = LAB_ROOT / "reports" / "race-sprint-v3-baseline.json"
RACE5_POLLEN_BASELINE_REPORT = LAB_ROOT / "reports" / "race5-pollen-factory-baseline.json"
RACE5_POLLEN_LINE_BASELINE_REPORT = (
    LAB_ROOT / "reports" / "race5-pollen-factory-line-baseline.json"
)
RACE5_TRAINED_INCUMBENT_REPORT = LAB_ROOT / "reports" / "race5-sprint-v3-incumbent.json"


def bounded_score(value: float, target: float, tolerance: float) -> float:
    return max(0.0, min(1.0, 1.0 - abs(value - target) / tolerance))


def sprint_steady_speed(phase: dict[str, Any]) -> float:
    return float(phase.get("steady_mean_forward_speed_mps", phase.get("mean_forward_speed_mps", 0.0)))


def race5_race_rank(score: dict[str, Any]) -> tuple[float, ...]:
    """Rank a qualified Race5 policy like a drag racer: A-to-B first."""
    performance = score.get("performance", {})
    finished = bool(performance.get("finished_100ft"))
    elapsed = performance.get("elapsed_time_100ft_s")
    elapsed_value = float(elapsed) if finished and elapsed is not None else float("inf")
    return (
        float(finished),
        -elapsed_value,
        float(performance.get("sustained_speed_mph", 0.0)),
        float(performance.get("trap_speed_100ft_mph") or 0.0),
        float(performance.get("top_speed_mph", 0.0)),
        -float(performance.get("long_run_max_drift_ft", float("inf"))),
        -float(performance.get("long_run_max_heading_error_deg", float("inf"))),
        float(performance.get("agility_score", 0.0)),
    )


def race5_advances_incumbent(
    candidate: dict[str, Any], incumbent: dict[str, Any]
) -> bool:
    """Require a faster heat without giving back straightness or basic skills."""
    candidate_performance = candidate.get("performance", {})
    incumbent_performance = incumbent.get("performance", {})
    required = (
        "elapsed_time_100ft_s",
        "long_run_max_drift_ft",
        "long_run_max_heading_error_deg",
        "agility_score",
        "auto_steering_percent",
    )
    if not candidate_performance.get("finished_100ft") or not all(
        candidate_performance.get(name) is not None
        and incumbent_performance.get(name) is not None
        for name in required
    ):
        return False
    return (
        float(candidate_performance["elapsed_time_100ft_s"])
        < float(incumbent_performance["elapsed_time_100ft_s"])
        and float(candidate_performance["long_run_max_drift_ft"])
        <= float(incumbent_performance["long_run_max_drift_ft"])
        and float(candidate_performance["long_run_max_heading_error_deg"])
        <= float(incumbent_performance["long_run_max_heading_error_deg"])
        and float(candidate_performance["agility_score"])
        >= float(incumbent_performance["agility_score"]) - 3.0
        and float(candidate_performance["auto_steering_percent"])
        <= float(incumbent_performance["auto_steering_percent"]) + 0.25
    )


def render_test_roster(
    registry: dict[str, Any],
    manifests: list[dict[str, Any]],
    task: str,
) -> str:
    """Render an operator-selected three-model comparison roster.

    The roster is deliberately separate from the verified-physics ranking: it
    may include a research replay with different physics, but every card must
    point at an immutable, hash-verified ONNX snapshot before Play is enabled.
    """
    configured = registry.get("tasks", {}).get(task, {}).get("test-roster", [])
    if not isinstance(configured, list):
        return ""
    by_id = {manifest.get("run_id"): manifest for manifest in manifests}
    cards = []
    medals = ("🥇", "🥈", "🥉")
    for index, configured_entry in enumerate(configured[:3]):
        entry = (
            {"run_id": configured_entry}
            if isinstance(configured_entry, str)
            else configured_entry
        )
        if not isinstance(entry, dict):
            continue
        run_id = str(entry.get("run_id", ""))
        manifest = by_id.get(run_id)
        if manifest is None or manifest.get("task") != task:
            continue
        policy = manifest.get("artifacts", {}).get("policy")
        policy_path = Path(policy.get("path", "")) if isinstance(policy, dict) else None
        playable = bool(
            policy_path
            and policy_path.is_file()
            and policy.get("sha256")
            and sha256(policy_path) == policy["sha256"]
        )
        label = str(
            entry.get("label")
            or display_experiment_label(
                task,
                str(manifest.get("experiment_label") or Path(manifest["source_run_dir"]).name),
            )
        )
        role = str(entry.get("role") or f"Comparison model {index + 1}")
        physics = str(entry.get("physics") or "Physics profile not recorded")
        sustained = entry.get("sustained_mph")
        top = entry.get("top_mph")
        speed_metric = str(entry.get("speed_metric") or "sustained")
        speed_copy = "Recorded speed unavailable"
        try:
            if sustained is not None and top is not None:
                speed_copy = (
                    f"{float(sustained):.2f} mph {speed_metric} · {float(top):.2f} mph peak"
                )
        except (TypeError, ValueError):
            pass
        note = str(entry.get("note") or "Immutable Policy Bench snapshot")
        actions = (
            f"<button class='play primary-action' data-run-id='{html.escape(run_id)}' "
            f"data-label='Try {html.escape(label)}'>Try in arena</button>"
            if playable
            else "<button class='secondary' disabled>Policy hash unavailable</button>"
        )
        actions += (
            f"<a class='text-action' href='runs/{html.escape(run_id)}/report.html'>View details</a>"
        )
        cards.append(
            f"<article class='podium-card {'winner' if index == 0 else ''}'>"
            f"<div class='podium-rank'>{medals[index]} TEST #{index + 1}</div>"
            f"<span class='podium-status {'good' if playable else 'bad'}'>{html.escape(role)}</span>"
            f"<h3>{html.escape(label)}</h3>"
            f"<div class='podium-score'><strong>{html.escape(str(entry.get('headline', 'READY' if playable else 'BLOCKED')))}</strong>"
            "<span>comparison slot</span></div>"
            f"<p class='podium-delta'>{html.escape(physics)}</p>"
            f"<p class='podium-outcome'>{html.escape(speed_copy)} · {html.escape(note)}</p>"
            f"<div class='podium-actions'>{actions}</div></article>"
        )
    if not cards:
        return ""
    return (
        "<div class='test-roster-heading'><p class='eyebrow'>PINNED TEST ROSTER</p>"
        "<p class='section-note'>Comparison slots are operator-selected. Physics labels are authoritative; "
        "the frictionless replay is not an official-friction record.</p></div>"
        "<div class='podium-grid test-roster-grid'>" + "".join(cards) + "</div>"
    )


def sprint_baseline_comparison(
    evaluation: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Build the operator-facing, direction-aware Sprint comparison."""
    candidate_phases = evaluation.get("phases", {})
    baseline_phases = baseline.get("phases", {})
    candidate_fast = candidate_phases.get("speed_055", {})
    baseline_fast = baseline_phases.get("speed_055", {})
    candidate_stop = candidate_phases.get("stop_055", {})
    baseline_stop = baseline_phases.get("stop_055", {})
    candidate_speed = sprint_steady_speed(candidate_fast)
    baseline_speed = sprint_steady_speed(baseline_fast)
    speed_delta = candidate_speed - baseline_speed
    speed_delta_percent = 100.0 * speed_delta / baseline_speed if baseline_speed else None

    def value(phase: dict[str, Any], key: str, default: float) -> float:
        return float(phase.get(key, default))

    candidate_lateral = value(candidate_fast, "steady_mean_abs_lateral_speed_mps", 1.0)
    baseline_lateral = value(baseline_fast, "steady_mean_abs_lateral_speed_mps", 1.0)
    candidate_tilt = value(candidate_fast, "tilt_max_deg", 90.0)
    baseline_tilt = value(baseline_fast, "tilt_max_deg", 90.0)
    candidate_stop_time = value(candidate_stop, "stop_time_below_0_05_mps_s", 99.0)
    baseline_stop_time = value(baseline_stop, "stop_time_below_0_05_mps_s", 99.0)
    command_rows = []
    for phase_name in SPRINT_SPEED_PHASES:
        candidate_phase = candidate_phases.get(phase_name, {})
        baseline_phase = baseline_phases.get(phase_name, {})
        command = float(candidate_phase.get("command_x_mps", baseline_phase.get("command_x_mps", 0.0)))
        candidate_value = sprint_steady_speed(candidate_phase)
        baseline_value = sprint_steady_speed(baseline_phase)
        command_rows.append(
            {
                "command_mps": command,
                "baseline_mps": round(baseline_value, 6),
                "candidate_mps": round(candidate_value, 6),
                "delta_mps": round(candidate_value - baseline_value, 6),
            }
        )
    comparison_checks = {
        "faster_at_0_55": speed_delta > 0.0,
        "lateral_not_worse": candidate_lateral <= baseline_lateral + 0.005,
        "tilt_not_worse": candidate_tilt <= baseline_tilt + 0.25,
        "stopping_not_worse": candidate_stop_time <= baseline_stop_time + 0.25,
        "faster_at_every_useful_command": all(row["delta_mps"] > 0.0 for row in command_rows),
    }
    improved = all(comparison_checks.values())
    return {
        "baseline_name": "Official Pollen roller",
        "baseline_report": str(SPRINT_BASELINE_REPORT),
        "test_command_mps": 0.55,
        "candidate_steady_speed_mps": round(candidate_speed, 6),
        "baseline_steady_speed_mps": round(baseline_speed, 6),
        "speed_delta_mps": round(speed_delta, 6),
        "speed_delta_percent": round(speed_delta_percent, 2) if speed_delta_percent is not None else None,
        "candidate_lateral_mps": round(candidate_lateral, 6),
        "baseline_lateral_mps": round(baseline_lateral, 6),
        "lateral_delta_mps": round(candidate_lateral - baseline_lateral, 6),
        "candidate_tilt_max_deg": round(candidate_tilt, 4),
        "baseline_tilt_max_deg": round(baseline_tilt, 4),
        "tilt_delta_deg": round(candidate_tilt - baseline_tilt, 4),
        "candidate_stop_time_s": round(candidate_stop_time, 4),
        "baseline_stop_time_s": round(baseline_stop_time, 4),
        "stop_time_delta_s": round(candidate_stop_time - baseline_stop_time, 4),
        "command_speeds": command_rows,
        "checks": comparison_checks,
        "improved": improved,
        "verdict": "Improved" if improved else "No clear improvement",
        "note": "One deterministic CPU MuJoCo replay per policy; hardware validation is still required.",
    }


MPS_TO_MPH = 2.2369362921


def race_baseline_comparison(
    evaluation: dict[str, Any], baseline: dict[str, Any], baseline_report: Path
) -> dict[str, Any]:
    candidate = evaluation.get("phases", {}).get("race", {})
    champion = baseline.get("phases", {}).get("race", {})
    candidate_speed = float(candidate.get(
        "steady_mean_world_forward_speed_mps",
        candidate.get("mean_world_forward_speed_mps", 0.0),
    ))
    baseline_speed = float(champion.get(
        "steady_mean_world_forward_speed_mps",
        champion.get("mean_world_forward_speed_mps", 0.0),
    ))
    candidate_long = evaluation.get("phases", {}).get("max_speed", {})
    baseline_long = baseline.get("phases", {}).get("max_speed", {})
    candidate_top = float(candidate_long.get(
        "verified_top_speed_0_5s_mps",
        candidate_long.get("peak_horizontal_speed_mps", 0.0),
    ))
    baseline_top = float(baseline_long.get(
        "verified_top_speed_0_5s_mps",
        baseline_long.get("peak_horizontal_speed_mps", 0.0),
    ))
    candidate_acceleration = float(candidate.get("acceleration_first_second_mps2", 0.0))
    baseline_acceleration = float(champion.get("acceleration_first_second_mps2", 0.0))
    candidate_finished = bool(candidate_long.get("finished_100ft"))
    baseline_finished = bool(baseline_long.get("finished_100ft"))
    candidate_et = candidate_long.get("finish_time_100ft_s")
    baseline_et = baseline_long.get("finish_time_100ft_s")
    a_to_b_improved = candidate_finished and (
        not baseline_finished or float(candidate_et) < float(baseline_et)
    )
    comparison_checks = {
        "a_to_b_faster": a_to_b_improved,
        "sustained_forward_speed_faster": candidate_speed > baseline_speed,
        "total_top_speed_faster": candidate_top > baseline_top,
        "first_second_acceleration_faster": candidate_acceleration > baseline_acceleration,
        "long_run_heading_not_worse": float(candidate_long.get("max_heading_error_deg", 3600.0))
        <= float(baseline_long.get("max_heading_error_deg", 3600.0)),
        "long_run_drift_not_worse": float(candidate_long.get("max_lateral_drift_ft", 999.0))
        <= float(baseline_long.get("max_lateral_drift_ft", 999.0)),
    }
    improved = all(comparison_checks.values())
    speed_improved = candidate_speed > baseline_speed and candidate_top > baseline_top
    return {
        "baseline_name": (
            "Official Pollen roller + same line hold"
            if baseline.get("line_hold", {}).get("enabled")
            else "Official Pollen roller (raw policy)"
        ),
        "baseline_report": str(baseline_report),
        "candidate_sustained_speed_mph": round(candidate_speed * MPS_TO_MPH, 3),
        "baseline_sustained_speed_mph": round(baseline_speed * MPS_TO_MPH, 3),
        "speed_delta_mph": round((candidate_speed - baseline_speed) * MPS_TO_MPH, 3),
        "candidate_total_top_speed_mph": round(candidate_top * MPS_TO_MPH, 3),
        "baseline_total_top_speed_mph": round(baseline_top * MPS_TO_MPH, 3),
        "candidate_acceleration_mph_s": round(candidate_acceleration * MPS_TO_MPH, 3),
        "baseline_acceleration_mph_s": round(baseline_acceleration * MPS_TO_MPH, 3),
        "candidate_finished_100ft": candidate_finished,
        "baseline_finished_100ft": baseline_finished,
        "candidate_elapsed_time_100ft_s": round(float(candidate_et), 3) if candidate_et is not None else None,
        "baseline_elapsed_time_100ft_s": round(float(baseline_et), 3) if baseline_et is not None else None,
        "candidate_trap_speed_mph": round(float(candidate_long.get("trap_speed_100ft_mph")), 3) if candidate_long.get("trap_speed_100ft_mph") is not None else None,
        "baseline_trap_speed_mph": round(float(baseline_long.get("trap_speed_100ft_mph")), 3) if baseline_long.get("trap_speed_100ft_mph") is not None else None,
        "candidate_long_run_drift_ft": round(float(candidate_long.get("max_lateral_drift_ft", 0.0)), 2),
        "baseline_long_run_drift_ft": round(float(baseline_long.get("max_lateral_drift_ft", 0.0)), 2),
        "candidate_long_run_heading_error_deg": round(float(candidate_long.get("max_heading_error_deg", 0.0)), 2),
        "baseline_long_run_heading_error_deg": round(float(baseline_long.get("max_heading_error_deg", 0.0)), 2),
        "candidate_yaw_change_deg": round(float(candidate.get("yaw_change_deg", 360.0)), 2),
        "baseline_yaw_change_deg": round(float(champion.get("yaw_change_deg", 360.0)), 2),
        "checks": comparison_checks,
        "speed_improved": speed_improved,
        "a_to_b_improved": a_to_b_improved,
        "improved": improved,
        "verdict": (
            "Beat Pollen from A to B; control qualification pending"
            if a_to_b_improved and not improved
            else "Beat Pollen from A to B"
            if improved
            else "Did not beat Pollen from A to B"
        ),
        "note": (
            "Deterministic CPU MuJoCo comparison against the immutable official Pollen policy. "
            "Both racers use the same measured line-hold controller when enabled; hardware validation is still required."
        ),
    }


def athletic_performance(
    evaluation: dict[str, Any], task: str, components: dict[str, float]
) -> dict[str, Any]:
    """Normalize operator-facing motion stats across skating evaluations."""
    phases = evaluation.get("phases", {})
    if task in {"race", "race5"}:
        phase = phases.get("race", {})
        duration = max(float(phase.get("duration_s", 0.0)), 1e-6)
        sustained_mps = float(
            phase.get(
                "steady_mean_world_forward_speed_mps",
                phase.get(
                    "mean_world_forward_speed_mps",
                    float(phase.get("forward_progress_m", 0.0)) / duration,
                ),
            )
        )
        top_phase = phases.get("max_speed", phase) if task == "race5" else phase
        top_mps = float(top_phase.get(
            "verified_top_speed_0_5s_mps",
            top_phase.get(
                "peak_horizontal_speed_mps",
                top_phase.get(
                    "peak_world_forward_speed_mps",
                    top_phase.get("peak_abs_forward_speed_mps", 0.0),
                ),
            ),
        ))
        acceleration = phase.get("acceleration_first_second_mps2")
        zero_to_half = phase.get("time_to_0_5_mps_s")
        agility = 100.0 * (
            0.55 * components.get("heading", 0.0)
            + 0.45 * components.get("low_lateral_drift", 0.0)
        )
    elif task == "sprint":
        phase = phases.get("speed_055", {})
        sustained_mps = sprint_steady_speed(phase)
        top_mps = float(phase.get("peak_abs_forward_speed_mps", sustained_mps))
        acceleration = phase.get("acceleration_first_second_mps2")
        zero_to_half = phase.get("time_to_0_5_mps_s")
        agility = 100.0 * (
            0.40 * components.get("stopping", 0.0)
            + 0.35 * components.get("low_lateral_drift", 0.0)
            + 0.25 * components.get("stability", 0.0)
        )
    else:
        phase = phases.get("forward", {})
        sustained_mps = float(
            phase.get("steady_mean_forward_speed_mps", phase.get("mean_forward_speed_mps", 0.0))
        )
        top_mps = float(phase.get("peak_abs_forward_speed_mps", abs(sustained_mps)))
        acceleration = phase.get("acceleration_first_second_mps2")
        zero_to_half = phase.get("time_to_0_5_mps_s")
        agility = 100.0 * sum(
            components.get(name, 0.0)
            for name in ("heading_control", "stopping", "low_lateral_slip")
        ) / 3.0
    return {
        "sustained_speed_mps": round(sustained_mps, 4),
        "sustained_speed_mph": round(sustained_mps * MPS_TO_MPH, 3),
        "top_speed_mps": round(top_mps, 4),
        "top_speed_mph": round(top_mps * MPS_TO_MPH, 3),
        "acceleration_first_second_mps2": round(float(acceleration), 4) if acceleration is not None else None,
        "zero_to_0_5_mps_s": round(float(zero_to_half), 3) if zero_to_half is not None else None,
        "agility_score": round(max(0.0, min(100.0, agility)), 2),
    }


def score_evaluation(evaluation: dict[str, Any], task: str) -> dict[str, Any]:
    """Return a transparent 0-100 heuristic; it is not an auto-promotion gate."""
    if task in {"race", "race5"}:
        race = evaluation.get("phases", {}).get("race", {})
        if not race:
            return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
        finished = bool(race.get("finished_5m"))
        finish_time = race.get("finish_time_5m_s")
        progress = float(race.get("forward_progress_m", 0.0))
        yaw = abs(float(race.get("yaw_change_deg", 360.0)))
        tilt = float(race.get("tilt_max_deg", 90.0))
        lateral = float(race.get("steady_mean_abs_lateral_speed_mps", 1.0))
        # Side-to-side body-frame velocity is gait sway, not trajectory drift.
        # A swizzle can have substantial oscillatory lateral speed while its
        # world-space path remains straight.  Prefer measured cross-track
        # excursion; retain a conservative legacy fallback for old reports.
        race_drift_ft = float(
            race.get(
                "max_lateral_drift_ft",
                lateral * max(float(race.get("duration_s", 0.0)), 1.0) / 0.3048,
            )
        )
        height = float(race.get("trunk_height_mean_m", 0.0))
        if task == "race5":
            phases = evaluation.get("phases", {})
            settle = phases.get("settle", {})
            cruise = phases.get("cruise", {})
            stop_cruise = phases.get("stop_cruise", {})
            turn_left = phases.get("turn_left", {})
            turn_right = phases.get("turn_right", {})
            max_speed_run = phases.get("max_speed", {})
            top_speed = float(race.get("peak_world_forward_speed_mps", 0.0))
            sustained_speed = float(
                race.get(
                    "steady_mean_world_forward_speed_mps",
                    race.get("mean_world_forward_speed_mps", 0.0),
                )
            )
            acceleration = max(0.0, float(race.get("acceleration_first_second_mps2", 0.0)))
            components = {
                "five_mph_speed": min(1.0, max(0.0, sustained_speed / 2.2352)),
                "acceleration": min(1.0, acceleration / 1.5),
                "heading": max(0.0, 1.0 - yaw / 60.0),
                "stability": max(0.0, 1.0 - tilt / 30.0),
                "low_lateral_drift": max(0.0, 1.0 - race_drift_ft / 0.75),
            }
            weights = {
                "five_mph_speed": 0.50,
                "acceleration": 0.15,
                "heading": 0.15,
                "stability": 0.10,
                "low_lateral_drift": 0.10,
            }
            idle_end_speed = float(settle.get("end_abs_forward_speed_mps", 99.0))
            idle_distance = abs(float(settle.get("forward_distance_m", 99.0)))
            idle_tilt = float(settle.get("tilt_max_deg", 90.0))
            idle_height = float(settle.get("trunk_height_mean_m", 0.0))
            idle_score = 0.5 * max(0.0, 1.0 - idle_end_speed / 0.05) + 0.5 * max(
                0.0, 1.0 - idle_distance / 0.15
            )
            cruise_speed = float(cruise.get("steady_mean_forward_speed_mps", 0.0))
            cruise_yaw = abs(float(cruise.get("yaw_change_deg", 360.0)))
            stop_time = stop_cruise.get("stop_time_below_0_05_mps_s")
            stop_time_value = float(stop_time) if stop_time is not None else 99.0
            stop_end_speed = float(stop_cruise.get("end_abs_forward_speed_mps", 99.0))
            left_yaw = float(turn_left.get("yaw_change_deg", -360.0))
            right_yaw = float(turn_right.get("yaw_change_deg", 360.0))
            retention_phases = (cruise, stop_cruise, turn_left, turn_right)
            retention_tilt = max(
                (float(phase.get("tilt_max_deg", 90.0)) for phase in retention_phases),
                default=90.0,
            )
            retention_height = min(
                (float(phase.get("trunk_height_mean_m", 0.0)) for phase in retention_phases),
                default=0.0,
            )
            long_run_heading = float(max_speed_run.get("max_heading_error_deg", 3600.0))
            long_run_drift_ft = float(max_speed_run.get("max_lateral_drift_ft", 999.0))
            long_run_tilt = float(max_speed_run.get("tilt_max_deg", 90.0))
            long_run_height = float(max_speed_run.get("trunk_height_mean_m", 0.0))
            finished_100ft = bool(max_speed_run.get("finished_100ft"))
            gates = {
                "idle_hold": {
                    "passed": idle_end_speed <= 0.05 and idle_distance <= 0.15
                    and idle_tilt <= 18.0 and idle_height >= 0.09,
                    "value": round(idle_end_speed * MPS_TO_MPH, 3),
                    "maximum": round(0.05 * MPS_TO_MPH, 3),
                    "unit": "mph end speed",
                },
                "race_heading": {"passed": yaw <= 25.0, "value": round(yaw, 2), "maximum": 25.0, "unit": "deg"},
                "lateral_drift": {
                    "passed": race_drift_ft <= 0.75,
                    "value": round(race_drift_ft, 2),
                    "maximum": 0.75,
                    "unit": "ft cross-track",
                },
                "race_tilt": {"passed": tilt <= 18.0, "value": round(tilt, 2), "maximum": 18.0, "unit": "deg"},
                "race_upright": {"passed": height >= 0.09, "value": "upright" if height >= 0.09 else "fell", "required": "upright"},
                "cruise_speed": {"passed": cruise_speed >= 0.25, "value": round(cruise_speed * MPS_TO_MPH, 3), "minimum": round(0.25 * MPS_TO_MPH, 3), "unit": "mph"},
                "cruise_heading": {"passed": cruise_yaw <= 60.0, "value": round(cruise_yaw, 2), "maximum": 60.0, "unit": "deg"},
                "braking": {"passed": stop_time_value <= 2.0 and stop_end_speed <= 0.05, "value": round(stop_time_value, 2) if stop_time is not None else "not stopped", "maximum": 2.0, "unit": "s"},
                "turn_left": {"passed": left_yaw >= 30.0, "value": round(left_yaw, 2), "minimum": 30.0, "unit": "deg"},
                "turn_right": {"passed": right_yaw <= -30.0, "value": round(right_yaw, 2), "maximum": -30.0, "unit": "deg"},
                "retention_stability": {"passed": retention_tilt <= 18.0 and retention_height >= 0.09, "value": round(retention_tilt, 2), "maximum": 18.0, "unit": "deg tilt"},
                "a_to_b_100ft": {"passed": finished_100ft, "value": "finished" if finished_100ft else "did not reach B", "required": "finish A to B"},
                "long_run_heading": {"passed": long_run_heading <= 45.0, "value": round(long_run_heading, 2), "maximum": 45.0, "unit": "deg"},
                "long_run_drift": {"passed": long_run_drift_ft <= 3.0, "value": round(long_run_drift_ft, 2), "maximum": 3.0, "unit": "ft"},
                "long_run_upright": {"passed": long_run_tilt <= 18.0 and long_run_height >= 0.09, "value": "upright" if long_run_tilt <= 18.0 and long_run_height >= 0.09 else "unstable", "required": "upright"},
            }
            champion_eligible = all(item["passed"] for item in gates.values())
            goal_reached = sustained_speed >= 2.2352
            overall = 100.0 * sum(components[name] * weights[name] for name in weights)
            performance = athletic_performance(evaluation, task, components)
            turn_agility = 0.5 * (
                min(1.0, max(0.0, left_yaw / 90.0))
                + min(1.0, max(0.0, -right_yaw / 90.0))
            )
            retention_agility = 100.0 * (
                0.15 * idle_score
                + 0.17 * min(1.0, max(0.0, cruise_speed / 0.30))
                + 0.17 * max(0.0, 1.0 - cruise_yaw / 60.0)
                + 0.17 * max(0.0, 1.0 - stop_time_value / 2.0)
                + 0.17 * turn_agility
                + 0.17 * max(0.0, 1.0 - retention_tilt / 18.0)
            )
            performance["agility_score"] = round(retention_agility, 2)
            performance["idle_hold_passed"] = gates["idle_hold"]["passed"]
            performance["idle_end_speed_mph"] = round(idle_end_speed * MPS_TO_MPH, 3)
            performance["idle_drift_ft"] = round(idle_distance / 0.3048, 3)
            performance["race_max_drift_ft"] = round(race_drift_ft, 2)
            performance["skill_retention_passed"] = all(
                gates[name]["passed"]
                for name in (
                    "idle_hold", "cruise_speed", "cruise_heading", "braking",
                    "turn_left", "turn_right", "retention_stability",
                )
            )
            performance["long_run_max_drift_ft"] = round(long_run_drift_ft, 2)
            performance["long_run_max_heading_error_deg"] = round(long_run_heading, 2)
            performance["long_run_line_hold_passed"] = all(
                gates[name]["passed"]
                for name in ("long_run_heading", "long_run_drift", "long_run_upright")
            )
            performance["finished_100ft"] = finished_100ft
            performance["elapsed_time_100ft_s"] = max_speed_run.get("finish_time_100ft_s")
            performance["trap_speed_100ft_mph"] = max_speed_run.get("trap_speed_100ft_mph")
            performance["split_time_10ft_s"] = max_speed_run.get("split_time_10ft_s")
            performance["split_time_25ft_s"] = max_speed_run.get("split_time_25ft_s")
            performance["split_time_50ft_s"] = max_speed_run.get("split_time_50ft_s")
            performance["distance_remaining_100ft_ft"] = round(
                float(max_speed_run.get("distance_remaining_100ft_ft", 100.0)), 2
            )
            line_hold = evaluation.get("line_hold", {})
            performance["line_hold_enabled"] = bool(line_hold.get("enabled"))
            performance["control_stack"] = (
                "Policy + measured line hold"
                if line_hold.get("enabled")
                else "Raw policy"
            )
            performance["auto_steering_percent"] = round(
                float(max_speed_run.get("auto_steering_percent", 0.0)), 2
            )
            performance["distance_time_average_speed_mph"] = max_speed_run.get(
                "distance_time_average_speed_mph"
            )
            performance["speed_integration_error_percent"] = max_speed_run.get(
                "position_velocity_integration_error_percent"
            )
            performance["five_mph_target_percent"] = round(100.0 * sustained_speed / 2.2352, 1)
            performance["ten_mph_stretch_percent"] = round(100.0 * sustained_speed / 4.4704, 1)
            return {
                "overall": round(overall, 2),
                "label": "Race5 speed development score",
                "components": {name: round(value * 100.0, 2) for name, value in components.items()},
                "weights": weights,
                "qualification_gates": gates,
                "qualified": champion_eligible,
                "record_qualified": goal_reached and champion_eligible,
                "five_mph_goal_reached": goal_reached,
                "simulation_champion_eligible": champion_eligible,
                "performance": performance,
            }
        pace = min(1.0, 10.0 / float(finish_time)) if finished and finish_time else min(1.0, max(0.0, progress / 5.0))
        components = {
            "finish": 1.0 if finished else min(1.0, max(0.0, progress / 5.0)),
            "pace": pace,
            "heading": max(0.0, 1.0 - yaw / 90.0),
            "stability": max(0.0, 1.0 - tilt / 30.0),
            "low_lateral_drift": max(0.0, 1.0 - lateral / 0.05),
        }
        weights = {"finish": 0.35, "pace": 0.30, "heading": 0.15, "stability": 0.10, "low_lateral_drift": 0.10}
        gates = {
            "finish_5m": {"passed": finished, "value": finished, "required": True},
            "finish_time": {"passed": finished and float(finish_time) <= 12.0, "value": finish_time, "maximum": 12.0, "unit": "s"},
            "heading": {"passed": yaw <= 30.0, "value": round(yaw, 2), "maximum": 30.0, "unit": "deg"},
            "lateral_drift": {"passed": lateral <= 0.05, "value": round(lateral * MPS_TO_MPH, 3), "maximum": round(0.05 * MPS_TO_MPH, 3), "unit": "mph"},
            "tilt": {"passed": tilt <= 18.0, "value": round(tilt, 2), "maximum": 18.0, "unit": "deg"},
            "no_fall": {"passed": height >= 0.09, "value": round(height, 4), "minimum": 0.09, "unit": "m trunk height"},
        }
        qualified = all(item["passed"] for item in gates.values())
        overall = 100.0 * sum(components[name] * weights[name] for name in weights)
        if not qualified:
            overall = min(overall, 49.0)
        return {
            "overall": round(overall, 2),
            "label": "Race-v1 five-metre score",
            "components": {name: round(value * 100.0, 2) for name, value in components.items()},
            "weights": weights,
            "qualification_gates": gates,
            "qualified": qualified,
            "performance": athletic_performance(evaluation, task, components),
        }

    if task == "sprint":
        phases = evaluation.get("phases", {})
        required = (*SPRINT_SPEED_PHASES, "stop_030", "stop_040", "stop_050", "stop_055")
        if any(name not in phases for name in required):
            return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
        speeds = {name: sprint_steady_speed(phases[name]) for name in SPRINT_SPEED_PHASES}
        fast = phases["speed_055"]
        stop_times = [
            float(phases[name].get("stop_time_below_0_05_mps_s", 99.0))
            for name in ("stop_030", "stop_040", "stop_050", "stop_055")
        ]
        max_stop = max(stop_times)
        fast_lateral = float(fast.get("steady_mean_abs_lateral_speed_mps", 1.0))
        fast_tilt = float(fast.get("tilt_max_deg", 90.0))
        min_height = min(float(phases[name].get("trunk_height_mean_m", 0.0)) for name in SPRINT_SPEED_PHASES)
        components = {
            "top_speed": min(1.0, max(0.0, speeds["speed_055"] / 0.55)),
            "useful_range": sum(min(1.0, max(0.0, speeds[name] / target)) for name, target in {
                "speed_030": 0.30, "speed_040": 0.40, "speed_050": 0.45, "speed_055": 0.50,
            }.items()) / 4.0,
            "stability": max(0.0, 1.0 - fast_tilt / 30.0),
            "low_lateral_drift": max(0.0, 1.0 - fast_lateral / 0.05),
            "stopping": max(0.0, 1.0 - max_stop / 1.5),
        }
        weights = {
            "top_speed": 0.40,
            "useful_range": 0.20,
            "stability": 0.15,
            "low_lateral_drift": 0.10,
            "stopping": 0.15,
        }
        gates = {
            "speed_at_0_30": {"passed": speeds["speed_030"] >= 0.30, "value": round(speeds["speed_030"] * MPS_TO_MPH, 3), "minimum": round(0.30 * MPS_TO_MPH, 3), "unit": "mph"},
            "speed_at_0_40": {"passed": speeds["speed_040"] >= 0.40, "value": round(speeds["speed_040"] * MPS_TO_MPH, 3), "minimum": round(0.40 * MPS_TO_MPH, 3), "unit": "mph"},
            "speed_at_0_50": {"passed": speeds["speed_050"] >= 0.45, "value": round(speeds["speed_050"] * MPS_TO_MPH, 3), "minimum": round(0.45 * MPS_TO_MPH, 3), "unit": "mph"},
            "speed_at_0_55": {"passed": speeds["speed_055"] >= 0.50, "value": round(speeds["speed_055"] * MPS_TO_MPH, 3), "minimum": round(0.50 * MPS_TO_MPH, 3), "unit": "mph"},
            "lateral_drift_at_0_55": {"passed": fast_lateral <= 0.05, "value": round(fast_lateral * MPS_TO_MPH, 3), "maximum": round(0.05 * MPS_TO_MPH, 3), "unit": "mph"},
            "tilt_at_0_55": {"passed": fast_tilt <= 15.0, "value": round(fast_tilt, 2), "maximum": 15.0, "unit": "deg"},
            "stop_from_useful_speeds": {"passed": max_stop <= 1.5, "value": round(max_stop, 2), "maximum": 1.5, "unit": "s"},
            "no_fall": {"passed": min_height >= 0.09, "value": round(min_height, 4), "minimum": 0.09, "unit": "m trunk height"},
        }
        qualified = all(item["passed"] for item in gates.values())
        overall = 100.0 * sum(components[key] * weights[key] for key in weights)
        if not qualified:
            overall = min(overall, 49.0)
        return {
            "overall": round(overall, 2),
            "label": "Sprint-v1 qualification score",
            "components": {key: round(value * 100.0, 2) for key, value in components.items()},
            "weights": weights,
            "qualification_gates": gates,
            "qualified": qualified,
            "performance": athletic_performance(evaluation, task, components),
        }

    if task == "backflip":
        flip = evaluation.get("frontflip", {})
        if not flip:
            return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
        episodes = int(flip.get("episodes", 0))
        success = float(flip.get("success_rate", 0.0))
        takeoff = float(flip.get("takeoff_rate", 0.0))
        landing = float(flip.get("landing_rate", 0.0))
        settled = float(flip.get("settled_rate", 0.0))
        strikes = float(flip.get("body_contact_rate", 1.0))
        clearance = float(flip.get("median_peak_clearance_m", 0.0))
        rotation = float(flip.get("median_forward_rotation_deg", 0.0))
        offaxis = float(flip.get("median_offaxis_rotation_deg", 360.0))
        drift = float(flip.get("median_horizontal_drift_m", 1.0))
        components = {
            "complete_maneuver": success,
            "takeoff": takeoff,
            "clean_landing": landing * settled,
            "no_body_strike": max(0.0, 1.0 - strikes),
            "clearance": min(1.0, max(0.0, clearance / 0.08)),
            "rotation": bounded_score(rotation, 360.0, 60.0),
            "sagittal_control": max(0.0, 1.0 - offaxis / 60.0),
            "low_drift": max(0.0, 1.0 - drift / 0.12),
        }
        weights = {
            "complete_maneuver": 0.40,
            "takeoff": 0.10,
            "clean_landing": 0.15,
            "no_body_strike": 0.10,
            "clearance": 0.075,
            "rotation": 0.075,
            "sagittal_control": 0.05,
            "low_drift": 0.05,
        }
        gates = {
            "sample_size": {"passed": episodes >= 256, "value": episodes, "minimum": 256},
            "unassisted": {"passed": bool(flip.get("unassisted")), "value": bool(flip.get("unassisted"))},
            "success_rate": {"passed": success >= 0.80, "value": round(success, 4), "minimum": 0.80},
            "takeoff_rate": {"passed": takeoff >= 0.90, "value": round(takeoff, 4), "minimum": 0.90},
            "landing_rate": {"passed": landing >= 0.85, "value": round(landing, 4), "minimum": 0.85},
            "settled_rate": {"passed": settled >= 0.80, "value": round(settled, 4), "minimum": 0.80},
            "body_strikes": {"passed": strikes <= 0.01, "value": round(strikes, 4), "maximum": 0.01},
            "clearance": {"passed": clearance >= 0.05, "value": round(clearance, 4), "minimum": 0.05},
            "rotation": {"passed": 300.0 <= rotation <= 420.0, "value": round(rotation, 2), "minimum": 300.0, "maximum": 420.0},
            "offaxis": {"passed": offaxis <= 60.0, "value": round(offaxis, 2), "maximum": 60.0},
            "drift": {"passed": drift <= 0.12, "value": round(drift, 4), "maximum": 0.12},
        }
        overall = 100.0 * sum(components[key] * weights[key] for key in weights)
        if not all(item["passed"] for item in gates.values()):
            overall = min(overall, 49.0)
        return {
            "overall": round(overall, 2),
            "label": "unassisted front-flip score",
            "components": {key: round(value * 100.0, 2) for key, value in components.items()},
            "weights": weights,
            "qualification_gates": gates,
            "qualified": all(item["passed"] for item in gates.values()),
        }

    if task == "hop":
        hop = evaluation.get("hop", {})
        if not hop:
            return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
        clearance = float(hop.get("peak_clearance_m", 0.0))
        air_time = float(hop.get("air_time_s", 0.0))
        drift = float(hop.get("horizontal_drift_m", 1.0))
        final_tilt = float(hop.get("final_tilt_mean_deg", 90.0))
        final_speed = float(hop.get("final_speed_mean_mps", 1.0))
        grounded = float(hop.get("final_both_grounded_fraction", 0.0))
        components = {
            # An obvious robot-scale jump, not a contact-sensor flicker. The
            # old 15 mm / 60 ms gates falsely qualified a head twitch.
            "clearance": bounded_score(clearance, 0.080, 0.080),
            "air_time": min(1.0, max(0.0, air_time / 0.20)),
            "landing": grounded if hop.get("landing_detected") else 0.0,
            "upright": max(0.0, 1.0 - final_tilt / 25.0),
            "stillness": max(0.0, 1.0 - final_speed / 0.15) if hop.get("landing_detected") else 0.0,
            "low_drift": max(0.0, 1.0 - drift / 0.08),
        }
        weights = {
            "clearance": 0.25,
            "air_time": 0.10,
            "landing": 0.25,
            "upright": 0.15,
            "stillness": 0.15,
            "low_drift": 0.10,
        }
        gates = {
            "takeoff": {"passed": bool(hop.get("takeoff_detected")), "value": bool(hop.get("takeoff_detected"))},
            "clearance": {"passed": clearance >= 0.050, "value": round(clearance, 4), "minimum": 0.050},
            "air_time": {"passed": air_time >= 0.12, "value": round(air_time, 4), "minimum": 0.12},
            "landing": {"passed": bool(hop.get("landing_detected")), "value": bool(hop.get("landing_detected"))},
            "upright": {"passed": final_tilt <= 15.0, "value": round(final_tilt, 2), "maximum": 15.0},
            "settled": {"passed": final_speed <= 0.10, "value": round(final_speed, 4), "maximum": 0.10},
            "drift": {"passed": drift <= 0.05, "value": round(drift, 4), "maximum": 0.05},
        }
        overall = 100.0 * sum(components[key] * weights[key] for key in weights)
        if not all(item["passed"] for item in gates.values()):
            overall = min(overall, 49.0)
        return {
            "overall": round(overall, 2),
            "label": "heuristic simulation score",
            "components": {key: round(value * 100.0, 2) for key, value in components.items()},
            "weights": weights,
            "qualification_gates": gates,
            "qualified": all(item["passed"] for item in gates.values()),
        }

    phases = evaluation.get("phases", {})
    forward = phases.get("forward", {})
    reverse = phases.get("reverse", {})
    powered = [phases[name] for name in ("forward", "reverse", "heading_left", "heading_right") if name in phases]
    if not forward:
        return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
    forward_speed = float(forward.get("steady_mean_forward_speed_mps", forward.get("mean_forward_speed_mps", 0.0)))
    components = {
        "forward_tracking": bounded_score(forward_speed, 0.3, 0.3),
        "ground_contact": sum(float(item.get("both_blades_grounded_fraction", 0.0)) for item in powered) / max(1, len(powered)),
        "stability": max(0.0, 1.0 - max(float(item.get("tilt_max_deg", 90.0)) for item in powered) / 20.0),
        "smoothness": max(0.0, 1.0 - sum(float(item.get("mean_action_acceleration", 1.0)) for item in powered) / max(1, len(powered)) / 0.08),
        "low_lateral_slip": max(0.0, 1.0 - sum(float(item.get("mean_abs_lateral_speed_mps", 1.0)) for item in powered) / max(1, len(powered)) / 0.1),
    }
    weights = {"forward_tracking": 0.25, "ground_contact": 0.2, "stability": 0.2, "smoothness": 0.15, "low_lateral_slip": 0.2}
    gates: dict[str, dict[str, Any]] = {}
    if task in {"swizzle", "roller"} and reverse:
        reverse_speed = float(reverse.get("steady_mean_forward_speed_mps", reverse.get("mean_forward_speed_mps", 0.0)))
        stop_forward = phases.get("stop_forward", phases.get("coast_forward", {}))
        stop_reverse = phases.get("stop_reverse", phases.get("coast_reverse", {}))
        heading_left = phases.get("heading_left", {})
        heading_right = phases.get("heading_right", {})
        stop_end = max(
            float(stop_forward.get("end_abs_forward_speed_mps", 1.0)),
            float(stop_reverse.get("end_abs_forward_speed_mps", 1.0)),
        )
        left_rate = float(heading_left.get("mean_yaw_rate_rad_s", 0.0))
        right_rate = float(heading_right.get("mean_yaw_rate_rad_s", 0.0))
        components["reverse_tracking"] = bounded_score(reverse_speed, -0.3, 0.3)
        components["stopping"] = max(0.0, 1.0 - stop_end / 0.15)
        components["turning"] = 0.5 * min(1.0, max(0.0, left_rate / 0.25)) + 0.5 * min(1.0, max(0.0, -right_rate / 0.25))
        components["swizzle_cycles"] = min(1.0, sum(float(phases.get(name, {}).get("estimated_swizzle_cycles", 0.0)) for name in ("forward", "reverse")) / 16.0)
        weights = {"forward_tracking": 0.2, "reverse_tracking": 0.15, "stopping": 0.15, "turning": 0.15, "stability": 0.1, "low_lateral_slip": 0.1, "ground_contact": 0.05, "smoothness": 0.05, "swizzle_cycles": 0.05}
        gates = {
            "forward_speed": {"passed": forward_speed >= 0.15, "value": round(forward_speed, 4), "minimum": 0.15},
            "reverse_speed": {"passed": reverse_speed <= -0.10, "value": round(reverse_speed, 4), "maximum": -0.10},
            "stopping": {"passed": stop_end <= 0.05, "value": round(stop_end, 4), "maximum": 0.05},
            "turn_left": {"passed": left_rate >= 0.15, "value": round(left_rate, 4), "minimum": 0.15},
            "turn_right": {"passed": right_rate <= -0.15, "value": round(right_rate, 4), "maximum": -0.15},
            "stability": {"passed": max(float(item.get("tilt_max_deg", 90.0)) for item in powered) <= 25.0, "value": round(max(float(item.get("tilt_max_deg", 90.0)) for item in powered), 2), "maximum": 25.0},
        }
    overall = 100.0 * sum(components[key] * weights[key] for key in weights) / sum(weights.values())
    if gates and not gates["forward_speed"]["passed"]:
        overall = min(overall, 49.0)
    if gates and not gates["reverse_speed"]["passed"]:
        overall = min(overall, 49.0)
    return {"overall": round(overall, 2), "label": "heuristic simulation score", "components": {key: round(value * 100.0, 2) for key, value in components.items()}, "weights": weights, "qualification_gates": gates, "qualified": bool(gates) and all(item["passed"] for item in gates.values())}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        try:
            remote = run("config", "--get", "remote.origin.url")
        except subprocess.CalledProcessError:
            remote = ""
        try:
            branch = run("branch", "--show-current")
        except subprocess.CalledProcessError:
            branch = ""
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
            "branch": branch or None,
            "remote": remote or None,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "branch": None, "remote": None}


def infer_task(run_dir: Path) -> str:
    parent = run_dir.parent.name
    return {
        "velocity_swizzle": "swizzle",
        "velocity_race": "race",
        # A speed-scout transfer uses the Race5 browser contract and must be
        # evaluated/presented against Race5 once it reaches official friction.
        "microduck_speed_scout_transfer": "race5",
        "microduck_speed_official_adaptation": "race5",
        "velocity_rollers": "roller",
        "velocity": "walking",
        "roller_hop": "hop",
        "roller_backflip": "backflip",
    }.get(parent, parent.replace("velocity_", "") or "unknown")


def display_task_name(task: str) -> str:
    """Return the operator-facing name for legacy internal task IDs."""
    spec = task_spec(task)
    return str(spec["display_name"]) if spec else task.replace("_", " ").title()


def display_experiment_label(task: str, label: str) -> str:
    """Correct the historical forward-rotation label without moving artifacts."""
    if task == "backflip":
        return re.sub("backflip", "frontflip", label, flags=re.IGNORECASE)
    return label


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match:
        return int(match.group(1))
    if path.name == "best_speed_discovery.pt":
        selection = path.parent / "best_speed_discovery.json"
        if selection.is_file():
            try:
                return int(read_json(selection)["best"]["iteration"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    return -1


def artifact_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def choose_artifacts(run_dir: Path) -> tuple[Path | None, Path | None]:
    selected_checkpoint = run_dir / "best_speed_discovery.pt"
    selected_policy = run_dir / "best_speed_discovery.onnx"
    if selected_checkpoint.is_file() and selected_policy.is_file():
        # Checkpoint selection is a first-class artifact: never pair the chosen
        # ONNX policy with the final model merely because both were written
        # before registration.
        return selected_checkpoint, selected_policy
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    policies = sorted(run_dir.glob("*.onnx"), key=lambda item: item.stat().st_mtime)
    policy = policies[-1] if policies else None
    if policy is not None:
        # Export follows checkpoint creation. During an active run, do not pair
        # a newly written checkpoint with the preceding ONNX export.
        exported_checkpoints = [
            item for item in checkpoints if item.stat().st_mtime <= policy.stat().st_mtime
        ]
        checkpoint = exported_checkpoints[-1] if exported_checkpoints else None
    else:
        checkpoint = checkpoints[-1] if checkpoints else None
    return checkpoint, policy


def make_run_id(run_dir: Path, task: str, iteration: int | None) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_dir.name).strip("-")
    suffix = f"i{iteration}" if iteration is not None and iteration >= 0 else "export"
    return f"{task}-{safe_name}-{suffix}"


def experiment_kind(run_dir: Path) -> str:
    if LAB_ROOT / "baselines" in run_dir.parents:
        return "factory"
    return "smoke" if "smoke" in run_dir.name.lower() else "training"


def experiment_id(run_dir: Path, task: str) -> str:
    return f"{task}:{run_dir.name}"


def experiment_label(run_dir: Path) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", run_dir.name)


def active_training_experiments() -> set[str]:
    experiments: set[str] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return experiments
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        match = re.search(r"--agent\.run-name\s+([^\s]+)", command)
        if match:
            experiments.add(match.group(1))
    return experiments


def snapshot_artifact(source: Path | None, destination_dir: Path) -> dict[str, Any] | None:
    if source is None:
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return artifact_record(destination)


def sprint_dashboard_verdict(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return ""
    score = evaluation.get("policy_bench_score", {})
    comparison = evaluation.get("baseline_comparison", {})
    if score.get("overall") is None:
        return ""
    qualified = bool(score.get("qualified"))
    improved = bool(comparison.get("improved"))
    passed = sum(bool(item.get("passed")) for item in score.get("qualification_gates", {}).values())
    total = len(score.get("qualification_gates", {}))
    tone = "good" if qualified and improved else "bad"
    headline = "YES — improved vs Pollen" if qualified and improved else "NO — improvement not proven"
    candidate = comparison.get("candidate_steady_speed_mps")
    baseline = comparison.get("baseline_steady_speed_mps")
    percent = comparison.get("speed_delta_percent")
    if candidate is None or baseline is None:
        detail = f"Score {score['overall']:.2f}/100 · {passed}/{total} qualification gates passed"
    else:
        detail = (
            f"1.23 mph command: {candidate * MPS_TO_MPH:.2f} vs {baseline * MPS_TO_MPH:.2f} mph · "
            f"{percent:+.1f}% · score {score['overall']:.2f}/100 · {passed}/{total} gates passed"
        )
    return (
        f"<div class='result-banner {tone}'><div><small>TRAINING RESULT</small>"
        f"<strong>{html.escape(headline)}</strong><span>{html.escape(detail)}</span></div>"
        "<span class='result-arrow'>View the evidence in Saved models ↓</span></div>"
    )


def athletic_kpis(score: dict[str, Any]) -> str:
    performance = score.get("performance", {})
    sustained = performance.get("sustained_speed_mph")
    top = performance.get("top_speed_mph")
    acceleration = performance.get("acceleration_first_second_mps2")
    zero_to_half = performance.get("zero_to_0_5_mps_s")
    agility = performance.get("agility_score")
    target = performance.get("five_mph_target_percent")
    stretch = performance.get("ten_mph_stretch_percent")
    finished_100ft = performance.get("finished_100ft")
    elapsed_100ft = performance.get("elapsed_time_100ft_s")
    trap_100ft = performance.get("trap_speed_100ft_mph")
    remaining_100ft = performance.get("distance_remaining_100ft_ft")
    has_100ft = "finished_100ft" in performance
    return (
        f"<div><small>Sustained speed</small><strong>{float(sustained):.2f} mph</strong></div>"
        if sustained is not None else "<div><small>Sustained speed</small><strong>Rerun needed</strong></div>"
    ) + (
        f"<div><small>Verified top · 0.5 s</small><strong>{float(top):.2f} mph</strong></div>"
        if top is not None else "<div><small>Verified top · 0.5 s</small><strong>Rerun needed</strong></div>"
    ) + (
        f"<div><small>First-second acceleration</small><strong>{float(acceleration) * MPS_TO_MPH:+.2f} mph/s</strong></div>"
        if acceleration is not None else "<div><small>First-second acceleration</small><strong>Rerun needed</strong></div>"
    ) + (
        f"<div><small>0 → 1.12 mph</small><strong>{float(zero_to_half):.2f} s</strong></div>"
        if zero_to_half is not None else "<div><small>0 → 1.12 mph</small><strong>Not reached</strong></div>"
    ) + (
        f"<div><small>Agility score</small><strong>{float(agility):.0f}/100</strong></div>"
        if agility is not None else "<div><small>Agility score</small><strong>—</strong></div>"
    ) + (
        f"<div><small>5 mph target</small><strong>{float(target):.1f}%</strong></div>"
        if target is not None else ""
    ) + (
        f"<div><small>10 mph stretch</small><strong>{float(stretch):.1f}%</strong></div>"
        if stretch is not None else ""
    ) + (
        f"<div><small>100 ft A → B</small><strong>{float(elapsed_100ft):.2f} s</strong></div>"
        if finished_100ft and elapsed_100ft is not None
        else f"<div><small>100 ft A → B</small><strong>{float(remaining_100ft):.1f} ft short</strong></div>"
        if has_100ft and remaining_100ft is not None else ""
    ) + (
        f"<div><small>Trap speed</small><strong>{float(trap_100ft):.2f} mph</strong></div>"
        if trap_100ft is not None else "<div><small>Trap speed</small><strong>Not reached</strong></div>"
        if has_100ft else ""
    )


def sprint_report_summary(evaluation: dict[str, Any]) -> str:
    score = evaluation.get("policy_bench_score", {})
    comparison = evaluation.get("baseline_comparison", {})
    if score.get("overall") is None:
        return ""
    qualified = bool(score.get("qualified"))
    improved = bool(comparison.get("improved"))
    speed_improved = bool(comparison.get("speed_improved", improved))
    verdict = "YES — this run improved skating speed" if qualified and improved else "NO — improvement is not proven"
    tone = "good" if qualified and improved else "bad"
    percent = comparison.get("speed_delta_percent")
    speed_copy = (
        f"At the 1.23 mph command, the trained policy sustained "
        f"{float(comparison.get('candidate_steady_speed_mps', 0.0)) * MPS_TO_MPH:.2f} mph versus "
        f"{float(comparison.get('baseline_steady_speed_mps', 0.0)) * MPS_TO_MPH:.2f} mph for Pollen "
        f"({float(percent or 0.0):+.1f}%)."
    ) if comparison else "The official Pollen baseline comparison is unavailable."
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
        f"<td><strong class='{'pass' if gate.get('passed') else 'fail'}'>{'PASS' if gate.get('passed') else 'FAIL'}</strong></td>"
        f"<td>{html.escape(str(gate.get('value', '—')))} {html.escape(str(gate.get('unit', '')))}</td>"
        f"<td>{'≥ ' + str(gate['minimum']) if 'minimum' in gate else '≤ ' + str(gate.get('maximum', '—'))}</td>"
        "</tr>"
        for name, gate in score.get("qualification_gates", {}).items()
    )
    command_rows = "".join(
        "<tr>"
        f"<td>{row['command_mps'] * MPS_TO_MPH:.2f} mph</td><td>{row['baseline_mps'] * MPS_TO_MPH:.2f} mph</td>"
        f"<td>{row['candidate_mps'] * MPS_TO_MPH:.2f} mph</td><td class='{'pass' if row['delta_mps'] > 0 else 'fail'}'>{row['delta_mps'] * MPS_TO_MPH:+.2f} mph</td>"
        "</tr>"
        for row in comparison.get("command_speeds", [])
    )
    comparison_rows = ""
    if comparison:
        comparison_rows = (
            f"<tr><td>Steady speed</td><td>{comparison['baseline_steady_speed_mps'] * MPS_TO_MPH:.2f} mph</td><td>{comparison['candidate_steady_speed_mps'] * MPS_TO_MPH:.2f} mph</td><td class='pass'>{comparison['speed_delta_mps'] * MPS_TO_MPH:+.2f} mph</td></tr>"
            f"<tr><td>Lateral drift</td><td>{comparison['baseline_lateral_mps'] * MPS_TO_MPH:.2f} mph</td><td>{comparison['candidate_lateral_mps'] * MPS_TO_MPH:.2f} mph</td><td>{comparison['lateral_delta_mps'] * MPS_TO_MPH:+.2f} mph</td></tr>"
            f"<tr><td>Maximum tilt</td><td>{comparison['baseline_tilt_max_deg']:.2f}°</td><td>{comparison['candidate_tilt_max_deg']:.2f}°</td><td>{comparison['tilt_delta_deg']:+.2f}°</td></tr>"
            f"<tr><td>Stop time</td><td>{comparison['baseline_stop_time_s']:.2f} s</td><td>{comparison['candidate_stop_time_s']:.2f} s</td><td>{comparison['stop_time_delta_s']:+.2f} s</td></tr>"
        )
    passed = sum(bool(item.get("passed")) for item in score.get("qualification_gates", {}).values())
    total = len(score.get("qualification_gates", {}))
    return (
        f"<section class='result-detail {tone}'><p class='eyebrow'>SPRINT-V1 VERDICT</p>"
        f"<h2>{html.escape(verdict)}</h2><p class='result-lede'>{html.escape(speed_copy)}</p>"
        f"<div class='result-kpis'><div><small>Score</small><strong>{score['overall']:.2f}/100</strong></div>"
        f"<div><small>Qualification</small><strong>{passed}/{total} gates passed</strong></div>"
        f"<div><small>Baseline verdict</small><strong>{html.escape(str(comparison.get('verdict', 'Unavailable')))}</strong></div>"
        f"{athletic_kpis(score)}</div>"
        f"<h2>At the 1.23 mph command</h2><div class='table-wrap'><table><tr><th>Metric</th><th>Pollen baseline</th><th>Trained policy</th><th>Change</th></tr>{comparison_rows}</table></div>"
        f"<h2>Speed across useful commands</h2><div class='table-wrap'><table><tr><th>Command</th><th>Pollen</th><th>Trained</th><th>Change</th></tr>{command_rows}</table></div>"
        f"<h2>Qualification gates</h2><div class='table-wrap'><table><tr><th>Gate</th><th>Result</th><th>Measured</th><th>Requirement</th></tr>{gate_rows}</table></div>"
        f"<p class='muted'>{html.escape(str(comparison.get('note', 'Simulation evidence only; hardware validation is still required.')))}</p></section>"
    )


def race_dashboard_verdict(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return ""
    score = evaluation.get("policy_bench_score", {})
    comparison = evaluation.get("baseline_comparison", {})
    if score.get("overall") is None:
        return ""
    qualified = bool(score.get("qualified"))
    improved = bool(comparison.get("improved"))
    speed_improved = bool(comparison.get("speed_improved", improved))
    race5 = evaluation.get("profile") == "race-5mph"
    champion_eligible = bool(score.get("simulation_champion_eligible", qualified))
    goal_reached = bool(score.get("five_mph_goal_reached"))
    simulation_improved = improved and champion_eligible
    tone = "good" if simulation_improved else "bad"
    headline = (
        "5 MPH GOAL REACHED" if goal_reached and champion_eligible
        else "FASTER THAN POLLEN" if simulation_improved
        else "FASTER PEAK, BUT A → B FAILED" if speed_improved
        else "NO VERIFIED SPEED IMPROVEMENT"
    ) if race5 else ("NEW RACE LEADER" if simulation_improved else "NOT YET — Sprint-v3 still leads")
    race = evaluation.get("phases", {}).get("race", {})
    candidate_mph = float(score.get("performance", {}).get("sustained_speed_mph", 0.0))
    if race5:
        baseline_mph = comparison.get("baseline_sustained_speed_mph")
        delta_mph = comparison.get("speed_delta_mph")
        performance = f"{candidate_mph:.2f} mph sustained"
        if baseline_mph is not None and delta_mph is not None:
            performance += f" vs Pollen {float(baseline_mph):.2f} mph ({float(delta_mph):+.2f} mph)"
    else:
        performance = f"{candidate_mph:.2f} mph sustained"
    gates = score.get("qualification_gates", {})
    passed = sum(bool(item.get("passed")) for item in gates.values())
    return (
        f"<div class='result-banner {tone}'><div><small>{'RACE5 RECORD RESULT' if race5 else 'RACE-V1 RESULT'}</small>"
        f"<strong>{html.escape(headline)}</strong>"
        f"<span>{html.escape(performance)} · {passed}/{len(gates)} safety/control checks passed</span></div>"
        "<span class='result-arrow'>Open the heat report ↓</span></div>"
    )


def race_report_summary(evaluation: dict[str, Any]) -> str:
    score = evaluation.get("policy_bench_score", {})
    comparison = evaluation.get("baseline_comparison", {})
    race = evaluation.get("phases", {}).get("race", {})
    if score.get("overall") is None or not race:
        return ""
    qualified = bool(score.get("qualified"))
    improved = bool(comparison.get("improved"))
    speed_improved = bool(comparison.get("speed_improved", improved))
    race5 = evaluation.get("profile") == "race-5mph"
    champion_eligible = bool(score.get("simulation_champion_eligible", qualified))
    goal_reached = bool(score.get("five_mph_goal_reached"))
    simulation_improved = improved and champion_eligible
    tone = "good" if simulation_improved else "bad"
    verdict = (
        "YES — 5 mph goal reached" if goal_reached and champion_eligible
        else "YES — faster than Pollen" if simulation_improved
        else "FASTER PEAK — but it did not win the unassisted A-to-B race" if speed_improved
        else "NO — sustained speed did not improve"
    ) if race5 else ("YES — new Race-v1 leader" if simulation_improved else "NO — Sprint-v3 remains the race leader")
    performance = score.get("performance", {})
    sustained_mph = float(performance.get("sustained_speed_mph", 0.0))
    pollen_mph = comparison.get("baseline_sustained_speed_mph")
    speed_delta = comparison.get("speed_delta_mph")
    comparison_copy = (
        f"Pollen {float(pollen_mph):.2f} mph · difference {float(speed_delta):+.2f} mph"
        if pollen_mph is not None and speed_delta is not None
        else "Pollen comparison unavailable"
    )
    control_copy = str(performance.get("control_stack", "Raw policy"))
    if performance.get("line_hold_enabled"):
        control_copy += f" · {float(performance.get('auto_steering_percent', 0.0)):.1f}% steering"
    gates = score.get("qualification_gates", {})
    passed = sum(bool(item.get("passed")) for item in gates.values())
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
        f"<td><strong class='{'pass' if gate.get('passed') else 'fail'}'>{'PASS' if gate.get('passed') else 'FAIL'}</strong></td>"
        f"<td>{html.escape(str(gate.get('value', '—')))} {html.escape(str(gate.get('unit', '')))}</td>"
        "</tr>"
        for name, gate in gates.items()
    )
    head_to_head_rows = ""
    if race5 and comparison:
        def display(value: Any, suffix: str, digits: int = 2) -> str:
            return f"{float(value):.{digits}f}{suffix}" if value is not None else "DNF"

        head_to_head_rows = "".join((
            f"<tr><td>100 ft elapsed time</td><td>{display(comparison.get('baseline_elapsed_time_100ft_s'), ' s')}</td><td>{display(comparison.get('candidate_elapsed_time_100ft_s'), ' s')}</td></tr>",
            f"<tr><td>Trap speed</td><td>{display(comparison.get('baseline_trap_speed_mph'), ' mph')}</td><td>{display(comparison.get('candidate_trap_speed_mph'), ' mph')}</td></tr>",
            f"<tr><td>Verified top speed (0.5 s)</td><td>{display(comparison.get('baseline_total_top_speed_mph'), ' mph')}</td><td>{display(comparison.get('candidate_total_top_speed_mph'), ' mph')}</td></tr>",
            f"<tr><td>First-second acceleration</td><td>{display(comparison.get('baseline_acceleration_mph_s'), ' mph/s')}</td><td>{display(comparison.get('candidate_acceleration_mph_s'), ' mph/s')}</td></tr>",
            f"<tr><td>Maximum lateral drift</td><td>{display(comparison.get('baseline_long_run_drift_ft'), ' ft')}</td><td>{display(comparison.get('candidate_long_run_drift_ft'), ' ft')}</td></tr>",
            f"<tr><td>Maximum heading error</td><td>{display(comparison.get('baseline_long_run_heading_error_deg'), '°')}</td><td>{display(comparison.get('candidate_long_run_heading_error_deg'), '°')}</td></tr>",
        ))
    head_to_head_html = (
        f"<h2>A-to-B comparison with Pollen</h2><div class='table-wrap'><table><tr><th>Race metric</th><th>Pollen</th><th>Candidate</th></tr>{head_to_head_rows}</table></div>"
        if head_to_head_rows else ""
    )
    return (
        f"<section class='result-detail {tone}'><p class='eyebrow'>{'RACE5 · 5 MPH VERDICT' if race5 else 'RACE-V1 VERDICT'}</p>"
        f"<h2>{html.escape(verdict)}</h2>"
        f"<p class='result-lede'>Sustained speed: {sustained_mph:.2f} mph · {html.escape(comparison_copy)} · "
        f"yaw change {float(race.get('yaw_change_deg', 0.0)):+.1f}° · {html.escape(control_copy)}.</p>"
        f"<div class='result-kpis'><div><small>Speed vs Pollen</small><strong>{float(speed_delta or 0.0):+.2f} mph</strong></div>"
        f"<div><small>Safety/control</small><strong>{passed}/{len(gates)} checks passed</strong></div>"
        f"<div><small>{'Record target' if race5 else 'Champion comparison'}</small><strong>{'5 mph · stretch 10 mph' if race5 else html.escape(str(comparison.get('verdict', 'Unavailable')))}</strong></div>"
        f"{athletic_kpis(score)}</div>"
        f"{head_to_head_html}"
        f"<h2>Safety and skill-retention checks</h2><div class='table-wrap'><table><tr><th>Check</th><th>Result</th><th>Measured</th></tr>{gate_rows}</table></div>"
        f"<p class='muted'>{html.escape(str(comparison.get('note', 'Simulation evidence only; hardware validation is required.')))}</p></section>"
    )


class Bench:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.runs_dir = self.state_dir / "runs"
        self.registry_path = self.state_dir / "registry.json"

    def initialize(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            write_json(
                self.registry_path,
                {"schema_version": SCHEMA_VERSION, "updated_at": utc_now(), "tasks": {}},
            )

    def manifest_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "manifest.json"

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        path = self.manifest_path(run_id)
        if not path.is_file():
            raise SystemExit(f"Unknown Policy Bench run: {run_id}")
        return read_json(path)

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.manifest_path(manifest["run_id"]), manifest)

    def register(self, run_dir: Path, task: str | None = None) -> dict[str, Any]:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Training run directory not found: {run_dir}")
        task = task or infer_task(run_dir)
        checkpoint, policy = choose_artifacts(run_dir)
        if checkpoint is None and policy is None:
            raise SystemExit(f"No model_*.pt or *.onnx artifacts found in {run_dir}")
        iteration = checkpoint_iteration(checkpoint) if checkpoint else None
        run_id = make_run_id(run_dir, task, iteration)
        existing = self.manifest_path(run_id)
        if existing.exists():
            previous = read_json(existing)
            changed = False
            for key, value in {
                "experiment_id": experiment_id(run_dir, task),
                "experiment_label": experiment_label(run_dir),
                "experiment_kind": experiment_kind(run_dir),
            }.items():
                if previous.get(key) != value:
                    previous[key] = value
                    changed = True
            source = previous.setdefault("source", {})
            for key, path in (("runtime", POLLEN_RUNTIME), ("simulator", POLLEN_SIMULATOR)):
                if key not in source:
                    source[key] = git_revision(path)
                    changed = True
            if changed:
                previous["updated_at"] = utc_now()
                self.save_manifest(previous)
            return previous
        created_at = utc_now()
        snapshot_dir = self.runs_dir / run_id / "artifacts"
        parameters = []
        for name in ("agent.yaml", "env.yaml"):
            source = run_dir / "params" / name
            if source.is_file():
                parameters.append(snapshot_artifact(source, snapshot_dir / "params"))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task": task,
            "experiment_id": experiment_id(run_dir, task),
            "experiment_label": experiment_label(run_dir),
            "experiment_kind": experiment_kind(run_dir),
            "stage": "experimental",
            "starred": False,
            "star_note": "",
            "created_at": created_at,
            "updated_at": utc_now(),
            "source_run_dir": str(run_dir),
            "has_exported_policy": policy is not None,
            "latest_iteration": iteration,
            "artifacts": {
                "checkpoint": snapshot_artifact(checkpoint, snapshot_dir),
                "policy": snapshot_artifact(policy, snapshot_dir),
                "parameters": parameters,
            },
            "source": {
                "lab": git_revision(LAB_ROOT),
                "upstream": git_revision(UPSTREAM),
                "runtime": git_revision(POLLEN_RUNTIME),
                "simulator": git_revision(POLLEN_SIMULATOR),
            },
            "evaluations": [],
            "promotion_history": [],
        }
        self.save_manifest(manifest)
        return manifest

    def discover(self, logs_root: Path, task: str | None = None) -> list[dict[str, Any]]:
        registered = []
        for run_dir in sorted(path for path in logs_root.glob("*/*") if path.is_dir()):
            inferred = infer_task(run_dir)
            if task and inferred != task:
                continue
            if any(run_dir.glob("model_*.pt")) or any(run_dir.glob("*.onnx")):
                registered.append(self.register(run_dir, task=inferred))
        self.render_dashboard()
        return registered

    def auto_promote_sim_champion(self, task: str) -> dict[str, Any] | None:
        """Promote the best verified improvement to simulation champion.

        Simulation evidence may select a sim-qualified champion automatically.
        Hardware-candidate and production remain explicit human-signoff stages.
        """
        candidates: list[
            tuple[tuple[float, ...], str, dict[str, Any], dict[str, Any]]
        ] = []
        manifests = [item for item in self.manifests() if item.get("task") == task and not item.get("archived")]
        for item in manifests:
            if item.get("experiment_kind") in {"smoke", "factory"}:
                continue
            evaluations = item.get("evaluations", [])
            if not evaluations:
                continue
            try:
                evaluation = read_json(Path(evaluations[-1]["path"]))
                score = evaluation.get("policy_bench_score", {})
                comparison = evaluation.get("baseline_comparison", {})
                beats_incumbent = comparison.get("improved", True)
                champion_eligible = score.get(
                    "simulation_champion_eligible", score.get("qualified")
                )
                if (
                    champion_eligible
                    and score.get("overall") is not None
                    and (task not in {"sprint", "race", "race5"} or beats_incumbent)
                ):
                    rank = (
                        race5_race_rank(score)
                        if task == "race5" else (float(score["overall"]),)
                    )
                    candidates.append((rank, item.get("created_at", ""), item, score))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        if not candidates:
            # Do not leave a stale champion badge pointing at a policy that
            # fails the current evaluation contract. Historical stage records
            # remain intact, but the active simulation-champion pointer clears.
            for item in manifests:
                if item.get("starred"):
                    item["starred"] = False
                    item["star_note"] = ""
                    item["updated_at"] = utc_now()
                    self.save_manifest(item)
            registry = read_json(self.registry_path)
            task_registry = registry.get("tasks", {}).get(task, {})
            if task_registry.pop("sim-qualified", None) is not None:
                registry["updated_at"] = utc_now()
                write_json(self.registry_path, registry)
            return None
        registry = read_json(self.registry_path)
        if task == "race5":
            incumbent_id = registry.get("tasks", {}).get(task, {}).get("sim-qualified")
            incumbent = next((row for row in candidates if row[2]["run_id"] == incumbent_id), None)
            if incumbent is not None:
                # The incumbent remains eligible, and a challenger may replace
                # it only by running a faster 100-foot heat without regressing
                # drift, heading, assistance, or more than three agility points.
                candidates = [
                    row for row in candidates
                    if row[2]["run_id"] == incumbent_id
                    or race5_advances_incumbent(row[3], incumbent[3])
                ]
        _, _, champion, _ = max(candidates, key=lambda row: (row[0], row[1]))
        champion_id = champion["run_id"]
        for item in manifests:
            changed = False
            should_star = item["run_id"] == champion_id
            if bool(item.get("starred")) != should_star:
                item["starred"] = should_star
                changed = True
            if should_star:
                item["star_note"] = (
                    "Automatically selected as fastest verified A-to-B and straight-line improvement"
                    if task == "race5"
                    else "Automatically selected as highest-scoring verified simulation improvement"
                )
                if item.get("stage") == "evaluated":
                    item["promotion_history"].append(
                        {
                            "from": "evaluated",
                            "to": "sim-qualified",
                            "at": utc_now(),
                            "automatic": True,
                            "note": (
                                "Faster verified 100-foot result without straight-line regression"
                                if task == "race5"
                                else "Highest-scoring verified improvement vs baseline"
                            ),
                        }
                    )
                    item["stage"] = "sim-qualified"
                    changed = True
            if changed:
                item["updated_at"] = utc_now()
                self.save_manifest(item)
        registry["tasks"].setdefault(task, {})["sim-qualified"] = champion_id
        registry["updated_at"] = utc_now()
        write_json(self.registry_path, registry)
        return self.load_manifest(champion_id)

    def select_sim_champion(self, run_id: str) -> dict[str, Any]:
        """Manually select a verified improved policy as simulation champion."""
        target = self.load_manifest(run_id)
        if target.get("experiment_kind") in {"smoke", "factory"}:
            raise SystemExit("Smoke checks and factory references cannot become the trained champion")
        evaluations = target.get("evaluations", [])
        if not evaluations:
            raise SystemExit("Only a scored, verified policy can become champion")
        evaluation = read_json(Path(evaluations[-1]["path"]))
        score = evaluation.get("policy_bench_score", {})
        champion_eligible = score.get(
            "simulation_champion_eligible", score.get("qualified")
        )
        comparison = evaluation.get("baseline_comparison", {})
        if not champion_eligible or (
            target["task"] in {"sprint", "race", "race5"}
            and not comparison.get("improved")
        ):
            raise SystemExit("This policy is not a verified improvement over the baseline")
        for item in self.manifests():
            if item.get("task") != target["task"]:
                continue
            item["starred"] = item["run_id"] == run_id
            if item["run_id"] == run_id:
                item["star_note"] = "Manually selected simulation champion"
                if item.get("stage") == "evaluated":
                    item["promotion_history"].append(
                        {
                            "from": "evaluated",
                            "to": "sim-qualified",
                            "at": utc_now(),
                            "automatic": False,
                            "note": "Selected from leaderboard",
                        }
                    )
                    item["stage"] = "sim-qualified"
            item["updated_at"] = utc_now()
            self.save_manifest(item)
            self.render_run_report(item["run_id"])
        registry = read_json(self.registry_path)
        registry["tasks"].setdefault(target["task"], {})["sim-qualified"] = run_id
        registry["updated_at"] = utc_now()
        write_json(self.registry_path, registry)
        self.render_dashboard()
        return self.load_manifest(run_id)

    def attach_evaluation(self, run_id: str, metrics_path: Path, suite: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        metrics_path = metrics_path.resolve()
        metrics = read_json(metrics_path)
        race_baseline_path = {
            "race": RACE_POLLEN_BASELINE_REPORT,
            "race5": (
                RACE5_POLLEN_LINE_BASELINE_REPORT
                if metrics.get("line_hold", {}).get("enabled")
                else RACE5_POLLEN_BASELINE_REPORT
            ),
        }.get(manifest["task"])
        if race_baseline_path and race_baseline_path.is_file():
            metrics["baseline_comparison"] = race_baseline_comparison(
                metrics, read_json(race_baseline_path), race_baseline_path
            )
        if manifest["task"] == "sprint" and SPRINT_BASELINE_REPORT.is_file():
            metrics["baseline_comparison"] = sprint_baseline_comparison(
                metrics, read_json(SPRINT_BASELINE_REPORT)
            )
        metrics["policy_bench_score"] = score_evaluation(metrics, manifest["task"])
        destination = self.runs_dir / run_id / "evaluations" / f"{suite}.json"
        write_json(destination, metrics)
        record = {
            "suite": suite,
            "created_at": utc_now(),
            "path": str(destination),
            "sha256": sha256(destination),
        }
        manifest["evaluations"] = [
            item for item in manifest.get("evaluations", []) if item.get("suite") != suite
        ] + [record]
        if manifest["stage"] == "experimental":
            manifest["stage"] = "evaluated"
            manifest["promotion_history"].append(
                {"from": "experimental", "to": "evaluated", "at": utc_now(), "automatic": True}
            )
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        champion = self.auto_promote_sim_champion(manifest["task"])
        if champion and champion["run_id"] != run_id:
            self.render_run_report(champion["run_id"])
        self.render_run_report(run_id)
        self.render_dashboard()
        return record

    def score(self, run_id: str, suite: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        evaluation = self.evaluation(manifest, suite)
        score = score_evaluation(evaluation, manifest["task"])
        evaluation["policy_bench_score"] = score
        for item in manifest["evaluations"]:
            if item["suite"] == suite:
                path = Path(item["path"])
                write_json(path, evaluation)
                item["sha256"] = sha256(path)
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_run_report(run_id)
        self.render_dashboard()
        return score

    def star(self, run_id: str, note: str = "") -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        for other in self.manifests():
            if other.get("task") == manifest["task"] and other.get("starred") and other["run_id"] != run_id:
                other["starred"] = False
                other["updated_at"] = utc_now()
                self.save_manifest(other)
        manifest["starred"] = True
        manifest["star_note"] = note
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_run_report(run_id)
        self.render_dashboard()
        return manifest

    def unstar(self, run_id: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        manifest["starred"] = False
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_run_report(run_id)
        self.render_dashboard()
        return manifest

    def archive(self, run_id: str, note: str = "") -> dict[str, Any]:
        """Hide a legacy run from the product view without deleting artifacts."""
        manifest = self.load_manifest(run_id)
        manifest["archived"] = True
        manifest["archived_at"] = utc_now()
        manifest["archive_note"] = note
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_dashboard()
        return manifest

    def unarchive(self, run_id: str) -> dict[str, Any]:
        """Restore an archived run to the normal product view."""
        manifest = self.load_manifest(run_id)
        manifest["archived"] = False
        manifest.pop("archived_at", None)
        manifest.pop("archive_note", None)
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_dashboard()
        return manifest

    def evaluate(self, run_id: str, suite: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        policy = manifest["artifacts"].get("policy")
        if not policy:
            raise SystemExit(f"Run {run_id} has no exported ONNX policy")
        spec = task_spec(str(manifest["task"]))
        if not spec or not spec.get("evaluator"):
            raise SystemExit(f"No evaluator is registered for task {manifest['task']!r}")
        policy_path = Path(policy["path"])
        if not policy_path.is_file() or sha256(policy_path) != policy["sha256"]:
            raise SystemExit(f"Policy snapshot is missing or corrupt: {policy_path}")
        output = self.runs_dir / run_id / "evaluations" / f"{suite}.json"
        evaluator_config = spec["evaluator"]
        evaluator = LAB_ROOT / evaluator_config["script"]
        uv = LAB_ROOT / ".tools" / "uv" / "bin" / "uv"
        if not uv.is_file():
            raise SystemExit("DuckLab uv environment is missing; run ./scripts/bootstrap.sh")
        command = [str(uv), "run", str(evaluator), str(policy_path)]
        command.extend(str(value) for value in evaluator_config.get("arguments", []))
        command.extend(["--output", str(output)])
        subprocess.run(
            command,
            cwd=UPSTREAM,
            check=True,
        )
        return self.attach_evaluation(run_id, output, suite)

    def metrics(self, run_id: str) -> Path:
        manifest = self.load_manifest(run_id)
        source_dir = Path(manifest["source_run_dir"])
        reader = load_metrics_module()
        data = reader.collect_metrics(source_dir)
        output = self.runs_dir / run_id / "metrics.json"
        write_json(output, data)
        manifest["metrics"] = {"path": str(output), "sha256": sha256(output), "scalar_count": data["scalar_count"]}
        manifest["updated_at"] = utc_now()
        self.save_manifest(manifest)
        self.render_run_report(run_id)
        self.render_dashboard()
        return output

    def evaluation(self, manifest: dict[str, Any], suite: str) -> dict[str, Any]:
        matches = [item for item in manifest.get("evaluations", []) if item["suite"] == suite]
        if not matches:
            raise SystemExit(f"Run {manifest['run_id']} has no {suite!r} evaluation")
        return read_json(Path(matches[-1]["path"]))

    def compare(self, candidate_id: str, baseline_id: str, suite: str) -> dict[str, Any]:
        candidate = self.load_manifest(candidate_id)
        baseline = self.load_manifest(baseline_id)
        if candidate["task"] != baseline["task"]:
            raise SystemExit("Cannot compare runs from different tasks")
        candidate_evaluation = self.evaluation(candidate, suite)
        baseline_evaluation = self.evaluation(baseline, suite)
        candidate_metrics = flatten_numbers(candidate_evaluation.get("phases", candidate_evaluation))
        baseline_metrics = flatten_numbers(baseline_evaluation.get("phases", baseline_evaluation))
        rows = []
        for key in sorted(candidate_metrics.keys() & baseline_metrics.keys()):
            before = baseline_metrics[key]
            after = candidate_metrics[key]
            rows.append({"metric": key, "baseline": before, "candidate": after, "delta": after - before})
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "task": candidate["task"],
            "suite": suite,
            "candidate": candidate_id,
            "baseline": baseline_id,
            "metrics": rows,
            "note": "Deltas are descriptive; promotion requires human review because metric direction and safety significance vary.",
        }
        output = self.runs_dir / candidate_id / "comparisons" / f"vs-{baseline_id}-{suite}.json"
        write_json(output, result)
        render_comparison_html(result, output.with_suffix(".html"))
        self.render_run_report(candidate_id)
        self.render_dashboard()
        return result

    def promote(
        self,
        run_id: str,
        stage: str,
        approved_by: str,
        note: str,
        hardware_signoff: bool,
    ) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        current = manifest["stage"]
        if stage not in STAGES:
            raise SystemExit(f"Unknown stage {stage!r}; choose from {', '.join(STAGES)}")
        if STAGES.index(stage) != STAGES.index(current) + 1:
            raise SystemExit(f"Promotion must be one stage at a time: {current} -> {STAGES[STAGES.index(current) + 1] if current != STAGES[-1] else 'none'}")
        if stage != "evaluated" and not manifest.get("evaluations"):
            raise SystemExit("A policy must have an attached evaluation before promotion")
        if stage in {"hardware-candidate", "production"} and not hardware_signoff:
            raise SystemExit(f"Promotion to {stage} requires --hardware-signoff")
        previous = current
        manifest["stage"] = stage
        manifest["updated_at"] = utc_now()
        manifest["promotion_history"].append(
            {"from": previous, "to": stage, "at": utc_now(), "approved_by": approved_by, "note": note}
        )
        self.save_manifest(manifest)
        registry = read_json(self.registry_path)
        task_registry = registry["tasks"].setdefault(manifest["task"], {})
        task_registry[stage] = run_id
        registry["updated_at"] = utc_now()
        write_json(self.registry_path, registry)
        self.render_run_report(run_id)
        self.render_dashboard()
        return manifest

    def resolve(self, task: str, stage: str, artifact: str) -> Path:
        registry = read_json(self.registry_path)
        run_id = registry.get("tasks", {}).get(task, {}).get(stage)
        if not run_id:
            raise SystemExit(f"No {stage} policy registered for task {task}")
        manifest = self.load_manifest(run_id)
        record = manifest["artifacts"].get(artifact)
        if not record:
            raise SystemExit(f"Run {run_id} has no {artifact} artifact")
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"Registered artifact is missing or its hash changed: {path}")
        return path

    def manifests(self) -> list[dict[str, Any]]:
        return [read_json(path) for path in sorted(self.runs_dir.glob("*/manifest.json"))]

    def render_run_report(self, run_id: str) -> Path:
        manifest = self.load_manifest(run_id)
        sprint_summary = ""
        if manifest.get("task") in {"sprint", "race", "race5"} and manifest.get("evaluations"):
            try:
                evaluation_data = read_json(Path(manifest["evaluations"][-1]["path"]))
                sprint_summary = (
                    race_report_summary(evaluation_data)
                    if manifest.get("task") in {"race", "race5"}
                    else sprint_report_summary(evaluation_data)
                )
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                sprint_summary = ""
        rows = "".join(
            f"<tr><td>{html.escape(item['suite'])}</td><td>{html.escape(item['created_at'])}</td>"
            f"<td><a href='{html.escape(os.path.relpath(item['path'], self.runs_dir / run_id))}'>JSON</a></td></tr>"
            for item in manifest.get("evaluations", [])
        ) or "<tr><td colspan='3'>No evaluations attached</td></tr>"
        comparison_rows = "".join(
            f"<li><a href='{html.escape(os.path.relpath(path, self.runs_dir / run_id))}'>{html.escape(path.stem)}</a></li>"
            for path in sorted((self.runs_dir / run_id / "comparisons").glob("*.html"))
        )
        if not comparison_rows:
            comparison_rows = (
                "<li>The official Pollen comparison is included in the Sprint-v1 verdict above.</li>"
                if sprint_summary else "<li>No comparisons generated</li>"
            )
        charts = f"<p>No TensorBoard curves ingested yet. Run <code>policy-bench.sh metrics {html.escape(run_id)}</code>.</p>"
        metrics_record = manifest.get("metrics")
        if metrics_record and Path(metrics_record["path"]).is_file():
            metrics_data = read_json(Path(metrics_record["path"]))
            preferred = [
                "Train/mean_reward", "Train/mean_episode_length", "Perf/total_fps",
                "Loss/value", "Loss/surrogate", "Episode_Reward/upright",
                "Episode_Reward/grounded", "Metrics/twist/error_vel_xy",
            ]
            charts = "".join(
                metric_svg(tag, metrics_data["scalars"][tag])
                for tag in preferred if tag in metrics_data.get("scalars", {})
            ) or "<p>No preferred scalar curves found.</p>"
        body = page(
            manifest["run_id"],
            f"<p><span class='badge'>{html.escape(manifest['stage'])}</span> Task: {html.escape(manifest['task'])} · "
            f"{'★ Starred' if manifest.get('starred') else 'Not starred'}</p>"
            f"<p>{'Factory policy' if manifest.get('experiment_kind') == 'factory' else 'Iteration: ' + str(manifest.get('latest_iteration'))} · ONNX export: {manifest.get('has_exported_policy')}</p>"
            f"<p class='mono'>{html.escape(manifest['source_run_dir'])}</p>"
            f"{sprint_summary}"
            f"<h2>Evaluations</h2><table><tr><th>Suite</th><th>Created</th><th>Data</th></tr>{rows}</table>"
            f"<h2>Comparisons</h2><ul>{comparison_rows}</ul>"
            f"<h2>Training curves</h2>{charts}"
            f"<h2>Manifest</h2><pre>{html.escape(json.dumps(manifest, indent=2, sort_keys=True))}</pre>",
        )
        output = self.runs_dir / run_id / "report.html"
        output.write_text(body)
        return output

    def render_dashboard(self, active_experiments: set[str] | None = None) -> Path:
        registry = read_json(self.registry_path)
        agent_receipts = load_agent_runs(self.state_dir / "agent-runs")
        agent_cards = []
        for receipt in agent_receipts[:12]:
            status = str(receipt.get("status", "unknown"))
            metrics = receipt.get("metrics", [])[:4]
            metric_html = "".join(
                "<span><small>" + html.escape(str(metric.get("name", "Metric"))) + "</small><strong>"
                + html.escape(str(metric.get("value", "—"))) + " "
                + html.escape(str(metric.get("unit", ""))) + "</strong></span>"
                for metric in metrics if isinstance(metric, dict)
            )
            actions = "".join(
                f"<a class='button-link {'primary-action' if action.get('kind') == 'simulation' else ''}' "
                f"href='{html.escape(str(action.get('url', '')))}' target='_blank' rel='noopener'>"
                f"{html.escape(str(action.get('label', 'Open')))}</a>"
                for action in receipt.get("actions", []) if isinstance(action, dict)
            )
            progress = receipt.get("progress") or {}
            progress_copy = ""
            if progress.get("current") is not None:
                progress_copy = (
                    f" · {html.escape(str(progress['current']))}"
                    + (f" / {html.escape(str(progress['total']))}" if progress.get("total") is not None else "")
                    + f" {html.escape(str(progress.get('unit', '')))}"
                )
            agent_cards.append(
                "<article class='finished-card'><div class='finished-card-top'><div>"
                f"<h3>{html.escape(str(receipt['title']))}</h3><div class='run-tags'>"
                f"<span class='pill'>{html.escape(str(receipt['project']))}</span>"
                f"<span class='complete-tag'>{html.escape(status.title())}{progress_copy}</span></div></div>"
                f"<div class='launch-cluster'>{actions}</div></div>"
                f"<p>{html.escape(str(receipt['goal']))}</p>"
                f"<div class='run-stats'>{metric_html}</div></article>"
            )
        agent_runs_html = (
            "<section id='agent-runs' class='experiment-library' data-revision='"
            + html.escape("|".join(
                f"{receipt.get('run_id', '')}:{receipt.get('updated_at', '')}"
                for receipt in agent_receipts
            ))
            + "'><div class='section-heading'><div>"
            "<p class='eyebrow'>AGENTIC RL WORKSPACE</p><h2>Agent-planned experiments</h2></div>"
            "<p>Codex owns the changing research code; the dashboard consumes a generic run receipt.</p></div>"
            + ("<div class='finished-grid'>" + "".join(agent_cards) + "</div>" if agent_cards else
               "<div class='empty-state'>No agent receipt published yet. Run scripts/publish-agent-run.sh with the example receipt.</div>")
            + "</section>"
        )
        # Archiving is presentation-only: immutable artifacts, evaluations,
        # and provenance stay available by run ID for recovery.
        all_manifests = sorted(
            (item for item in self.manifests() if not item.get("archived", False)),
            key=lambda item: item["created_at"],
            reverse=True,
        )
        # Smoke launches validate wiring for a few iterations; they are not
        # user-facing training jobs and should never inflate the run count.
        real_tasks = {item.get("task") for item in all_manifests if item.get("experiment_kind", "training") != "smoke"}
        # Keep a task visible when it only has a smoke baseline (for example
        # the original walking run), but never add smoke rows beside a real
        # training job for the same task.
        manifests = [item for item in all_manifests if item.get("experiment_kind", "training") != "smoke" or item.get("task") not in real_tasks]
        def experiment_key(manifest: dict[str, Any]) -> str:
            # A run name is the user-facing job identity. Resume attempts may
            # create a new timestamped source directory, but must remain one
            # training job in the UI.
            label = manifest.get("experiment_label") or manifest.get("source_run_dir", "").rsplit("/", 1)[-1]
            label = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", label)
            if manifest.get("experiment_kind") == "smoke":
                return f"smoke:{manifest.get('task', 'unknown')}"
            if label:
                return f"{manifest.get('task', 'unknown')}:{label}"
            return manifest.get("experiment_id") or f"{manifest.get('task', 'unknown')}:{manifest.get('source_run_dir', manifest['run_id'])}"

        active_experiments = active_training_experiments() if active_experiments is None else active_experiments
        grouped: dict[str, list[dict[str, Any]]] = {}
        for manifest in manifests:
            grouped.setdefault(experiment_key(manifest), []).append(manifest)
        finished_rows = []
        active_rows = []
        for group in grouped.values():
            newest = group[0]
            label = newest.get("experiment_label") or newest["source_run_dir"].rsplit("/", 1)[-1]
            label = display_experiment_label(str(newest.get("task", "unknown")), label)
            state = "active" if newest.get("experiment_kind") == "training" and label in active_experiments else "finished/snapshot"
            # Retries and discovery passes can snapshot the same iteration
            # more than once. That is useful provenance, but it is not another
            # user-facing saved model.
            distinct_versions: dict[int | None, dict[str, Any]] = {}
            for manifest in sorted(group, key=lambda item: item.get("created_at", ""), reverse=True):
                distinct_versions.setdefault(manifest.get("latest_iteration"), manifest)
            snapshots = []
            run_evaluations: dict[str, dict[str, Any]] = {}
            for manifest in sorted(distinct_versions.values(), key=lambda item: item.get("latest_iteration") or -1, reverse=True):
                score = None
                saved_metric_label = "Score"
                for evaluation in manifest.get("evaluations", []):
                    try:
                        evaluation_data = read_json(Path(evaluation["path"]))
                        run_evaluations[manifest["run_id"]] = evaluation_data
                        calculated = score_evaluation(evaluation_data, manifest["task"])
                        if manifest["task"] == "race5":
                            speed = calculated.get("performance", {}).get("sustained_speed_mph")
                            score = f"{float(speed):.2f} mph" if speed is not None else None
                            saved_metric_label = "Sustained speed"
                        else:
                            score = evaluation_data.get("policy_bench_score", {}).get("overall")
                            if score is None:
                                score = calculated.get("overall")
                    except (OSError, KeyError, TypeError, json.JSONDecodeError):
                        pass
                version_name = (
                    "Factory release"
                    if manifest.get("experiment_kind") == "factory"
                    else f"Iteration {manifest['latest_iteration']}"
                    if manifest.get("latest_iteration") is not None
                    else "Composite ONNX"
                )
                snapshots.append(
                    "<article class='saved-model'>"
                    "<div class='saved-model-title'>"
                    f"<strong>{html.escape(version_name)}</strong>"
                    f"<span class='stage-badge'>{html.escape(manifest['stage'])}</span></div>"
                    "<div class='saved-model-stats'>"
                    f"<span><small>{saved_metric_label}</small>{score if score is not None else '—'}</span>"
                    f"<span><small>Evaluations</small>{len(manifest.get('evaluations', []))}</span>"
                    f"<span><small>Shortlisted</small>{'Yes ★' if manifest.get('starred') else 'No'}</span></div>"
                    "<div class='saved-model-actions'>"
                    f"<a class='text-action' href='runs/{html.escape(manifest['run_id'])}/report.html'>View details</a>"
                    f"<button class='star secondary' data-run-id='{html.escape(manifest['run_id'])}'>{'Remove star' if manifest.get('starred') else '★ Star model'}</button>"
                    "</div></article>"
                )
            latest = max(
                distinct_versions.values(),
                key=lambda item: (item.get("latest_iteration") or -1, item.get("created_at", "")),
            )
            is_factory = latest.get("experiment_kind") == "factory"
            result_banner = (
                sprint_dashboard_verdict(run_evaluations.get(latest["run_id"]))
                if latest.get("task") == "sprint" else ""
            )
            if latest.get("task") in {"race", "race5"}:
                result_banner = race_dashboard_verdict(run_evaluations.get(latest["run_id"]))
            latest_iteration = max((item.get("latest_iteration") for item in group if item.get("latest_iteration") is not None), default=None)
            latest_iteration_label = (
                f"{latest_iteration:,}" if latest_iteration is not None else "Composite ONNX"
            )
            if state == "active":
                active_rows.append(
                    "<article class='run-card'>"
                    f"<div><span class='status-dot'></span><strong>{html.escape(label)}</strong> "
                    f"<span class='pill'>{html.escape(display_task_name(newest['task']))}</span>"
                    f"<p class='muted active-checkpoint-copy'>4,096 environments train headless · view 6 evaluation robots using saved checkpoint {latest.get('latest_iteration')}</p></div>"
                    f"<button class='watch-training' data-run-id='{html.escape(latest['run_id'])}' "
                    f"data-label='Watch checkpoint {latest.get('latest_iteration')}'>Watch checkpoint {latest.get('latest_iteration')}</button></article>"
                )
            primary_action = (
                f"<a class='button-link primary-action' href='{FACTORY_ARENA_URL}' target='_blank'>Open simulator</a>"
                if is_factory
                else f"<button class='play primary-action' data-run-id='{html.escape(latest['run_id'])}' data-label='Open simulator'>Open simulator</button>"
            )
            row = (
                "<article class='finished-card'>"
                "<div class='finished-card-top'><div>"
                f"<h3>{html.escape(label)}</h3>"
                f"<div class='run-tags'><span class='pill'>{html.escape(display_task_name(newest['task']))}</span>"
                f"<span class='kind-tag'>{'Factory baseline' if is_factory else ('Smoke check' if newest.get('experiment_kind') == 'smoke' else 'Training run')}</span>"
                "<span class='complete-tag'>Finished</span></div></div>"
                "<div class='launch-cluster'>"
                f"{primary_action}"
                f"<button class='deployment secondary' data-run-id='{html.escape(latest['run_id'])}' "
                f"{'disabled ' if not latest.get('has_exported_policy') or not task_has_evaluator(latest.get('task')) else ''}"
                "title='Score the exported ONNX in Pollen CPU MuJoCo'>Evaluate</button></div></div>"
                f"{result_banner}"
                "<div class='run-stats'>"
                f"<div><small>{'Source' if is_factory else 'Latest iteration'}</small><strong>{'Pollen official' if is_factory else latest_iteration_label}</strong></div>"
                f"<div><small>Saved models</small><strong>{len(distinct_versions)}</strong></div>"
                f"<div><small>Skill</small><strong>{html.escape(display_task_name(newest['task']))}</strong></div>"
                f"<div><small>Upstream</small><strong>{html.escape(str(latest.get('source', {}).get('upstream', {}).get('commit') or 'unknown')[:8])}</strong></div></div>"
                "<details class='saved-dropdown'><summary><span>Saved models</span>"
                f"<span class='summary-count'>{len(distinct_versions)}</span><span class='chevron'>⌄</span></summary>"
                "<div class='saved-list'>"
                + "".join(snapshots)
                + "</div></details></article>"
            )
            if state != "active":
                finished_rows.append(row)

        # The primary view is a scoreboard, not an experiment database. Keep
        # one best scored snapshot per training job; the full immutable
        # history remains available in the collapsed archive below.
        scored_entries: list[dict[str, Any]] = []
        best_by_experiment: dict[str, dict[str, Any]] = {}
        # A scored smoke check may appear as a clearly labelled diagnostic,
        # even though it stays hidden from the main experiment library and can
        # never be promoted. This keeps the newest evidence visible.
        for manifest in all_manifests:
            if not manifest.get("evaluations"):
                continue
            try:
                evaluation = read_json(Path(manifest["evaluations"][-1]["path"]))
                score = score_evaluation(evaluation, manifest["task"])
                if score.get("overall") is None:
                    continue
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            label = manifest.get("experiment_label") or Path(manifest["source_run_dir"]).name
            entry = {
                "manifest": manifest,
                "evaluation": evaluation,
                "score": score,
                "label": display_experiment_label(manifest["task"], str(label)),
                "synthetic": False,
                "baseline": False,
                "incumbent": False,
            }
            key = experiment_key(manifest)
            previous = best_by_experiment.get(key)
            rank = (
                race5_race_rank(score)
                if manifest["task"] == "race5" else (float(score["overall"]),)
            )
            previous_rank = (
                race5_race_rank(previous["score"])
                if previous is not None and manifest["task"] == "race5"
                else (float(previous["score"]["overall"]),) if previous is not None else None
            )
            if previous is None or rank > previous_rank:
                best_by_experiment[key] = entry
        scored_entries.extend(best_by_experiment.values())
        # Skating remains the flagship decision view. Other projects are
        # visible in agent receipts and the experiment library without forcing
        # their unlike metrics into a speed table.
        race5_entries = [item for item in scored_entries if item["manifest"]["task"] == "race5"]
        focus_task = "race5" if race5_entries else (
            max(scored_entries, key=lambda item: item["manifest"].get("created_at", ""))["manifest"]["task"]
            if scored_entries else "sprint"
        )
        leaderboard = [
            item for item in scored_entries
            if item["manifest"]["task"] == focus_task
            and item["manifest"].get("experiment_kind") not in {"smoke", "factory"}
        ]
        fastest_speed_entry = None
        speed_scout_entry = None
        if focus_task == "race5":
            speed_candidates = [
                item for item in leaderboard
                if item["score"].get("performance", {}).get("top_speed_mph") is not None
            ]
            if speed_candidates:
                fastest_speed_entry = max(
                    speed_candidates,
                    key=lambda item: float(item["score"]["performance"]["top_speed_mph"]),
                )
            # Discovery policies are intentionally excluded from the official
            # scoreboard.  They remain directly playable as a separate,
            # physics-labelled research reference.
            speed_scout_entry = next(
                (
                    item for item in all_manifests
                    if item.get("task") == "race5"
                    and item.get("arena_preview", {}).get("profile") == "frictionless-speed-scout"
                ),
                None,
            )

        baseline_path = {
            "sprint": SPRINT_BASELINE_REPORT,
            "race": RACE_POLLEN_BASELINE_REPORT,
            "race5": RACE5_POLLEN_LINE_BASELINE_REPORT,
        }.get(focus_task)
        baseline_entry = None
        if baseline_path and baseline_path.is_file():
            try:
                baseline_evaluation = read_json(baseline_path)
                baseline_score = score_evaluation(baseline_evaluation, focus_task)
                if baseline_score.get("overall") is not None:
                    baseline_entry = {
                        "manifest": None,
                        "evaluation": baseline_evaluation,
                        "score": baseline_score,
                        "label": (
                            "Official Pollen roller + line hold"
                            if baseline_evaluation.get("line_hold", {}).get("enabled")
                            else "Official Pollen roller"
                        ),
                        "synthetic": True,
                        "baseline": True,
                        "incumbent": False,
                    }
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        incumbent_path = {
            "race": RACE_TRAINED_INCUMBENT_REPORT,
            "race5": RACE5_TRAINED_INCUMBENT_REPORT,
        }.get(focus_task)
        incumbent_entry = None
        if incumbent_path and incumbent_path.is_file():
            try:
                incumbent_evaluation = read_json(incumbent_path)
                incumbent_score = score_evaluation(incumbent_evaluation, focus_task)
                if incumbent_score.get("overall") is not None:
                    incumbent_entry = {
                        "manifest": None,
                        "evaluation": incumbent_evaluation,
                        "score": incumbent_score,
                        "label": "Sprint-v3 trained reference",
                        "synthetic": True,
                        "baseline": False,
                        "incumbent": True,
                    }
                    leaderboard.append(incumbent_entry)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        # Race5's primary question is now deliberately simple: which saved
        # policy is fastest under the official physics?  Qualification remains
        # visible on every card and controls auto-promotion, but must never
        # bury a faster checkpoint beneath an older, safer one.
        if focus_task == "race5":
            leaderboard.sort(
                key=lambda item: (
                    float(item["score"].get("performance", {}).get("top_speed_mph", 0.0)),
                    float(item["score"].get("performance", {}).get("sustained_speed_mph", 0.0)),
                    -float(item["score"].get("performance", {}).get("long_run_max_drift_ft", float("inf"))),
                ),
                reverse=True,
            )
        else:
            leaderboard.sort(key=lambda item: float(item["score"]["overall"]), reverse=True)
        leaderboard = leaderboard[:3]
        leader_score = (
            float(leaderboard[0]["score"].get("performance", {}).get("sustained_speed_mph", 0.0))
            if leaderboard and focus_task == "race5"
            else float(leaderboard[0]["score"]["overall"]) if leaderboard else 0.0
        )
        leader_time = (
            leaderboard[0]["score"].get("performance", {}).get("elapsed_time_100ft_s")
            if leaderboard and focus_task == "race5" else None
        )
        champion_id = registry.get("tasks", {}).get(focus_task, {}).get("sim-qualified")
        king_policy = next(
            (
                item for item in scored_entries
                if item.get("manifest") and item["manifest"].get("run_id") == champion_id
            ),
            None,
        )
        podium_cards = []
        medals = ("🥇", "🥈", "🥉")
        for index in range(3):
            if index >= len(leaderboard):
                podium_cards.append(
                    f"<article class='podium-card empty'><div class='podium-rank'>{medals[index]} #{index + 1}</div>"
                    "<h3>Awaiting a scored challenger</h3><p>Run and evaluate another policy to fill this place.</p></article>"
                )
                continue
            entry = leaderboard[index]
            score = entry["score"]
            performance = score.get("performance", {})
            overall = float(score["overall"])
            qualified = bool(score.get("qualified"))
            manifest = entry["manifest"]
            is_champion = bool(manifest and manifest["run_id"] == champion_id)
            speed = performance.get("sustained_speed_mph")
            top_speed = performance.get("top_speed_mph")
            acceleration = performance.get("acceleration_first_second_mps2")
            agility = performance.get("agility_score")
            control_stack = str(performance.get("control_stack", "Raw policy"))
            if performance.get("line_hold_enabled"):
                control_stack += f" · {float(performance.get('auto_steering_percent', 0.0)):.0f}% auto steering"
            ranking_value = float(speed or 0.0) if focus_task == "race5" else overall
            delta = ranking_value - leader_score
            speed_copy = (
                f"{float(speed):.2f} mph sustained · {float(top_speed):.2f} mph top"
                if speed is not None and top_speed is not None else "Speed rerun needed"
            )
            acceleration_copy = f"{float(acceleration) * MPS_TO_MPH:+.2f} mph/s" if acceleration is not None else "rerun needed"
            agility_copy = f"{float(agility):.0f}/100" if agility is not None else "—"
            if focus_task == "race5":
                if performance.get("finished_100ft") and performance.get("elapsed_time_100ft_s") is not None:
                    outcome = (
                        f"100 ft in {float(performance['elapsed_time_100ft_s']):.2f} s · "
                        f"{float(performance.get('trap_speed_100ft_mph', 0.0)):.2f} mph trap · {control_stack}"
                    )
                else:
                    remaining = float(performance.get("distance_remaining_100ft_ft", 100.0))
                    outcome = f"{speed_copy} · did not reach B ({remaining:.1f} ft short) · {control_stack}"
            elif focus_task == "race":
                race = entry["evaluation"].get("phases", {}).get("race", {})
                outcome = (
                    f"5 m in {float(race['finish_time_5m_s']):.2f} s"
                    if race.get("finished_5m") else f"{float(race.get('forward_progress_m', 0.0)):.2f} / 5.00 m"
                )
            else:
                outcome = speed_copy
            actions = ""
            if manifest:
                actions = (
                    f"<a class='text-action' href='runs/{html.escape(manifest['run_id'])}/report.html'>View proof</a>"
                    f"<button class='play secondary' data-run-id='{html.escape(manifest['run_id'])}' data-label='Try in arena'>Try in arena</button>"
                )
                champion_eligible = bool(score.get("simulation_champion_eligible", qualified))
                if champion_eligible and not is_champion and manifest.get("experiment_kind") not in {"smoke", "factory"}:
                    actions += f"<button class='champion-select' data-run-id='{html.escape(manifest['run_id'])}'>Make champion</button>"
            if entry.get("incumbent"):
                status = "CURRENT TRAINED REFERENCE"
            elif manifest and manifest.get("experiment_kind") == "smoke":
                status = "SMOKE CHECK · NOT PROMOTABLE"
            else:
                if is_champion:
                    status = "AUTO-PROMOTED CHAMPION"
                elif qualified:
                    status = "CONTROL + SKILLS PASS"
                else:
                    failed = [
                        name.replace("_", " ").upper()
                        for name, gate in score.get("qualification_gates", {}).items()
                        if not gate.get("passed")
                    ]
                    status = f"CONTROL FAIL · {failed[0]}" if focus_task == "race5" and failed else "CONTROL CHECKS FAIL"
            status_good = qualified or is_champion or entry.get("incumbent")
            primary_score = (
                f"<div class='podium-score'><strong>{float(top_speed):.2f}</strong><span>mph verified top · 0.5 s</span></div>"
                if focus_task == "race5" and top_speed is not None
                else f"<div class='podium-score'><strong>{overall:.2f}</strong><span>/ 100</span></div>"
            )
            elapsed = performance.get("elapsed_time_100ft_s")
            delta_copy = (
                "FASTEST VERIFIED SPEED" if not qualified else "FASTEST CONTROL-QUALIFIED A → B"
                if index == 0
                else f"{float(elapsed) - float(leader_time):.2f} s behind"
                if elapsed is not None and leader_time is not None
                else f"{abs(delta):.2f} mph behind"
            ) if focus_task == "race5" else (
                "LEADER" if index == 0 else f"{abs(delta):.2f} points behind"
            )
            podium_cards.append(
                f"<article class='podium-card {'winner' if index == 0 else ''}'>"
                f"<div class='podium-rank'>{medals[index]} #{index + 1}</div>"
                f"<span class='podium-status {'good' if status_good else 'bad'}'>{status}</span>"
                f"<h3>{html.escape(entry['label'])}</h3>"
                f"{primary_score}<p class='podium-delta'>{html.escape(delta_copy)}</p>"
                f"<p class='podium-outcome'>{html.escape(outcome)}</p>"
                "<div class='athletic-stats'>"
                f"<span><small>Acceleration</small><strong>{html.escape(acceleration_copy)}</strong></span>"
                f"<span><small>Agility</small><strong>{html.escape(agility_copy)}</strong></span></div>"
                f"<div class='podium-actions'>{actions}</div></article>"
            )
        # Pollen head-to-head is always about the all-around champion, never
        # whichever speed-only experiment happens to be first in the challenger
        # list.  This keeps “king” and “fastest” unambiguous.
        best_policy = king_policy or next((item for item in leaderboard if not item["synthetic"]), None)
        current_king_html = ""
        if focus_task == "race5" and king_policy:
            king_manifest = king_policy["manifest"]
            king_performance = king_policy["score"].get("performance", {})
            king_run_id = html.escape(king_manifest["run_id"])
            current_king_html = (
                "<div class='baseline-reference race5-headtohead'><div class='baseline-duel-head'><div>"
                "<small>CURRENT ALL-AROUND KING · FULL POLLEN GATES PASS</small>"
                f"<strong>{html.escape(king_policy['label'])}</strong>"
                "<span>This is the active benchmark donor and the only policy allowed to represent DuckLab versus Pollen today.</span></div>"
                "<div class='baseline-improvement good'><small>CONTROL-SAFE RACE5</small>"
                f"<strong>{float(king_performance.get('top_speed_mph', 0.0)):.2f} mph</strong>"
                f"<span>{float(king_performance.get('sustained_speed_mph', 0.0)):.2f} mph sustained · {float(king_performance.get('elapsed_time_100ft_s', 0.0)):.2f} s / 100 ft</span></div></div>"
                f"<p class='baseline-method'><button class='play primary-action' data-run-id='{king_run_id}' data-label='Try all-around king'>Try all-around king in arena</button> "
                f"<a class='text-action' href='runs/{king_run_id}/report.html'>View full proof</a></p></div>"
            )
        baseline_reference_html = ""
        if baseline_entry:
            pollen_score = float(baseline_entry["score"]["overall"])
            pollen_performance = baseline_entry["score"].get("performance", {})
            pollen_sustained = pollen_performance.get("sustained_speed_mph")
            pollen_top = pollen_performance.get("top_speed_mph")
            speed_reference = (
                f"{float(pollen_sustained):.2f} mph sustained · {float(pollen_top):.2f} mph top"
                if pollen_sustained is not None and pollen_top is not None else "speed evidence unavailable"
            )
            if focus_task in {"race", "race5"} and SPRINT_BASELINE_REPORT.is_file():
                try:
                    known_pollen = score_evaluation(read_json(SPRINT_BASELINE_REPORT), "sprint").get("performance", {})
                    known_sustained = known_pollen.get("sustained_speed_mph")
                    if known_sustained is not None:
                        speed_reference = (
                            f"Known Pollen: {float(known_sustained):.2f} mph sustained at its 1.23 mph command · "
                            f"this exact heat: {speed_reference}"
                        )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            if focus_task == "race5" and best_policy:
                champion_performance = best_policy["score"].get("performance", {})
                metric_specs = (
                    ("Verified top", "top_speed_mph", "mph", 2, True),
                    ("Sustained speed", "sustained_speed_mph", "mph", 2, True),
                    ("100 ft time", "elapsed_time_100ft_s", "s", 2, False),
                    ("Trap speed", "trap_speed_100ft_mph", "mph", 2, True),
                    ("Launch acceleration", "acceleration_first_second_mps2", "mph/s", 2, True),
                    ("Max straight-line drift", "long_run_max_drift_ft", "ft", 2, False),
                    ("Max heading error", "long_run_max_heading_error_deg", "°", 2, False),
                    ("Agility retained", "agility_score", "/100", 0, True),
                    ("Automatic steering", "auto_steering_percent", "%", 1, False),
                )
                comparison_cards = []
                comparison_passes = []
                for label, key, unit, digits, higher_is_better in metric_specs:
                    champion_value = champion_performance.get(key)
                    pollen_value = pollen_performance.get(key)
                    if champion_value is None or pollen_value is None:
                        continue
                    champion_value = float(champion_value)
                    pollen_value = float(pollen_value)
                    if key == "acceleration_first_second_mps2":
                        champion_value *= MPS_TO_MPH
                        pollen_value *= MPS_TO_MPH
                    improved = (
                        champion_value > pollen_value
                        if higher_is_better else champion_value < pollen_value
                    )
                    comparison_passes.append(improved)
                    relative_delta = (
                        100.0 * (champion_value - pollen_value) / abs(pollen_value)
                        if abs(pollen_value) > 1.0e-9 else 0.0
                    )
                    value_copy = f"{champion_value:.{digits}f}"
                    pollen_copy = f"{pollen_value:.{digits}f}"
                    comparison_cards.append(
                        f"<div class='baseline-stat {'improved' if improved else 'regressed'}'>"
                        f"<small>{html.escape(label)}</small><strong>{value_copy}<em>{html.escape(unit)}</em></strong>"
                        f"<span>Pollen {pollen_copy} · {relative_delta:+.1f}%</span></div>"
                    )
                all_shown_improved = bool(comparison_passes) and all(comparison_passes)
                improvement_count = sum(comparison_passes)
                champion_label = html.escape(best_policy["label"])
                baseline_reference_html = (
                    "<div class='baseline-reference race5-headtohead'>"
                    "<div class='baseline-duel-head'><div><small>IMMUTABLE TRUE BASELINE</small>"
                    "<strong>Pollen official roller</strong><span>Same policy, physics profile, course, and line controller.</span></div>"
                    f"<div class='baseline-improvement {'good' if all_shown_improved else 'mixed'}'>"
                    f"<small>{html.escape(champion_label)} VS POLLEN</small>"
                    f"<strong>{'IMPROVED' if all_shown_improved else 'MIXED RESULT'}</strong>"
                    f"<span>{improvement_count}/{len(comparison_passes)} measured race dimensions improved</span></div></div>"
                    f"<div class='baseline-comparison-grid'>{''.join(comparison_cards)}</div>"
                    "<p class='baseline-method'>Verified simulation comparison · top speed is the best 0.5 s rolling horizontal chassis speed · lower is better for time, drift, heading error, and steering assistance.</p>"
                    "</div>"
                )
            else:
                baseline_reference_html = (
                    "<div class='baseline-reference'><div><small>IMMUTABLE TRUE BASELINE</small>"
                    "<strong>Pollen official roller</strong><span>Every run is compared with this exact policy.</span></div>"
                    + (
                        f"<div class='baseline-numbers'><strong>{float(pollen_sustained):.2f}<small>mph sustained</small></strong>"
                        if focus_task == "race5" and pollen_sustained is not None
                        else f"<div class='baseline-numbers'><strong>{pollen_score:.2f}<small>/100</small></strong>"
                    )
                    + f"<span>{html.escape(speed_reference)}</span></div></div>"
                )
        if best_policy and baseline_entry:
            score_delta = float(best_policy["score"]["overall"]) - float(baseline_entry["score"]["overall"])
            speed_delta = (
                float(best_policy["score"].get("performance", {}).get("sustained_speed_mph", 0.0))
                - float(baseline_entry["score"].get("performance", {}).get("sustained_speed_mph", 0.0))
            )
            comparison = best_policy["evaluation"].get("baseline_comparison", {})
            all_dimensions_beat_pollen = bool(comparison.get("improved"))
            a_to_b_beats_pollen = bool(
                comparison.get(
                    "a_to_b_improved",
                    comparison.get("checks", {}).get("a_to_b_faster", all_dimensions_beat_pollen),
                )
            )
            speed_beats_pollen = bool(
                comparison.get("speed_improved", all_dimensions_beat_pollen)
            )
            record_qualified = bool(best_policy["score"].get("record_qualified", best_policy["score"].get("qualified")))
            champion_eligible = bool(
                best_policy["score"].get("simulation_champion_eligible", record_qualified)
            )
            beats_pollen = all_dimensions_beat_pollen and champion_eligible
            record_beats_pollen = all_dimensions_beat_pollen and record_qualified
            beats_trained_raw = (
                incumbent_entry is None
                or (
                    float(best_policy["score"].get("performance", {}).get("sustained_speed_mph", 0.0))
                    > float(incumbent_entry["score"].get("performance", {}).get("sustained_speed_mph", 0.0))
                    if focus_task == "race5"
                    else float(best_policy["score"]["overall"]) > float(incumbent_entry["score"]["overall"])
                )
            )
            got_better = beats_pollen and beats_trained_raw
            scoreboard_verdict = (
                "YES — 5 MPH GOAL REACHED" if focus_task == "race5" and got_better and record_qualified
                else "YES — NEW SIMULATION CHAMPION" if got_better
                else "A → B RECORD — CONTROL QUALIFICATION PENDING"
                if focus_task == "race5" and a_to_b_beats_pollen
                else "FASTER, BUT DID NOT WIN A → B" if focus_task == "race5" and speed_beats_pollen
                else "NO — NO VERIFIED IMPROVEMENT YET"
            )
            record_copy = (
                f"5 mph goal reached: {'YES' if record_beats_pollen else 'NO'}"
                if focus_task == "race5"
                else f"qualified record vs Pollen: {'YES' if record_beats_pollen else 'NO'}"
            )
            scoreboard_detail = (
                (
                    f"A → B beats Pollen: {'YES' if a_to_b_beats_pollen else 'NO'} · "
                    f"sustained/top speed improved: {'YES' if speed_beats_pollen else 'NO'} · "
                    if focus_task == "race5"
                    else f"Raw heat vs Pollen: {'YES' if all_dimensions_beat_pollen else 'NO'} · "
                )
                + f"simulation champion vs Pollen: {'YES' if beats_pollen else 'NO'} · {record_copy} · "
                + (
                    f"sustained speed beats trained reference: {'YES' if beats_trained_raw else 'NO'} · "
                    if focus_task == "race5"
                    else f"raw beat trained reference: {'YES' if beats_trained_raw else 'NO'} · "
                )
                + (
                    f"speed vs Pollen {speed_delta:+.2f} mph."
                    if focus_task == "race5"
                    else f"development score vs Pollen {score_delta:+.2f}; qualification is required."
                )
            )
            verdict_tone = "good" if got_better else "bad"
        elif best_policy:
            qualified = bool(best_policy["score"].get("qualified"))
            scoreboard_verdict = "CONTROL-SAFE RECORD LEADER" if qualified else "NO CONTROL-SAFE LEADER YET"
            scoreboard_detail = "Pollen evidence is not registered for this task."
            verdict_tone = "good" if qualified else "bad"
        else:
            scoreboard_verdict = "NO SCORED CHALLENGER YET"
            scoreboard_detail = "Finish an evaluation to compare it with Pollen."
            verdict_tone = "bad"
        archive_count = len(finished_rows)
        fastest_speed_html = ""
        if fastest_speed_entry:
            manifest = fastest_speed_entry["manifest"]
            performance = fastest_speed_entry["score"].get("performance", {})
            top_speed = float(performance.get("top_speed_mph", 0.0))
            sustained_speed = float(performance.get("sustained_speed_mph", 0.0))
            run_id = html.escape(manifest["run_id"])
            label = html.escape(fastest_speed_entry["label"])
            qualified = bool(fastest_speed_entry["score"].get("qualified"))
            fastest_speed_html = (
                "<div class='fastest-speed-card'><div><small>FASTEST OFFICIAL-FRICTION SPEED CHECKPOINT</small>"
                f"<strong>{label}</strong>"
                f"<span>{top_speed:.2f} mph verified top · {sustained_speed:.2f} mph sustained over 20 s</span></div>"
                "<div class='fastest-speed-actions'>"
                f"<button class='play primary-action' data-run-id='{run_id}' data-label='Fastest speed checkpoint'>Try fastest in arena</button>"
                f"<a class='text-action' href='runs/{run_id}/report.html'>View proof</a></div>"
                + ("<p>Also passes the full control qualification.</p>" if qualified else "<p>Speed leader only: not auto-promoted because its long-run control gates fail.</p>")
                + "</div>"
            )
        physics_notice_html = (
            "<div class='baseline-reference race5-headtohead'><div class='baseline-duel-head'><div>"
            "<small>TWO PHYSICS PROFILES · DO NOT MIX THESE NUMBERS</small>"
            "<strong>Official Race5 uses roller friction 0.003</strong>"
            "<span>Podium top speed is a verified 0.5-second window under official physics.</span></div>"
            "<div class='baseline-improvement mixed'><small>PRESERVED SPEED-DISCOVERY DONOR · FRICTION 0.000</small>"
            "<strong>5.41 mph instantaneous</strong>"
            "<span>5.06 mph best 1 second · 4.17 mph average over 20 seconds</span></div></div>"
            "<p class='baseline-method'>The 5.41 mph checkpoint still exists and seeds transfer training, but it is not an official-friction Race5 record. Compare a policy only with results from the same physics profile and measurement window.</p></div>"
            if focus_task == "race5" else ""
        )
        speed_scout_html = ""
        if speed_scout_entry:
            scout_id = html.escape(speed_scout_entry["run_id"])
            speed_scout_html = (
                "<div class='fastest-speed-card'><div><small>PINNED EXPERIMENT · FRICTIONLESS SPEED SCOUT</small>"
                "<strong>5.41 mph peak-speed discovery checkpoint</strong>"
                "<span>5.05 mph best 1 second · 4.18 mph mean over 20 seconds · 0 falls</span></div>"
                "<div class='fastest-speed-actions'>"
                f"<button class='play primary-action' data-run-id='{scout_id}' data-label='Try 5.41 mph speed scout'>Try it in matching arena</button>"
                "</div>"
                "<p>Uses wheel frictionloss 0.000 exactly as evaluated. This is a research replay, not an official Race5 or V11 replacement.</p></div>"
            )
        # Results are grouped by decision role. A speed experiment, a stable
        # controller, and a deployable all-around policy are useful for
        # different reasons and should never masquerade as one flat ranking.
        metric_specs = (
            ("Idle creep", "idle_end_speed_mph", "mph", 3, False),
            ("Top speed", "top_speed_mph", "mph", 2, True),
            ("Sustained speed", "sustained_speed_mph", "mph", 2, True),
            ("100 ft time", "elapsed_time_100ft_s", "s", 2, False),
            ("Trap speed", "trap_speed_100ft_mph", "mph", 2, True),
            ("Launch acceleration", "acceleration_first_second_mps2", "mph/s", 2, True),
            ("Straight-line drift", "long_run_max_drift_ft", "ft", 2, False),
            ("Heading error", "long_run_max_heading_error_deg", "°", 2, False),
            ("Agility", "agility_score", "/100", 1, True),
            # Controller effort is diagnostic, not ordinal: zero can mean a
            # naturally straight policy or a disabled controller.  Rank the
            # resulting heading/drift, while still showing steering effort.
            ("Automatic steering", "auto_steering_percent", "%", 1, None),
        )
        pollen_performance = baseline_entry["score"].get("performance", {}) if baseline_entry else {}

        def shown_value(performance: dict[str, Any], key: str) -> float | None:
            value = performance.get(key)
            if value is None:
                return None
            value = float(value)
            return value * MPS_TO_MPH if key == "acceleration_first_second_mps2" else value

        def numeric(performance: dict[str, Any], key: str, default: float = 0.0) -> float:
            value = performance.get(key)
            return default if value is None else float(value)

        def cell(performance: dict[str, Any], key: str, digits: int = 2, suffix: str = "") -> str:
            value = performance.get(key)
            return "—" if value is None else f"{float(value):.{digits}f}{suffix}"

        def pollen_record(entry: dict[str, Any]) -> tuple[int, int]:
            performance = entry["score"].get("performance", {})
            wins = total = 0
            for _, key, _, _, higher_is_better in metric_specs:
                if higher_is_better is None:
                    continue
                candidate = shown_value(performance, key)
                pollen = shown_value(pollen_performance, key)
                if candidate is None or pollen is None:
                    continue
                total += 1
                wins += int(candidate > pollen if higher_is_better else candidate < pollen)
            return wins, total

        def decision_label(entry: dict[str, Any]) -> str:
            run_id = entry["manifest"]["run_id"]
            aliases = (
                ("duckwing-v66-v65-control-fusion", "V66 · V65 Control Fusion"),
                ("rtx5090-v65-v63-immediate-switch", "RTX 5090 V65 · V63/V59 Switch"),
                ("rtx5090-v63-embedded-mid", "RTX 5090 V63 · Embedded Mid"),
                ("duckwing-v62-v59-launch-fusion", "V62 · V59 Launch Fusion"),
                ("duckwing-v61-v57b-control-fusion", "V61 · V57b Control Fusion"),
                ("rtx5090-v59-i99", "RTX 5090 V59 · Speed Leader"),
                ("rtx5090-v60-i20", "RTX 5090 V60 · Balanced"),
                ("rtx5090-v58-i20", "RTX 5090 V58 · Peak Line"),
                ("rtx5090-v57b-i50", "RTX 5090 V57b · Frontier Speed"),
                ("rtx5090-v56-i10", "RTX 5090 V56 · Line Speed"),
                ("microduck_hybrid_controlaware", "Control-aware V11 × Speed"),
                ("ducklab-race5-v17-frontier-i250", "V17 · Frontier i250"),
                ("race5-microduck_hybrid-export", "Speed-first Race5 Hybrid"),
                ("speed-retention-pilot2-brake-line", "Speed Retention Pilot 2"),
                ("v14-lean-glide", "V14 Lean Glide"),
                ("v11-drag-launch", "V11 Drag Launch"),
                ("v8-centerline", "V8 Centerline"),
                ("v13-100m", "V13 100 m"),
                ("v15-v11-hybrid-speedline", "V15 Speedline"),
                ("speed-retention-v3", "Speed Retention V3"),
                ("friction-transfer-pilot2", "Friction Transfer Pilot 2"),
            )
            for token, label in aliases:
                if token in run_id:
                    return label
            return entry["label"]

        result_entries = [
            item for item in scored_entries
            if item["manifest"]["task"] == focus_task
            and item["manifest"].get("experiment_kind") not in {"smoke", "factory"}
        ]
        result_entries.sort(
            key=lambda item: (
                bool(item["score"].get("qualified")),
                pollen_record(item)[0],
                -numeric(item["score"].get("performance", {}), "elapsed_time_100ft_s", float("inf")),
                numeric(item["score"].get("performance", {}), "top_speed_mph"),
            ),
            reverse=True,
        )
        overall_best = result_entries[0] if result_entries else None

        def play_button(entry: dict[str, Any], label: str = "Try simulation") -> str:
            run_id = html.escape(entry["manifest"]["run_id"])
            return f"<button class='play' data-run-id='{run_id}' data-label='{html.escape(label)}'>{html.escape(label)}</button>"

        comparison_html = ""
        if overall_best and baseline_entry:
            best_performance = overall_best["score"].get("performance", {})
            rows = []
            for label, key, unit, digits, higher_is_better in metric_specs:
                if higher_is_better is None:
                    continue
                candidate = shown_value(best_performance, key)
                pollen = shown_value(pollen_performance, key)
                if candidate is None or pollen is None:
                    continue
                improved = candidate > pollen if higher_is_better else candidate < pollen
                index = (
                    candidate / pollen if higher_is_better and abs(pollen) > 1.0e-9
                    else pollen / candidate if not higher_is_better and abs(candidate) > 1.0e-9
                    else 1.0
                ) * 100.0
                delta = (
                    (candidate - pollen) / abs(pollen) if higher_is_better and abs(pollen) > 1.0e-9
                    else (pollen - candidate) / abs(pollen) if not higher_is_better and abs(pollen) > 1.0e-9
                    else 0.0
                ) * 100.0
                rows.append(
                    "<article class='comparison-row'>"
                    f"<div class='comparison-name'><strong>{html.escape(label)}</strong><span>{'Higher' if higher_is_better else 'Lower'} is better</span></div>"
                    "<div class='comparison-bars'>"
                    f"<div><small>Leader</small><span class='bar-track'><i class='leader-bar' style='width:{min(100.0, index / 2.0):.1f}%'></i></span><b>{candidate:.{digits}f} {html.escape(unit)}</b></div>"
                    f"<div><small>Pollen</small><span class='bar-track'><i class='pollen-bar' style='width:50%'></i></span><b>{pollen:.{digits}f} {html.escape(unit)}</b></div></div>"
                    f"<span class='win-chip {'win' if improved else 'loss'}'>{'WIN' if improved else 'LOSS'} · {delta:+.1f}%</span></article>"
                )
            comparison_html = "".join(rows)

        leaderboard_rows = []
        for rank, entry in enumerate(result_entries, 1):
            performance = entry["score"].get("performance", {})
            gates = entry["score"].get("qualification_gates", {})
            passed = sum(bool(gate.get("passed")) for gate in gates.values())
            wins, total = pollen_record(entry)
            acceleration = shown_value(performance, "acceleration_first_second_mps2")
            acceleration_cell = "—" if acceleration is None else f"{acceleration:.2f}"
            run_id = html.escape(entry["manifest"]["run_id"])
            role = "Best overall" if entry is overall_best else "Speed specialist" if entry is fastest_speed_entry else "Candidate"
            leaderboard_rows.append(
                f"<tr class='{'best-row' if entry is overall_best else ''}'><td class='rank-cell'>#{rank}</td>"
                f"<td><div class='model-cell'><strong>{html.escape(decision_label(entry))}</strong><span>{role}</span></div></td>"
                f"<td><strong class='record'>{wins}/{total}</strong></td><td>{passed}/{len(gates)}</td>"
                f"<td>{cell(performance, 'elapsed_time_100ft_s', 2, 's')}</td>"
                f"<td>{cell(performance, 'top_speed_mph')}</td>"
                f"<td>{acceleration_cell}</td>"
                f"<td>{cell(performance, 'long_run_max_drift_ft')}</td>"
                f"<td>{cell(performance, 'long_run_max_heading_error_deg', 2, '°')}</td>"
                f"<td>{cell(performance, 'auto_steering_percent', 1, '%')}</td>"
                f"<td><button class='play table-play' data-run-id='{run_id}' data-label='Try'>Try</button> <a href='runs/{run_id}/report.html'>Proof</a></td></tr>"
            )
        if baseline_entry:
            p = pollen_performance
            pollen_acceleration = shown_value(p, "acceleration_first_second_mps2")
            pollen_acceleration_cell = "—" if pollen_acceleration is None else f"{pollen_acceleration:.2f}"
            reference_total = pollen_record(overall_best)[1] if overall_best else 0
            leaderboard_rows.append(
                "<tr class='reference-row'><td>—</td><td><div class='model-cell'><strong>Pollen official roller</strong><span>Immutable baseline</span></div></td>"
                f"<td>0/{reference_total}</td><td>Reference</td>"
                f"<td>{cell(p, 'elapsed_time_100ft_s', 2, 's')}</td><td>{cell(p, 'top_speed_mph')}</td>"
                f"<td>{pollen_acceleration_cell}</td>"
                f"<td>{cell(p, 'long_run_max_drift_ft')}</td><td>{cell(p, 'long_run_max_heading_error_deg', 2, '°')}</td>"
                f"<td>{cell(p, 'auto_steering_percent', 1, '%')}</td><td>Baseline</td></tr>"
            )

        role_cards = []
        if overall_best:
            best_wins_for_copy, best_total_for_copy = pollen_record(overall_best)
            overall_copy = (
                "Only candidate that wins the complete measured head-to-head while passing every control gate."
                if best_total_for_copy and best_wins_for_copy == best_total_for_copy
                else "Highest-ranked qualified candidate across the comparable evidence available for this task."
            )
            role_cards.append(("DEPLOY / TEST NOW", "Best overall", overall_best, overall_copy))
        if fastest_speed_entry and fastest_speed_entry is not overall_best:
            role_cards.append(("OFFICIAL SPEED SPECIALIST", "Pure speed", fastest_speed_entry, "Fastest official-friction top speed; useful as a speed donor, but weaker all-around control."))
        qualified_entries = [item for item in result_entries if item["score"].get("qualified") and item is not overall_best]
        if qualified_entries:
            control_anchor = min(
                qualified_entries,
                key=lambda item: (
                    numeric(item["score"].get("performance", {}), "long_run_max_drift_ft", float("inf"))
                    + numeric(item["score"].get("performance", {}), "long_run_max_heading_error_deg", float("inf")) / 10.0
                ),
            )
            if control_anchor is not fastest_speed_entry:
                role_cards.append(("CONTROL REFERENCE", "Stable anchor", control_anchor, "Best qualified control reference to protect steering, heading, and recovery behavior during future training."))
        role_cards_html = "".join(
            "<article class='role-card'>"
            f"<small>{eyebrow}</small><h3>{title}</h3><strong>{html.escape(decision_label(entry))}</strong><p>{copy}</p>"
            f"<div>{play_button(entry, 'Try model')} <a href='runs/{html.escape(entry['manifest']['run_id'])}/report.html'>View evidence</a></div></article>"
            for eyebrow, title, entry, copy in role_cards
        )
        if speed_scout_entry:
            scout_id = html.escape(speed_scout_entry["run_id"])
            role_cards_html += (
                "<article class='role-card seed-card'><small>NEXT TRAINING DONOR</small><h3>High-upside seed</h3>"
                "<strong>5.41 mph frictionless speed scout</strong><p>Preserve as a teacher or transfer seed. It proves the gait has speed headroom, but it is not deployable under official 0.003 friction.</p>"
                f"<div><button class='play' data-run-id='{scout_id}' data-label='Try matching replay'>Try matching replay</button></div></article>"
            )

        if overall_best:
            best_performance = overall_best["score"].get("performance", {})
            best_wins, best_total = pollen_record(overall_best)
            best_id = html.escape(overall_best["manifest"]["run_id"])
            best_time = cell(best_performance, "elapsed_time_100ft_s", 2, "s")
            baseline_time = cell(pollen_performance, "elapsed_time_100ft_s", 2, "s")
            benchmark_name = "Official Race5" if focus_task == "race5" else display_task_name(focus_task)
            hero_html = (
                f"<section class='leader-hero'><div class='hero-copy'><span class='winner-label'>BEST OVERALL · {html.escape(benchmark_name.upper())}</span>"
                f"<h1>{html.escape(decision_label(overall_best))}</h1>"
                f"<p>The clear default: <strong>{best_wins}/{best_total} comparable wins versus Pollen</strong> and {sum(bool(g.get('passed')) for g in overall_best['score'].get('qualification_gates', {}).values())}/{len(overall_best['score'].get('qualification_gates', {}))} qualification gates pass. Use this unless you are deliberately studying one specialist.</p>"
                f"<div class='hero-actions'>{play_button(overall_best, 'Try best model')} <a class='proof-link' href='runs/{best_id}/report.html'>View full evidence</a></div></div>"
                "<div class='hero-kpis'>"
                f"<div><small>Pollen record</small><strong>{best_wins}/{best_total}</strong><span>measured wins</span></div>"
                f"<div><small>100 ft</small><strong>{best_time}</strong><span>vs {baseline_time}</span></div>"
                f"<div><small>Top speed</small><strong>{cell(best_performance, 'top_speed_mph')}</strong><span>mph verified</span></div>"
                f"<div><small>Control gates</small><strong>{sum(bool(g.get('passed')) for g in overall_best['score'].get('qualification_gates', {}).values())}/{len(overall_best['score'].get('qualification_gates', {}))}</strong><span>passed</span></div></div></section>"
            )
        else:
            hero_html = "<section class='leader-hero'><div class='hero-copy'><span class='winner-label'>NO EVALUATED MODEL</span><h1>Run an official evaluation</h1></div></section>"

        next_move_html = (
            "<section class='next-move'><span>RECOMMENDED NEXT MOVE</span><div><strong>Start from the control-aware champion at official 0.003 friction.</strong> Optimize acceleration and top speed, but reject any checkpoint that loses one of its 9 Pollen wins or 15 control gates. Use the 5.41 mph scout as a teacher—not as the deployment baseline.</div></section>"
            if focus_task == "race5"
            else "<section class='next-move'><span>RECOMMENDED NEXT MOVE</span><div><strong>Start from the qualified overall leader.</strong> Preserve its passing gates while optimizing the weakest Pollen-relative metric.</div></section>"
        )
        leaderboard_context = (
            "Official Race5 · friction 0.003 · 1.75 A · same replay and controller"
            if focus_task == "race5"
            else f"{display_task_name(focus_task)} · same evaluation suite and baseline"
        )

        content = (
            "<header class='analysis-header'><div><span class='wordmark'>DUCKLAB</span><span class='header-subtitle'>Agentic robotics RL · DuckWing flagship · MicroDuck model results</span></div>"
            "<div class='header-status'><span id='system-status'>Checking training status…</span></div></header>"
            "<section id='live-training' class='live-training-panel' hidden><div class='live-training-head'><div><p class='eyebrow'>LIVE TRAINING</p><h2 id='training-title'>Active MicroDuck run</h2><p id='training-verdict' class='live-training-verdict'>Waiting for the first update…</p></div><div class='live-training-count'><strong id='training-iteration'>—</strong><span id='training-percent'>starting</span></div></div>"
            "<div class='live-training-progress'><span id='training-progress-bar'></span></div>"
            "<div class='live-training-grid'><div><small>Current reward</small><strong id='training-current-reward'>—</strong></div><div><small>Best reward</small><strong id='training-best-reward'>—</strong></div><div><small>Throughput</small><strong id='training-throughput'>—</strong></div><div><small>ETA</small><strong id='training-eta'>—</strong></div><div><small>Latest checkpoint</small><strong id='training-checkpoint'>—</strong></div></div>"
            "<div id='training-chart-wrap' class='live-training-chart' hidden><div><span>Reward history</span><small id='training-chart-range'>—</small></div><svg viewBox='0 0 720 180' role='img' aria-label='Live training reward history'><line x1='32' y1='150' x2='704' y2='150'></line><line x1='32' y1='18' x2='32' y2='150'></line><polyline id='training-reward-line' points=''></polyline></svg></div></section>"
            + agent_runs_html
            + hero_html
            + "<main id='results'><div class='results-heading'><div><p class='eyebrow'>DECISION VIEW</p><h2>What each checkpoint is for</h2></div>"
            f"<p>{len(result_entries)} evaluated candidates · {archive_count} historical experiment groups preserved off the main view</p></div>"
            + "<div class='role-grid'>" + role_cards_html + "</div>"
            + next_move_html
            + "<section class='comparison-panel'><div class='panel-heading'><div><p class='eyebrow'>HEAD-TO-HEAD</p><h2>Best overall vs Pollen</h2></div><p>Performance index bars normalize Pollen to 100. Longer is always better; exact values remain visible.</p></div>"
            + comparison_html + "</section>"
            + f"<section class='leaderboard-panel'><div class='panel-heading'><div><p class='eyebrow'>DEFINITIVE RANKING</p><h2>All comparable results</h2></div><p>{html.escape(leaderboard_context)}</p></div>"
            + "<div class='table-wrap'><table class='definitive-table'><thead><tr><th>Rank</th><th>Model</th><th>vs Pollen</th><th>Gates</th><th>100 ft ↓</th><th>Top mph ↑</th><th>Accel ↑</th><th>Drift ft ↓</th><th>Heading ↓</th><th>Steering ↓</th><th>Action</th></tr></thead><tbody>"
            + "".join(leaderboard_rows) + "</tbody></table></div></section></main>"
            + "<footer>Every artifact, raw run, and evaluation remains immutable. DuckLab keeps the agent flexible and the evidence structured.</footer>"
            + "<script>const TOKEN='__CONTROL_TOKEN__';"
            + "async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Policy-Bench-Token':TOKEN},body:JSON.stringify(body)});const j=await r.json();if(r.status===403){location.reload();throw new Error('Dashboard reconnected. Click once more.');}if(!r.ok)throw new Error(j.error||'Request failed');return j;}"
            + "async function openWhenReady(win,url){for(let i=0;i<16;i++){try{await fetch(url,{mode:'no-cors',cache:'no-store'});if(win&&!win.closed)win.location=url;return true;}catch(e){await new Promise(r=>setTimeout(r,500));}}return false;}"
            + "async function playRun(button){const label=button.dataset.label||'Try';const view=window.open('about:blank','microduck-'+button.dataset.runId.replace(/[^a-zA-Z0-9]/g,'-'));button.disabled=true;button.textContent='Starting…';try{const result=await api('/api/play',{run_id:button.dataset.runId});const ready=await openWhenReady(view,result.open_url||result.viser_url);if(!ready)throw new Error('Simulator started, but its viewer port is not forwarded.');}catch(error){if(view&&!view.closed)view.close();alert(error.message);}finally{button.disabled=false;button.textContent=label;}}"
            + "document.querySelectorAll('.play').forEach(button=>button.addEventListener('click',()=>playRun(button)));"
            + "function metric(value,digits=2){return Number.isFinite(Number(value))?Number(value).toLocaleString(undefined,{maximumFractionDigits:digits}):'—';}"
            + "function rewardPoints(history){const rows=(history||[]).slice(-120).filter(row=>Number.isFinite(Number(row.reward)));if(rows.length<2)return null;const values=rows.map(row=>Number(row.reward));const low=Math.min(...values),high=Math.max(...values),span=high-low||1;return {points:rows.map((row,index)=>{const x=32+(672*index/(rows.length-1));const y=150-(132*(Number(row.reward)-low)/span);return x.toFixed(1)+','+y.toFixed(1);}).join(' '),low,high,count:rows.length};}"
            + "async function refreshStatus(){try{const r=await fetch('/api/status',{cache:'no-store'});const s=await r.json();const agentSection=document.querySelector('#agent-runs');const agentRevision=(s.agent_runs||[]).map(x=>(x.run_id||'')+':'+(x.updated_at||'')).join('|');if(agentSection&&agentSection.dataset.revision!==agentRevision){location.reload();return;}const active=s.training.detected.length>0;const p=s.training.progress;const card=document.querySelector('#live-training');document.querySelector('#system-status').textContent=active?('Training · iteration '+(p?p.iteration:'starting')+(p&&p.total?' / '+p.total:'')):'No training running';card.hidden=!active;if(!active)return;if(!p){document.querySelector('#training-title').textContent='Training process starting';return;}const intel=p.intelligence||{};const pct=p.total?Math.max(0,Math.min(100,100*Number(p.iteration)/Number(p.total))):0;document.querySelector('#training-title').textContent=(s.training.config&&s.training.config.run_name)||'Active MicroDuck frontier run';document.querySelector('#training-iteration').textContent='Iteration '+metric(p.iteration,0)+' / '+metric(p.total,0);document.querySelector('#training-percent').textContent=pct.toFixed(1)+'% complete';document.querySelector('#training-progress-bar').style.width=pct+'%';document.querySelector('#training-current-reward').textContent=metric(intel.current_reward);document.querySelector('#training-best-reward').textContent=metric(intel.best_reward);document.querySelector('#training-throughput').textContent=metric(intel.steps_per_second,0)+' steps/s';document.querySelector('#training-eta').textContent=p.eta||'—';document.querySelector('#training-checkpoint').textContent=intel.latest_checkpoint_iteration===null||intel.latest_checkpoint_iteration===undefined?'Pending':'Iteration '+metric(intel.latest_checkpoint_iteration,0);const verdict=document.querySelector('#training-verdict');verdict.textContent=intel.verdict||'Collecting enough history to judge the run.';verdict.dataset.tone=intel.verdict_tone||'neutral';const curve=rewardPoints(p.reward_history);const wrap=document.querySelector('#training-chart-wrap');wrap.hidden=!curve;if(curve){document.querySelector('#training-reward-line').setAttribute('points',curve.points);document.querySelector('#training-chart-range').textContent=metric(curve.low)+' → '+metric(curve.high)+' · last '+curve.count+' updates';}}catch(e){document.querySelector('#system-status').textContent='Status unavailable';}}refreshStatus();setInterval(refreshStatus,5000);</script>"
        )
        output = self.state_dir / "index.html"
        output.write_text(page("DuckLab Robotics RL", content, show_title=False))
        return output


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            output.update(flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output


def load_metrics_module():
    path = LAB_ROOT / "tools" / "rl_metrics.py"
    spec = importlib.util.spec_from_file_location("microduck_rl_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load metrics reader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def page(title: str, body: str, show_title: bool = True) -> str:
    heading = f"<h1>{html.escape(title)}</h1>" if show_title else ""
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>
<style>
:root{{--bg:#080610;--surface:#12101d;--surface-2:#19152a;--line:#322a49;--text:#f6f2ff;--muted:#aaa1bd;--brand:#8157ff;--brand-2:#2dd4c5;--accent:#f3c969;--danger:#dc5d68}}
*{{box-sizing:border-box}}body{{font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1240px;margin:0 auto;padding:44px 24px 80px;background:radial-gradient(circle at 50% -20%,#162638 0,var(--bg) 42%);color:var(--text)}}
h1{{font-size:clamp(2rem,4vw,3.2rem);letter-spacing:-.045em;margin:0 0 8px}}h2{{font-size:1.22rem;letter-spacing:-.015em;margin:34px 0 10px}}p{{margin:7px 0}}a{{color:var(--brand-2);text-decoration:none}}a:hover{{text-decoration:underline}}
.lede{{font-size:1.05rem;color:var(--muted);margin-bottom:22px}}.summary-strip{{display:flex;align-items:center;min-height:48px;padding:12px 16px;border:1px solid #285345;border-radius:12px;background:#10251f;color:#c9f9e7}}
.panel{{background:color-mix(in srgb,var(--surface) 94%,transparent);border:1px solid var(--line);border-radius:14px;padding:18px;margin:10px 0;box-shadow:0 14px 35px rgba(0,0,0,.14)}}
.section-heading,.run-card,.session-card,.session-actions{{display:flex;align-items:center;gap:12px}}.section-heading,.run-card,.session-card{{justify-content:space-between}}.run-card{{padding:4px 0 14px}}.run-card+.run-card{{border-top:1px solid var(--line);padding-top:14px}}
.status-dot{{display:inline-block;width:9px;height:9px;margin-right:9px;border-radius:50%;background:var(--brand-2);box-shadow:0 0 0 5px rgba(45,212,197,.12)}}.pill,.badge{{display:inline-block;background:#28203e;color:#ddd1ff;border-radius:999px;padding:2px 9px;font-size:.78rem;margin-left:7px}}
button,.session-actions a{{appearance:none;border:0;border-radius:9px;padding:9px 13px;background:#6842df;color:white;font:inherit;font-weight:650;cursor:pointer;white-space:nowrap}}button:hover,.session-actions a:hover{{filter:brightness(1.12);text-decoration:none}}button:disabled{{opacity:.48;cursor:not-allowed}}button.secondary{{background:#2b2639}}button.danger{{background:#552b32;color:#ffcdd2}}
.session-grid{{display:grid;gap:10px}}.session-card{{background:var(--surface-2);border:1px solid var(--line);border-radius:11px;padding:14px 15px}}.session-actions a{{background:#333057}}.session-actions a:first-child{{background:#6842df}}
.progress{{height:12px;background:#272235;border-radius:99px;overflow:hidden;margin:15px 0 8px}}.progress span{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--brand),var(--brand-2));transition:width .5s}}.progress-copy{{color:#ddd6e9}}
.live-curve{{margin-top:15px;padding:12px 13px;background:#0d141b;border:1px solid var(--line);border-radius:10px}}.live-curve[hidden]{{display:none}}.curve-heading{{display:flex;justify-content:space-between;gap:12px;align-items:center}}.curve-heading>div{{min-width:0}}.curve-scope{{padding:6px 10px;font-size:.8rem}}.live-curve svg{{display:block;width:100%;height:auto;max-height:180px}}.live-curve line{{stroke:#30404e}}.live-curve polyline{{fill:none;stroke:var(--brand-2);stroke-width:2.5;vector-effect:non-scaling-stroke}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--surface)}}table{{width:100%;border-collapse:collapse;min-width:920px}}th,td{{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#b7c7d5;font-size:.78rem;text-transform:uppercase;letter-spacing:.055em;background:#151e28}}tr:last-child td{{border-bottom:0}}tbody tr:hover,table tr:hover td{{background:rgba(93,189,255,.025)}}
details{{margin-top:9px;padding:9px 11px;background:#0e151c;border:1px solid var(--line);border-radius:9px;min-width:360px}}details summary{{cursor:pointer;font-weight:650;color:#d7e4ee}}details table{{margin-top:10px;min-width:670px;font-size:.9em}}td button{{margin:2px 4px 2px 0}}
form{{display:flex;gap:9px;margin-top:12px}}input,select{{min-width:0;background:#0b1117;color:var(--text);border:1px solid #435567;border-radius:9px;padding:11px 12px;font:inherit}}input{{flex:1}}input:focus,select:focus{{outline:2px solid rgba(93,189,255,.4);border-color:var(--brand-2)}}.chat-log{{max-height:280px;overflow:auto}}#chat-action{{margin-top:10px}}#chat-action a{{margin-left:10px}}
pre,.mono{{font-family:ui-monospace,SFMono-Regular,monospace;overflow:auto;background:#10171f;padding:13px;border-radius:9px}}.chart{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;margin:12px 0}}.chart svg{{display:block;width:100%;height:auto}}.muted{{color:var(--muted);font-size:.9em}}
html{{scroll-behavior:smooth}}body{{max-width:1320px;padding:30px 28px 90px;background:radial-gradient(circle at 82% -14%,rgba(105,64,190,.27),transparent 36%),radial-gradient(circle at 0 18%,rgba(22,113,111,.14),transparent 30%),var(--bg)}}section{{scroll-margin-top:86px;margin-top:36px}}
.product-header{{position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:center;gap:28px;min-height:140px;padding:25px 30px;border:1px solid #433763;border-radius:20px;background:linear-gradient(135deg,rgba(30,23,49,.98),rgba(13,11,23,.98));box-shadow:0 20px 55px rgba(0,0,0,.28)}}.product-header:after{{content:'';position:absolute;width:260px;height:260px;right:-85px;top:-165px;border-radius:50%;background:radial-gradient(circle,rgba(129,87,255,.28),transparent 68%);pointer-events:none}}.brand-lockup{{display:flex;align-items:center;gap:18px;z-index:1}}.duck-mark{{position:relative;display:grid;place-items:center;flex:0 0 62px;height:62px;border:1px solid rgba(243,201,105,.55);border-radius:18px 18px 18px 6px;background:linear-gradient(145deg,#7950ed,#3b236f);box-shadow:0 12px 34px rgba(86,49,169,.32);color:#fff;font-weight:950;font-size:1.05rem;letter-spacing:-.08em}}.duck-mark:after{{content:'';position:absolute;right:-10px;width:15px;height:10px;background:var(--accent);clip-path:polygon(0 0,100% 50%,0 100%)}}.product-header h1{{font-size:clamp(1.9rem,4vw,3rem);line-height:1.04;letter-spacing:-.05em;margin:5px 0 8px}}.product-header h1 em{{color:var(--accent);font-style:normal;font-weight:650}}.eyebrow{{margin:0;color:var(--brand-2);font-size:.7rem;font-weight:850;letter-spacing:.15em}}.tagline{{margin:0;color:#bdb5cd;font-size:.98rem}}.header-side{{display:flex;align-items:flex-end;z-index:1;max-width:460px}}.header-status{{padding:10px 13px;border:1px solid #4c3e6b;border-radius:10px;background:rgba(9,7,16,.62);color:#e4ddf0;text-align:right}}
.quick-nav{{position:sticky;top:12px;z-index:10;display:flex;gap:6px;width:max-content;max-width:100%;margin:14px auto 0;padding:6px;border:1px solid rgba(58,79,96,.9);border-radius:12px;background:rgba(13,19,26,.9);box-shadow:0 12px 35px rgba(0,0,0,.26);backdrop-filter:blur(14px);overflow:auto}}.quick-nav a{{padding:7px 12px;border-radius:8px;color:#b9c8d4;font-size:.84rem;font-weight:650;white-space:nowrap}}.quick-nav a:hover{{background:#1c2934;color:#fff;text-decoration:none}}
.section-title{{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:11px}}.section-title h2{{margin:0;font-size:1.35rem}}.section-title .eyebrow{{margin-bottom:3px}}.section-note{{color:var(--muted);font-size:.9rem;text-align:right;max-width:480px}}.panel{{margin:0;border-radius:16px;padding:20px;background:rgba(18,25,34,.94)}}
button,.session-actions a{{border:1px solid transparent;font-weight:700;transition:transform .15s ease,filter .15s ease,border-color .15s ease}}button:hover,.session-actions a:hover{{transform:translateY(-1px)}}button:disabled{{transform:none}}button.secondary{{background:#263644;border-color:#3a4c5c}}.primary-action{{padding:10px 16px}}
.button-link{{display:inline-flex;align-items:center;justify-content:center;padding:10px 16px;border:0;border-radius:9px;background:linear-gradient(135deg,var(--accent),#e6a94c);color:#20140a;font-weight:850;white-space:nowrap;text-decoration:none}}.button-link:hover{{filter:brightness(1.07);text-decoration:none}}
.training-intelligence{{display:grid;gap:12px;margin-top:16px}}.training-verdict{{display:flex;align-items:center;gap:12px;padding:13px 15px;border:1px solid #44385f;border-radius:11px;background:#171224}}.training-verdict span{{padding:3px 7px;border-radius:6px;background:#2b2240;color:#c9b9f5;font-size:.68rem;font-weight:900;letter-spacing:.1em}}.training-verdict.good{{border-color:#21655e;background:#0d2524}}.training-verdict.good span{{background:#164a45;color:#9df1e7}}.training-verdict.watch{{border-color:#6a5528;background:#28200e}}.training-verdict.watch span{{background:#57451d;color:#ffe09a}}.training-verdict.bad{{border-color:#723943;background:#2a1117}}.training-verdict.bad span{{background:#5c2832;color:#ffc3cb}}.metric-grid{{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#0d0a15}}.metric-grid>div{{min-width:0;padding:12px 13px;border-right:1px solid var(--line)}}.metric-grid>div:last-child{{border:0}}.metric-grid strong{{display:block;margin-top:3px;font-size:.92rem;overflow-wrap:anywhere}}.skill-signals{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}.skill-signals:empty{{display:none}}.skill-signals span{{padding:10px 12px;border:1px solid #332b49;border-radius:9px;background:#151121}}.skill-signals strong{{display:block;margin-top:2px;color:#dbd2ea}}.metric-help{{margin:0;color:var(--muted);font-size:.8rem}}
.finished-grid{{display:grid;gap:14px}}.finished-card{{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:linear-gradient(145deg,rgba(20,29,39,.98),rgba(15,22,29,.98));box-shadow:0 12px 30px rgba(0,0,0,.12)}}.finished-card-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding:20px 21px 13px}}.finished-card h3{{font-size:1.08rem;margin:0;letter-spacing:-.015em}}.run-tags{{display:flex;align-items:center;flex-wrap:wrap;gap:7px;margin-top:9px}}.run-tags .pill{{margin:0}}.kind-tag,.complete-tag,.stage-badge{{padding:3px 8px;border-radius:999px;background:#252f39;color:#b6c4d0;font-size:.74rem;font-weight:700}}.complete-tag{{background:#143c31;color:#9ee8ce}}.launch-cluster{{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}}.run-stats{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid rgba(42,57,72,.65)}}.run-stats>div{{display:flex;flex-direction:column;gap:2px;padding:13px 21px;border-right:1px solid rgba(42,57,72,.65)}}.run-stats>div:last-child{{border:0}}small{{display:block;color:var(--muted);font-size:.69rem;text-transform:uppercase;letter-spacing:.07em;font-weight:750}}
.result-banner{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:0 20px 16px;padding:14px 16px;border:1px solid #6b3b43;border-radius:11px;background:#281319}}.result-banner.good{{border-color:#267164;background:linear-gradient(135deg,#102b27,#10221f)}}.result-banner strong,.result-banner span{{display:block}}.result-banner strong{{margin:2px 0;font-size:1.06rem;color:#ffc6cd}}.result-banner.good strong{{color:#91f0d7}}.result-banner span{{color:#c8d5db;font-size:.87rem}}.result-arrow{{flex:0 0 auto!important;color:#8edcca!important;font-weight:700}}
.result-detail{{margin:24px 0;padding:22px;border:1px solid #6b3b43;border-radius:16px;background:#241217}}.result-detail.good{{border-color:#267164;background:linear-gradient(145deg,#102822,#101d1b)}}.result-detail h2{{margin-top:18px}}.result-detail>.eyebrow+h2{{margin-top:4px;font-size:1.55rem;color:#ffc6cd}}.result-detail.good>.eyebrow+h2{{color:#91f0d7}}.result-lede{{font-size:1.05rem;max-width:850px}}.result-kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:17px 0}}.result-kpis>div{{padding:13px 14px;border:1px solid #315348;border-radius:10px;background:rgba(4,15,13,.45)}}.result-kpis strong{{display:block;margin-top:3px}}.pass{{color:#76e7c9}}.fail{{color:#ff9daa}}
.baseline-reference{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:10px;padding:15px 20px;border:1px solid #776133;border-radius:14px;background:linear-gradient(135deg,#251f13,#18140e)}}.baseline-reference strong,.baseline-reference span{{display:block}}.baseline-reference>div>strong{{margin:2px 0;color:#ffe09a;font-size:1.08rem}}.baseline-reference span{{color:#d5c9ae;font-size:.86rem}}.baseline-numbers{{text-align:right}}.baseline-numbers>strong{{font-size:1.7rem;font-variant-numeric:tabular-nums}}.baseline-numbers>strong small{{display:inline;margin-left:3px}}.baseline-reference.race5-headtohead{{display:block;padding:19px 20px;background:radial-gradient(circle at 92% 0,rgba(62,215,176,.13),transparent 38%),linear-gradient(135deg,#251f13,#141a17)}}.baseline-duel-head{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:15px}}.baseline-duel-head>div:first-child>strong{{margin:3px 0;color:#ffe09a;font-size:1.08rem}}.baseline-improvement{{text-align:right}}.baseline-improvement>small{{color:#b8a9c9}}.baseline-improvement>strong{{margin:2px 0;color:#ffb2bc;font-size:1.55rem}}.baseline-improvement.good>strong{{color:#82efd1}}.baseline-comparison-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}.baseline-stat{{padding:12px 13px;border:1px solid #453746;border-radius:11px;background:rgba(10,9,14,.52)}}.baseline-stat.improved{{border-color:#285e53;background:rgba(18,57,49,.48)}}.baseline-stat.regressed{{border-color:#713c47;background:rgba(67,25,35,.48)}}.baseline-stat>small{{color:#a99db7;font-weight:800;letter-spacing:.035em;text-transform:uppercase}}.baseline-stat>strong{{margin:4px 0 2px;color:#f5f0fb;font-size:1.38rem;font-variant-numeric:tabular-nums}}.baseline-stat.improved>strong{{color:#93efd7}}.baseline-stat>strong em{{margin-left:4px;color:#bfb4ca;font-size:.68rem;font-style:normal}}.baseline-stat>span{{font-size:.75rem}}.baseline-method{{margin:12px 1px 0;color:#9f94aa;font-size:.75rem}}.scoreboard-verdict{{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:14px;padding:17px 20px;border:1px solid #713c47;border-radius:14px;background:#29131a}}.scoreboard-verdict.good{{border-color:#277668;background:#102a25}}.scoreboard-verdict strong{{font-size:1.15rem;color:#ffb2bc}}.scoreboard-verdict.good strong{{color:#82efd1}}.scoreboard-verdict span{{color:#d1c8dc}}.podium-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;align-items:stretch}}.podium-card{{position:relative;display:flex;flex-direction:column;min-height:345px;padding:19px;border:1px solid #3a3150;border-radius:16px;background:linear-gradient(155deg,#191526,#100d18);box-shadow:0 15px 35px rgba(0,0,0,.16)}}.podium-card.winner{{border-color:#aa873b;background:radial-gradient(circle at 85% 0,rgba(243,201,105,.17),transparent 37%),linear-gradient(155deg,#201a26,#100d18)}}.podium-card.empty{{justify-content:center;opacity:.62;border-style:dashed}}.podium-rank{{font-size:1.08rem;font-weight:900;color:var(--accent)}}.podium-status{{align-self:flex-start;margin-top:12px;padding:3px 8px;border-radius:999px;background:#49242c;color:#ffb2bc;font-size:.68rem;font-weight:900;letter-spacing:.06em}}.podium-status.good{{background:#16483f;color:#93efd7}}.podium-card h3{{margin:12px 0 4px;font-size:1.08rem}}.podium-score{{display:flex;align-items:baseline;gap:5px;margin-top:4px}}.podium-score strong{{font-size:2.35rem;line-height:1;font-variant-numeric:tabular-nums}}.podium-score span,.podium-delta{{color:var(--muted)}}.podium-delta{{margin:3px 0 10px;font-size:.78rem;font-weight:800}}.podium-outcome{{min-height:45px;color:#f0e9fa;font-weight:750}}.athletic-stats{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:5px 0 15px}}.athletic-stats span{{padding:9px;border:1px solid #332b49;border-radius:9px;background:#110e1a}}.athletic-stats strong{{display:block;margin-top:2px;font-size:.86rem}}.podium-actions{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-top:auto}}.podium-actions button{{padding:8px 10px}}.all-experiments{{min-width:0;margin-top:16px;padding:0;border:1px solid #30283f;border-radius:13px;background:#0d0b13}}.all-experiments>summary{{display:flex;justify-content:space-between;padding:15px 17px;color:#b9afc8}}.all-experiments>summary span{{display:grid;place-items:center;min-width:24px;height:24px;padding:0 7px;border-radius:999px;background:#2a2337}}.all-experiments>p{{padding:0 17px}}.all-experiments>.finished-grid{{padding:0 12px 12px}}
.saved-dropdown{{margin:0;min-width:0;padding:0;border:0;border-top:1px solid var(--line);border-radius:0;background:#0d141b}}.saved-dropdown>summary{{display:flex;align-items:center;gap:9px;padding:13px 21px;cursor:pointer;color:#cfdae3;font-weight:700;list-style:none;user-select:none}}.saved-dropdown>summary::-webkit-details-marker{{display:none}}.saved-dropdown>summary:hover{{background:#111b24}}.summary-count{{display:grid;place-items:center;min-width:22px;height:22px;padding:0 6px;border-radius:999px;background:#243441;color:#b9d3e5;font-size:.72rem}}.chevron{{margin-left:auto;font-size:1.2rem;transition:transform .18s ease}}.saved-dropdown[open] .chevron{{transform:rotate(180deg)}}.saved-list{{display:grid;gap:8px;padding:0 12px 12px}}.saved-model{{display:grid;grid-template-columns:minmax(150px,1fr) minmax(250px,1.6fr) auto;align-items:center;gap:18px;padding:13px 14px;border:1px solid #263744;border-radius:10px;background:#131d25}}.saved-model-title{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}.stage-badge{{background:#203545;color:#bfe2fa}}.saved-model-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.saved-model-stats span{{font-weight:750}}.saved-model-actions{{display:flex;align-items:center;justify-content:flex-end;gap:10px}}.text-action{{font-weight:700;white-space:nowrap}}
.assistant-panel{{background:linear-gradient(145deg,#121b24,#101820)}}
.resource-control{{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:14px 0;padding:13px 14px;border:1px solid #2b4050;border-radius:11px;background:#101a22}}.resource-control select{{min-width:300px}}
.analysis-header{{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:4px 2px 22px;border-bottom:1px solid #29233a}}.wordmark{{font-size:1.15rem;font-weight:950;letter-spacing:.12em}}.header-subtitle{{margin-left:15px;color:var(--muted)}}.analysis-header .header-status{{padding:7px 11px;border-color:#343044;background:#11101a;font-size:.78rem}}
.live-training-panel{{margin:20px 0 0;padding:21px 23px;border:1px solid #3e3560;border-radius:17px;background:radial-gradient(circle at 92% 0,rgba(129,87,255,.17),transparent 34%),linear-gradient(145deg,#171326,#0d121b)}}.live-training-panel[hidden]{{display:none}}.live-training-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:24px}}.live-training-head h2{{margin:3px 0 2px;font-size:1.2rem}}.live-training-verdict{{max-width:760px;margin:4px 0 0;color:#b8afc5;font-size:.84rem}}.live-training-verdict[data-tone='good']{{color:#83ead3}}.live-training-verdict[data-tone='watch']{{color:#f3d38d}}.live-training-verdict[data-tone='bad']{{color:#ff9daa}}.live-training-count{{text-align:right}}.live-training-count strong,.live-training-count span{{display:block}}.live-training-count strong{{font-size:1.05rem;font-variant-numeric:tabular-nums}}.live-training-count span{{color:var(--muted);font-size:.76rem}}.live-training-progress{{height:8px;margin:16px 0;border-radius:999px;overflow:hidden;background:#282238}}.live-training-progress span{{display:block;width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--brand),var(--brand-2));transition:width .35s ease}}.live-training-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid #332c48;border-radius:11px;overflow:hidden;background:rgba(8,7,14,.52)}}.live-training-grid>div{{padding:12px 13px;border-right:1px solid #332c48}}.live-training-grid>div:last-child{{border:0}}.live-training-grid small,.live-training-grid strong{{display:block}}.live-training-grid small{{color:#948ba3;font-size:.7rem}}.live-training-grid strong{{margin-top:3px;font-size:.9rem;font-variant-numeric:tabular-nums}}.live-training-chart{{margin-top:12px;padding:11px 13px 5px;border:1px solid #302a42;border-radius:11px;background:#0b0a12}}.live-training-chart[hidden]{{display:none}}.live-training-chart>div{{display:flex;justify-content:space-between;gap:16px;color:#bcb3c9;font-size:.75rem}}.live-training-chart svg{{display:block;width:100%;height:150px;margin-top:4px}}.live-training-chart line{{stroke:#312b40;stroke-width:1}}.live-training-chart polyline{{fill:none;stroke:var(--brand-2);stroke-width:2.5;vector-effect:non-scaling-stroke}}
.leader-hero{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(420px,1fr);gap:26px;margin:28px 0 34px;padding:34px;border:1px solid #3a655b;border-radius:20px;background:radial-gradient(circle at 90% 8%,rgba(45,212,197,.16),transparent 38%),linear-gradient(135deg,#151326,#0d1918);box-shadow:0 24px 65px rgba(0,0,0,.25)}}.winner-label{{display:inline-block;padding:5px 9px;border-radius:7px;background:#17463e;color:#8cf0d6;font-size:.7rem;font-weight:900;letter-spacing:.09em}}.leader-hero h1{{max-width:720px;margin:15px 0 8px;font-size:clamp(2rem,4vw,3.4rem);line-height:1.02;overflow-wrap:anywhere}}.hero-copy>p{{max-width:720px;color:#c8c2d2;font-size:1rem}}.hero-actions{{display:flex;align-items:center;gap:16px;margin-top:21px}}.hero-actions button{{padding:12px 18px;background:linear-gradient(135deg,#7653f1,#5b36cb)}}.proof-link{{font-weight:750}}.hero-kpis{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}.hero-kpis>div{{display:flex;flex-direction:column;justify-content:center;min-height:112px;padding:16px;border:1px solid #31544d;border-radius:13px;background:rgba(7,20,18,.65)}}.hero-kpis strong{{margin:2px 0;font-size:1.9rem;line-height:1;font-variant-numeric:tabular-nums}}.hero-kpis span{{color:#9facaa;font-size:.78rem}}
#results{{display:grid;gap:24px}}.results-heading,.panel-heading{{display:flex;align-items:flex-end;justify-content:space-between;gap:26px}}.results-heading h2,.panel-heading h2{{margin:2px 0 0;font-size:1.45rem}}.results-heading>p,.panel-heading>p{{max-width:520px;margin:0;color:var(--muted);font-size:.83rem;text-align:right}}.role-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px}}.role-card{{display:flex;flex-direction:column;min-height:230px;padding:18px;border:1px solid #332d45;border-radius:14px;background:linear-gradient(150deg,#171421,#100e18)}}.role-card>small{{color:#8fe6d5}}.role-card h3{{margin:5px 0 13px;font-size:1rem;color:#bbb2c9}}.role-card>strong{{font-size:1.05rem}}.role-card p{{color:#aaa1b6;font-size:.83rem}}.role-card>div{{display:flex;align-items:center;gap:10px;margin-top:auto;padding-top:13px}}.role-card button{{padding:8px 10px}}.seed-card{{border-color:#63512e;background:linear-gradient(150deg,#211c12,#111016)}}.seed-card>small{{color:#f3c969}}
.next-move{{display:grid;grid-template-columns:210px 1fr;gap:22px;padding:19px 21px;border:1px solid #514177;border-radius:14px;background:#181326}}.next-move>span{{color:#bba7f7;font-size:.72rem;font-weight:900;letter-spacing:.09em}}.next-move strong{{color:#f6f1ff}}
.comparison-panel,.leaderboard-panel{{padding:23px;border:1px solid #302a41;border-radius:18px;background:rgba(17,14,26,.82)}}.comparison-panel{{display:grid;gap:10px}}.panel-heading{{margin-bottom:10px}}.comparison-row{{display:grid;grid-template-columns:175px minmax(0,1fr) 110px;align-items:center;gap:18px;padding:13px 0;border-top:1px solid #292338}}.comparison-name span{{display:block;color:var(--muted);font-size:.72rem}}.comparison-bars{{display:grid;gap:6px}}.comparison-bars>div{{display:grid;grid-template-columns:48px minmax(120px,1fr) 105px;align-items:center;gap:8px}}.comparison-bars small{{font-size:.62rem}}.comparison-bars b{{font-size:.75rem;font-variant-numeric:tabular-nums;text-align:right}}.bar-track{{display:block;height:8px;border-radius:999px;background:#252033;overflow:hidden}}.bar-track i{{display:block;height:100%;border-radius:999px}}.leader-bar{{background:linear-gradient(90deg,#7251ef,#2dd4c5)}}.pollen-bar{{background:#746b80}}.win-chip{{justify-self:end;padding:4px 7px;border-radius:6px;font-size:.68rem;font-weight:900;font-variant-numeric:tabular-nums}}.win-chip.win{{background:#153e37;color:#8becd4}}.win-chip.loss{{background:#4b2029;color:#ffadb8}}
.leaderboard-panel{{padding:23px 0 0;overflow:hidden}}.leaderboard-panel .panel-heading{{padding:0 23px 12px}}.leaderboard-panel .table-wrap{{border:0;border-top:1px solid #302a41;border-radius:0}}.definitive-table{{min-width:1160px;font-size:.83rem}}.definitive-table th{{padding:11px 10px;background:#12101b;color:#8f879d}}.definitive-table td{{padding:14px 10px;vertical-align:middle;font-variant-numeric:tabular-nums}}.rank-cell{{color:#8f879d;font-weight:800}}.model-cell strong,.model-cell span{{display:block}}.model-cell strong{{max-width:245px}}.model-cell span{{margin-top:3px;color:#938aa0;font-size:.7rem}}.record{{color:#8cebd3}}.best-row td{{background:rgba(23,70,62,.24)}}.best-row td:first-child{{box-shadow:inset 3px 0 #2dd4c5}}.reference-row{{color:#9991a5}}.reference-row td{{background:#0e0d14}}.table-play{{padding:6px 9px;font-size:.76rem}}footer{{padding:28px 2px 0;color:#777080;font-size:.76rem;text-align:center}}
@media(max-width:1000px){{.metric-grid{{grid-template-columns:repeat(3,1fr)}}.metric-grid>div:nth-child(3){{border-right:0}}.metric-grid>div:nth-child(-n+3){{border-bottom:1px solid var(--line)}}.podium-grid{{grid-template-columns:1fr}}.podium-card{{min-height:0}}}}
@media(max-width:1050px){{.leader-hero{{grid-template-columns:1fr}}.role-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:900px){{.product-header{{align-items:flex-start;flex-direction:column;min-height:0}}.header-side{{align-items:flex-start;max-width:none}}.header-status{{text-align:left}}.saved-model{{grid-template-columns:1fr}}.saved-model-actions{{justify-content:flex-start}}.comparison-row{{grid-template-columns:145px minmax(0,1fr)}}.win-chip{{grid-column:2;justify-self:start}}}}
@media(max-width:720px){{body{{padding:14px 13px 60px}}.analysis-header,.results-heading,.panel-heading,.live-training-head{{align-items:flex-start;flex-direction:column}}.analysis-header .header-status,.live-training-count{{text-align:left}}.live-training-panel{{padding:18px 15px}}.live-training-grid{{grid-template-columns:1fr 1fr}}.live-training-grid>div{{border-bottom:1px solid #332c48}}.live-training-chart svg{{height:120px}}.leader-hero{{padding:22px 18px}}.hero-kpis,.role-grid{{grid-template-columns:1fr}}.hero-actions{{align-items:flex-start;flex-direction:column}}.results-heading>p,.panel-heading>p{{text-align:left}}.next-move{{grid-template-columns:1fr;gap:7px}}.comparison-row{{grid-template-columns:1fr}}.comparison-bars>div{{grid-template-columns:44px minmax(90px,1fr) 92px}}.win-chip{{grid-column:1}}.product-header{{padding:21px 18px;border-radius:18px}}.brand-lockup{{align-items:flex-start;gap:14px}}.duck-mark{{flex-basis:52px;height:52px;border-radius:14px;font-size:.95rem}}.quick-nav{{justify-content:flex-start;margin-top:10px}}.section-title,.section-heading,.run-card,.session-card,.finished-card-top,.resource-control,.training-verdict,.result-banner,.scoreboard-verdict,.baseline-reference,.baseline-duel-head{{align-items:flex-start;flex-direction:column}}.baseline-numbers,.baseline-improvement{{text-align:left}}.baseline-comparison-grid{{grid-template-columns:1fr}}.section-note{{text-align:left}}.session-actions,.launch-cluster{{width:100%;flex-wrap:wrap}}.session-actions a,.session-actions button,.run-card>button,.launch-cluster button,.launch-cluster a{{width:100%;text-align:center}}.resource-control select{{width:100%;min-width:0}}.run-stats,.metric-grid,.result-kpis{{grid-template-columns:1fr}}.run-stats>div,.metric-grid>div{{border-right:0;border-bottom:1px solid var(--line)}}.skill-signals{{grid-template-columns:repeat(2,1fr)}}.saved-model-stats{{grid-template-columns:repeat(3,1fr)}}.saved-model-actions{{flex-wrap:wrap}}form{{flex-direction:column}}}}
</style></head>
<body>{heading}{body}</body></html>"""


def metric_svg(tag: str, points: list[dict[str, float]]) -> str:
    if not points:
        return ""
    width, height, pad = 720, 180, 28
    values = [point["value"] for point in points]
    low, high = min(values), max(values)
    span = high - low or 1.0
    polyline = " ".join(
        f"{pad + index * (width - 2 * pad) / max(1, len(points) - 1):.1f},"
        f"{height - pad - (point['value'] - low) / span * (height - 2 * pad):.1f}"
        for index, point in enumerate(points)
    )
    return (
        f"<div class='chart'><strong>{html.escape(tag)}</strong> "
        f"<span class='muted'>step {int(points[0]['step'])} → {int(points[-1]['step'])}; {low:.4g} → {high:.4g}</span>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(tag)} curve'>"
        f"<polyline fill='none' stroke='#65c7ff' stroke-width='2' points='{polyline}'/>"
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#34404b'/></svg></div>"
    )


def render_comparison_html(result: dict[str, Any], output: Path) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(row['metric'])}</td><td>{row['baseline']:.6g}</td>"
        f"<td>{row['candidate']:.6g}</td><td>{row['delta']:+.6g}</td></tr>"
        for row in result["metrics"]
    )
    output.write_text(page(
        f"{result['candidate']} vs {result['baseline']}",
        f"<p>Suite: {html.escape(result['suite'])}</p><p>{html.escape(result['note'])}</p>"
        f"<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr>{rows}</table>",
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Register training runs under an RSL-RL log root")
    discover.add_argument("--logs-root", type=Path, default=UPSTREAM / "logs" / "rsl_rl")
    discover.add_argument("--task")
    register = commands.add_parser("register", help="Register or refresh one training run")
    register.add_argument("run_dir", type=Path)
    register.add_argument("--task")
    listing = commands.add_parser("list", help="List registered runs")
    listing.add_argument("--task")
    listing.add_argument("--latest", action="store_true", help="Print only the newest matching candidate")
    listing.add_argument("--all", action="store_true", help="Include archived runs")
    attach = commands.add_parser("attach-eval", help="Attach evaluation JSON to a run")
    attach.add_argument("run_id")
    attach.add_argument("metrics", type=Path)
    attach.add_argument("--suite", default="skating-v1")
    evaluate = commands.add_parser("evaluate", help="Run the deployment-rehearsal evaluation")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--suite", default="skating-v1")
    metrics = commands.add_parser("metrics", help="Ingest TensorBoard scalar curves for a run")
    metrics.add_argument("run_id")
    score = commands.add_parser("score", help="Compute the transparent heuristic score")
    score.add_argument("run_id")
    score.add_argument("--suite", default="skating-v1")
    star = commands.add_parser("star", help="Star one candidate for a task")
    star.add_argument("run_id")
    star.add_argument("--note", default="")
    unstar = commands.add_parser("unstar", help="Remove a candidate star")
    unstar.add_argument("run_id")
    archive = commands.add_parser("archive", help="Hide a legacy run without deleting it")
    archive.add_argument("run_id")
    archive.add_argument("--note", default="")
    unarchive = commands.add_parser("unarchive", help="Restore an archived run")
    unarchive.add_argument("run_id")
    compare = commands.add_parser("compare", help="Compare two runs evaluated by the same suite")
    compare.add_argument("candidate")
    compare.add_argument("baseline")
    compare.add_argument("--suite", default="skating-v1")
    promote = commands.add_parser("promote", help="Promote a reviewed run by one stage")
    promote.add_argument("run_id")
    promote.add_argument("stage", choices=STAGES)
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--note", required=True)
    promote.add_argument("--hardware-signoff", action="store_true")
    resolve = commands.add_parser("resolve", help="Resolve and verify a promoted artifact")
    resolve.add_argument("task")
    resolve.add_argument("--stage", default="sim-qualified", choices=STAGES)
    resolve.add_argument("--artifact", default="checkpoint", choices=("checkpoint", "policy"))
    commands.add_parser("dashboard", help="Regenerate the local HTML dashboard")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bench = Bench(args.state_dir)
    bench.initialize()
    if args.command == "discover":
        runs = bench.discover(args.logs_root, args.task)
        print(f"Registered {len(runs)} runs. Dashboard: {bench.state_dir / 'index.html'}")
    elif args.command == "register":
        manifest = bench.register(args.run_dir, args.task)
        bench.render_dashboard()
        print(manifest["run_id"])
    elif args.command == "list":
        manifests = [
            manifest for manifest in bench.manifests()
            if (args.all or not manifest.get("archived", False))
            and (not args.task or manifest["task"] == args.task)
        ]
        if args.latest and manifests:
            manifests = [max(manifests, key=lambda item: item["created_at"])]
        for manifest in manifests:
            print(f"{manifest['run_id']}\t{manifest['task']}\t{manifest['stage']}\t{manifest.get('latest_iteration')}")
    elif args.command == "attach-eval":
        record = bench.attach_evaluation(args.run_id, args.metrics, args.suite)
        print(record["path"])
    elif args.command == "evaluate":
        record = bench.evaluate(args.run_id, args.suite)
        print(record["path"])
    elif args.command == "metrics":
        print(bench.metrics(args.run_id))
    elif args.command == "score":
        print(json.dumps(bench.score(args.run_id, args.suite), indent=2, sort_keys=True))
    elif args.command == "star":
        print(f"Starred {bench.star(args.run_id, args.note)['run_id']}")
    elif args.command == "unstar":
        print(f"Unstarred {bench.unstar(args.run_id)['run_id']}")
    elif args.command == "archive":
        print(f"Archived {bench.archive(args.run_id, args.note)['run_id']}")
    elif args.command == "unarchive":
        print(f"Restored {bench.unarchive(args.run_id)['run_id']}")
    elif args.command == "compare":
        result = bench.compare(args.candidate, args.baseline, args.suite)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "promote":
        manifest = bench.promote(args.run_id, args.stage, args.approved_by, args.note, args.hardware_signoff)
        print(f"{manifest['run_id']} -> {manifest['stage']}")
    elif args.command == "resolve":
        print(bench.resolve(args.task, args.stage, args.artifact))
    elif args.command == "dashboard":
        print(bench.render_dashboard())


if __name__ == "__main__":
    main()
