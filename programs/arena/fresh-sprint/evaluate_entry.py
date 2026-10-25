#!/usr/bin/env python3
"""Observe the pinned official Arena evaluator without changing its policy or physics.

One unchanged candidate is evaluated per invocation. This is CPU competition
qualification, not training, an entry submission, or physical robot validation.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
import zipfile

OFFICIAL_REVISION = "221dc512f64aad0814251062df0511543ab2b3b4"
RUNTIME_VERSIONS = {"numpy": "2.4.1", "mujoco": "3.10.0", "onnxruntime": "1.24.4"}
SPEEDS = tuple(round(0.3 + 0.1 * i, 4) for i in range(18))


class QualificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    """Never publish NaN as a JSON number; retain trace arrays separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise QualificationError(f"source escapes root: {relative}")
    return path


def verify_source(root: Path, reference: dict) -> dict:
    if reference.get("revision") != OFFICIAL_REVISION:
        raise QualificationError("official Arena revision differs from frozen runner V4 reference")
    files = reference.get("files")
    if not isinstance(files, dict) or not files:
        raise QualificationError("official source hashes are required")
    required = {
        "arena/__init__.py", "arena/evaluate.py", "arena/runner.py", "arena/actuator.py",
        "arena/constants.py", "arena/course.py", "arena/policy.py", "arena/manifest.py",
        "arena/timeline.py", "arena/events/roller-sprint-2m.toml",
        "arena/golden.py", "arena/events/sprint-2m.toml",
        "arena/assets/policies/BEST_alpha_walking.onnx",
        "arena/assets/robot/rollers/robot_allcollisions_rollers.xml",
    }
    if not required.issubset(files):
        raise QualificationError(f"missing official source hashes: {sorted(required - files.keys())}")
    observed = {}
    for relative, expected in sorted(files.items()):
        path = checked_path(root, relative)
        actual = sha256(path)
        if actual != expected:
            raise QualificationError(f"official source SHA mismatch: {relative}")
        observed[relative] = actual
    # Assets can affect a score as much as Python. Every vendored roller asset
    # must be covered, and all official Python modules must be covered too.
    needed = {
        path.relative_to(root).as_posix()
        for pattern in ("arena/*.py", "arena/assets/**/*")
        for path in root.glob(pattern) if path.is_file()
    }
    if not needed.issubset(files):
        raise QualificationError(f"unhashed official runtime files: {sorted(needed - files.keys())}")
    return observed


def verify_manifest(manifest) -> None:
    expected = {
        "id": "roller-sprint-2m", "rig": "rollers", "kind": "perpetual",
        "time_limit_s": 40.0, "metric": "mean_finish_time", "seeds": (1, 2, 3, 4, 5),
    }
    for key, value in expected.items():
        if getattr(manifest, key) != value:
            raise QualificationError(f"official event {key} differs from frozen contract")
    if manifest.sweep.speeds() != SPEEDS or manifest.sweep.finalists != 3:
        raise QualificationError("official speed sweep differs from 18 speeds / 3 finalists")
    if dataclasses.asdict(manifest.course) != {
        "kind": "lane", "length_m": 2.0, "width_m": 0.8,
        "wall_height_m": 0.15, "slope_deg": 0.0,
    }:
        raise QualificationError("official roller course differs from frozen contract")
    if dataclasses.asdict(manifest.start) != {
        "keyframe": "STAND", "settle_s": 0.5, "joint_rad": 0.01, "yaw_deg": 2.0,
    }:
        raise QualificationError("official roller start differs from frozen contract")
    if dataclasses.asdict(manifest.termination) != {
        "fall_gravity_z": -0.5, "fall_hold_s": 0.3,
    }:
        raise QualificationError("official fall termination differs from frozen contract")
    if tuple(manifest.command) != (0.6,) + (0.0,) * 12:
        raise QualificationError("official event command differs from frozen contract")


def finite_number(value) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def outcome_record(outcome, speed: float, trace_path: Path, samples=None) -> dict:
    """Preserve all outcomes, including failed and non-finite traces."""
    import numpy as np
    trace = np.asarray(outcome.trace)
    settle = np.asarray(outcome.settle)
    problems = []
    if trace.ndim != 2 or trace.shape[1] != 21 or trace.shape[0] < 1:
        problems.append("invalid_trace_shape")
    if settle.shape != (26, 21):
        problems.append("invalid_settle_shape")
    if not np.isfinite(trace).all() or not np.isfinite(settle).all():
        problems.append("nonfinite_trace_or_settle")
    if not math.isfinite(float(outcome.progress_m)) or not math.isfinite(float(outcome.start_progress_m)):
        problems.append("nonfinite_progress")
    if outcome.status not in {"finish", "fall", "timeout"}:
        problems.append("unexpected_status")
    if outcome.status == "finish":
        if outcome.time_s is None or not math.isfinite(float(outcome.time_s)) or not 0 < outcome.time_s <= 40:
            problems.append("invalid_finish_time")
        if not math.isfinite(float(outcome.progress_m)) or outcome.progress_m < 2:
            problems.append("finish_without_two_metres")
    elif outcome.time_s is not None:
        problems.append("dnf_has_finish_time")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"trace": trace, "settle": settle}
    if samples is not None:
        saved["sampled_observations"] = np.asarray(samples.observations, dtype=np.float32)
        saved["sampled_actions"] = np.asarray(samples.actions, dtype=np.float32)
        if saved["sampled_observations"].shape != (16, 61) or saved["sampled_actions"].shape != (16, 14):
            problems.append("invalid_sampled_policy_io_shape")
        if not np.isfinite(saved["sampled_observations"]).all() or not np.isfinite(saved["sampled_actions"]).all():
            problems.append("nonfinite_sampled_policy_io")
    np.savez_compressed(trace_path, **saved)
    return {
        "command_vx": float(speed), "seed": int(outcome.seed), "status": outcome.status,
        "time_s": finite_number(outcome.time_s), "progress_m": finite_number(outcome.progress_m),
        "start_progress_m": finite_number(outcome.start_progress_m),
        "trace_shape": list(trace.shape), "settle_shape": list(settle.shape),
        "trace_dtype": str(trace.dtype), "settle_dtype": str(settle.dtype),
        "trace_file": trace_path.name, "trace_sha256": sha256(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "policy_io_sampling": "first 16 policy calls of each distinct run, including settle; incomplete rollout" if samples else None,
        "valid": not problems, "validation_problems": problems,
    }


class ObserveRuns:
    """Forward identical arguments and return the original outcome unchanged."""
    def __init__(self, original, output: Path, budget_s: float, started: float, samples=None):
        self.original = original
        self.output = output
        self.budget_s = budget_s
        self.started = started
        self.records = []
        self.samples = samples

    def __call__(self, *args, **kwargs):
        if time.monotonic() - self.started > self.budget_s:
            raise QualificationError("bounded evaluation wall-clock budget exhausted between runs")
        command = kwargs.get("command")
        manifest = args[0]
        seed = args[2] if len(args) > 2 else kwargs["seed"]
        speed = float((manifest.command if command is None else command)[0])
        progress = {"phase": "run_start", "run": len(self.records) + 1,
                    "command_vx": speed, "seed": seed}
        write_json(self.output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if self.samples is not None:
            self.samples.reset()
        outcome = self.original(*args, **kwargs)
        path = self.output / "raw-traces" / f"run-{len(self.records)+1:02d}-vx-{speed:.4f}-seed-{seed}.npz"
        record = outcome_record(outcome, speed, path, self.samples)
        record["elapsed_wall_s"] = time.monotonic() - self.started
        self.records.append(record)
        write_json(self.output / "outcomes.json", self.records)
        write_json(self.output / "progress.json", {"phase": "run_done", "completed_runs": len(self.records), **record})
        print(json.dumps({"phase": "run_done", **{key: record[key] for key in
              ("command_vx", "seed", "status", "time_s", "progress_m", "valid")}}), flush=True)
        return outcome


class ObservePolicy:
    """Copy a bounded IO sample after delegating; never replace the returned action."""
    def __init__(self, original):
        self.original = original
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []

    def __call__(self, observation):
        action = self.original(observation)
        if len(self.observations) < 16:
            self.observations.append(observation.copy())
            self.actions.append(action.copy())
        return action


def evaluation_record(evaluation) -> dict:
    converted = dataclasses.asdict(evaluation)
    return {key: value for key, value in converted.items() if key not in {"trace", "settle"}}


def validate_golden(evaluation, system: str, machine: str) -> None:
    if (system, machine) != ("Linux", "x86_64"):
        raise QualificationError("competition qualification requires the official Linux-x86_64 golden platform")
    if evaluation.runner_version != 4 or evaluation.seeds_finished != 5:
        raise QualificationError("golden reference did not reproduce runner V4 with 5/5 finishes")
    if abs(evaluation.command_vx - 0.45) > 1e-8:
        raise QualificationError("golden reference chose a different speed than 0.45 m/s")
    if not math.isfinite(evaluation.score) or abs(evaluation.score - 9.404) > 9.404 * 0.02:
        raise QualificationError("golden reference score differs from 9.404 s by more than 2 percent")


def validate_evaluation(evaluation, records: list[dict], manifest, official_finalists) -> dict:
    """Validate evidence coverage separately from success at the racing task."""
    problems = []
    shortlist = official_finalists(evaluation.scan, 3)
    recorded = {(record["command_vx"], record["seed"]): record for record in records}
    if len(records) != 30 or len(recorded) != 30:
        problems.append("expected_30_distinct_runs")
    if tuple(point.command_vx for point in evaluation.scan) != SPEEDS:
        problems.append("scan_grid_incomplete")
    if not all(record["valid"] for record in records):
        problems.append("invalid_run_evidence")
    if evaluation.runner_version != 4:
        problems.append("runner_version_is_not_4")
    if not math.isfinite(float(evaluation.score)):
        problems.append("nonfinite_score")
    finalists = []
    for speed in shortlist:
        outcomes = [recorded.get((speed, seed)) for seed in manifest.seeds]
        if any(outcome is None for outcome in outcomes):
            problems.append("missing_finalist_seed")
            continue
        score = statistics.fmean(outcome["time_s"] if outcome["time_s"] is not None else 40.0
                                 for outcome in outcomes)
        finalists.append({"command_vx": speed, "score_s": score,
                          "seeds_finished": sum(outcome["status"] == "finish" for outcome in outcomes),
                          "per_seed": [{key: outcome[key] for key in ("seed", "status", "time_s", "progress_m")}
                                       for outcome in outcomes]})
    if len(finalists) != 3:
        problems.append("expected_three_full_finalists")
    elif min((item["score_s"], item["command_vx"]) for item in finalists) != (evaluation.score, evaluation.command_vx):
        problems.append("official_best_score_does_not_match_recorded_runs")
    expected_best = [recorded.get((evaluation.command_vx, seed)) for seed in manifest.seeds]
    if any(record is None for record in expected_best):
        problems.append("missing_selected_seed")
    else:
        actual_finished = sum(record["status"] == "finish" for record in expected_best)
        if actual_finished != evaluation.seeds_finished:
            problems.append("selected_finish_count_mismatch")
        for result, expected in zip(evaluation.per_seed, expected_best):
            if (result.seed, result.status, result.time_s) != (expected["seed"], expected["status"], expected["time_s"]):
                problems.append("selected_per_seed_mismatch")
    return {"evidence_valid": not problems, "validation_problems": sorted(set(problems)),
            "finalists": finalists,
            "submission_readiness_advisory": not problems and evaluation.seeds_finished == 5,
            "readiness_rule": "valid official full sweep plus 5/5 finishes at selected speed"}


def package_evidence(output: Path, max_bytes: int = 8 * 1024 * 1024) -> None:
    """ZIP complete small evidence if it fits; keep all raw files in worker scratch."""
    paths = sorted(path for path in output.rglob("*")
                   if path.is_file() and path.name not in {"evidence.zip", "evidence-index.json"}
                   and not path.name.endswith(".partial"))
    items = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
              "sha256": sha256(path), "retained_worker_path": str(path.resolve())} for path in paths]
    included = []
    # Bound by uncompressed bytes too; filenames and ZIP overhead cannot grow
    # unbounded because the evaluator produces at most 30 traces plus reports.
    with zipfile.ZipFile(output / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        total = 0
        for item in sorted(items, key=lambda value: (value["path"].endswith(".npz"), value["path"])):
            if total + item["bytes"] > max_bytes - 64 * 1024:
                continue
            archive.write(output / item["path"], item["path"])
            total += item["bytes"]
            included.append(item["path"])
    if (output / "evidence.zip").stat().st_size > max_bytes:
        raise QualificationError("evidence archive exceeded declared bound")
    write_json(output / "evidence-index.json", {
        "worker_output": str(output.resolve()), "files": items, "archived": included,
        "omitted_but_retained": [item["path"] for item in items if item["path"] not in included],
        "archive_sha256": sha256(output / "evidence.zip"),
        "archive_bytes": (output / "evidence.zip").stat().st_size,
        "archive_limit_bytes": max_bytes,
    })


def run(contract_path: Path, candidate_id: str, output: Path) -> int:
    started = time.monotonic()
    if output.exists() and any(output.iterdir()):
        raise QualificationError("output directory is not empty; preserve earlier evidence")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    state = {"experiment": "arena-roller-entry-r1", "candidate_id": candidate_id,
             "state": "initializing", "training": False, "submitted": False,
             "hardware_validation": False, "competition_win_claim": False,
             "compute_path": "official CPU MuJoCo/BAM and CPU ONNXRuntime"}
    write_json(output / "status.json", state)
    write_json(output / "runtime.json", {"state": "initializing", "candidate_id": candidate_id})
    write_json(output / "trace-index.json", {"traces": [], "state": "initializing"})
    observer = None
    try:
        contract_path = contract_path.resolve()
        contract = json.loads(contract_path.read_text())
        root = contract_path.parent
        source = contract["official_source"]
        arena_root = checked_path(root, source.get("path", "official-arena"))
        source_hashes = verify_source(arena_root, source)
        candidate = contract["candidates"][candidate_id]
        model_path = checked_path(root, candidate["path"])
        model_sha = sha256(model_path)
        if model_sha != candidate["sha256"]:
            raise QualificationError("candidate SHA differs from unchanged source model")
        actual_versions = {package: importlib.metadata.version(package) for package in RUNTIME_VERSIONS}
        expected_versions = contract.get("runtime_versions", RUNTIME_VERSIONS)
        if expected_versions != RUNTIME_VERSIONS or actual_versions != RUNTIME_VERSIONS:
            raise QualificationError(f"official runtime version mismatch: {actual_versions}")
        sys.path.insert(0, str(arena_root))
        official = importlib.import_module("arena.evaluate")
        manifest_module = importlib.import_module("arena.manifest")
        policy_module = importlib.import_module("arena.policy")
        for module in (official, manifest_module, policy_module):
            if not Path(module.__file__).resolve().is_relative_to(arena_root):
                raise QualificationError("loaded Arena module is outside pinned source root")
        manifest = manifest_module.load_event("roller-sprint-2m")
        verify_manifest(manifest)
        policy = policy_module.load_policy(model_path)
        if policy.sha256 != model_sha:
            raise QualificationError("official loader SHA differs from frozen candidate")
        provenance = {
            "candidate_id": candidate_id, "candidate": candidate, "observed_model_sha256": model_sha,
            "official_revision": OFFICIAL_REVISION, "official_source_sha256": source_hashes,
            "contract_sha256": sha256(contract_path), "runner_wrapper_sha256": sha256(Path(__file__)),
            "runtime_versions": actual_versions, "python": sys.version, "platform": platform.platform(),
            "onnx_providers": policy.session.get_providers(),
            "cpu_count": os.cpu_count(), "event": dataclasses.asdict(manifest),
            "cpu_qualification_exception": "Official competition runner uses CPU; no GPU training occurs.",
        }
        write_json(output / "runtime.json", provenance)
        write_json(output / "contract.json", contract)
        state["state"] = "validating_official_golden"
        write_json(output / "status.json", state)
        write_json(output / "progress.json", {"phase": "golden_start"})
        print(json.dumps({"phase": "golden_start"}), flush=True)
        if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
            raise QualificationError("competition qualification requires Linux-x86_64")
        golden_policy = policy_module.load_policy(arena_root / "arena/assets/policies/BEST_alpha_walking.onnx")
        golden = official.evaluate(manifest_module.load_event("sprint-2m"), golden_policy)
        write_json(output / "golden.json", {
            **evaluation_record(golden), "reference_score_s": 9.404, "reference_command_vx": 0.45,
            "reference_tolerance_fraction": 0.02, "golden_policy_sha256": golden_policy.sha256,
        })
        validate_golden(golden, platform.system(), platform.machine())
        print(json.dumps({"phase": "golden_passed", "score_s": golden.score,
                          "command_vx": golden.command_vx, "seeds_finished": golden.seeds_finished}), flush=True)
        budget = float(contract.get("budget_s", 600))
        sampled_policy = ObservePolicy(policy.act)
        observer = ObserveRuns(official.run_once, output, budget, started, sampled_policy)
        state["state"] = "evaluating_official_full_sweep"
        write_json(output / "status.json", state)
        original = official.run_once
        original_act = policy.act
        policy.act = sampled_policy
        official.run_once = observer
        try:
            evaluation = official.evaluate(manifest, policy)
        finally:
            official.run_once = original
            policy.act = original_act
        # Source and candidate are checked again after the official scorer ran.
        verify_source(arena_root, source)
        if sha256(model_path) != model_sha:
            raise QualificationError("candidate changed during evaluation")
        evaluation_json = evaluation_record(evaluation)
        write_json(output / "official-evaluation.json", evaluation_json)
        import numpy as np
        np.savez_compressed(output / "selected-replay.npz", trace=evaluation.trace, settle=evaluation.settle)
        qualification = validate_evaluation(evaluation, observer.records, manifest, official.finalists)
        summary = {
            **state, "state": "complete", "elapsed_wall_s": time.monotonic() - started,
            "candidate_sha256": model_sha, "official_revision": OFFICIAL_REVISION,
            "runner_version": evaluation.runner_version, "event_id": evaluation.event_id,
            "score_s": evaluation.score, "command_vx": evaluation.command_vx,
            "seeds_finished": evaluation.seeds_finished, "per_seed": evaluation_json["per_seed"],
            "scan": evaluation_json["scan"], "trace_seed": evaluation.trace_seed,
            "distinct_runs": len(observer.records), **qualification,
            "interpretation": "Local reproduction of pinned Arena V4; official live submission score may differ if its revision changes.",
            "baseline_models_modified": False,
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", summary)
        write_json(output / "trace-index.json", {
            "traces": [{key: record[key] for key in ("command_vx", "seed", "status", "trace_file", "trace_sha256", "trace_bytes", "policy_io_sampling")}
                       for record in observer.records],
            "selected_replay": {"path": "selected-replay.npz", "sha256": sha256(output / "selected-replay.npz")},
            "trace_frames": "official qpos-only 21-value replay frames; no full qvel/servo-state snapshots",
            "policy_io": "first 16 observed policy calls per run; original action object returned unchanged",
        })
        write_json(output / "progress.json", {"phase": "complete", "completed_runs": len(observer.records)})
        package_evidence(output)
        print(json.dumps({"phase": "complete", "candidate_id": candidate_id,
                          "score_s": evaluation.score, "seeds_finished": evaluation.seeds_finished,
                          "evidence_valid": qualification["evidence_valid"],
                          "submission_readiness_advisory": qualification["submission_readiness_advisory"]}), flush=True)
        return 0 if qualification["evidence_valid"] else 2
    except Exception as exc:
        state.update({"state": "failed", "error_type": type(exc).__name__, "error": str(exc),
                      "elapsed_wall_s": time.monotonic() - started,
                      "completed_runs": len(observer.records) if observer else 0,
                      "evidence_valid": False, "submission_readiness_advisory": False})
        write_json(output / "summary.json", state)
        write_json(output / "status.json", state)
        write_json(output / "progress.json", {"phase": "failed", "completed_runs": state["completed_runs"]})
        (output / "failure.txt").write_text(traceback.format_exc())
        package_evidence(output)
        print(json.dumps({"phase": "failed", **state}), flush=True)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    return run(args.contract, args.candidate, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
