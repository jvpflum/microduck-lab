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


LAB_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = LAB_ROOT / "upstream" / "microduck_rl"
DEFAULT_STATE = LAB_ROOT / "policy-bench"
SCHEMA_VERSION = 1
STAGES = ("experimental", "evaluated", "sim-qualified", "hardware-candidate", "production")


def bounded_score(value: float, target: float, tolerance: float) -> float:
    return max(0.0, min(1.0, 1.0 - abs(value - target) / tolerance))


def score_evaluation(evaluation: dict[str, Any], task: str) -> dict[str, Any]:
    """Return a transparent 0-100 heuristic; it is not an auto-promotion gate."""
    phases = evaluation.get("phases", {})
    forward = phases.get("forward", {})
    reverse = phases.get("reverse", {})
    powered = [phases[name] for name in ("forward", "reverse", "heading_left", "heading_right") if name in phases]
    if not forward:
        return {"overall": None, "label": "not scorable", "components": {}, "weights": {}}
    components = {
        "forward_tracking": bounded_score(float(forward.get("mean_forward_speed_mps", 0.0)), 0.3, 0.3),
        "ground_contact": sum(float(item.get("both_blades_grounded_fraction", 0.0)) for item in powered) / max(1, len(powered)),
        "stability": max(0.0, 1.0 - max(float(item.get("tilt_max_deg", 90.0)) for item in powered) / 20.0),
        "smoothness": max(0.0, 1.0 - sum(float(item.get("mean_action_acceleration", 1.0)) for item in powered) / max(1, len(powered)) / 0.08),
        "low_lateral_slip": max(0.0, 1.0 - sum(float(item.get("mean_abs_lateral_speed_mps", 1.0)) for item in powered) / max(1, len(powered)) / 0.1),
    }
    weights = {"forward_tracking": 0.25, "ground_contact": 0.2, "stability": 0.2, "smoothness": 0.15, "low_lateral_slip": 0.2}
    if task == "swizzle" and reverse:
        components["reverse_tracking"] = bounded_score(abs(float(reverse.get("mean_forward_speed_mps", 0.0))), 0.3, 0.3)
        components["swizzle_cycles"] = min(1.0, sum(float(phases.get(name, {}).get("estimated_swizzle_cycles", 0.0)) for name in ("forward", "reverse")) / 16.0)
        weights = {"forward_tracking": 0.2, "reverse_tracking": 0.2, "ground_contact": 0.15, "stability": 0.15, "smoothness": 0.1, "low_lateral_slip": 0.1, "swizzle_cycles": 0.1}
    overall = 100.0 * sum(components[key] * weights[key] for key in weights) / sum(weights.values())
    return {"overall": round(overall, 2), "label": "heuristic simulation score", "components": {key: round(value * 100.0, 2) for key, value in components.items()}, "weights": weights}


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
        "velocity_rollers": "roller",
        "velocity": "walking",
    }.get(parent, parent.replace("velocity_", "") or "unknown")


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


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

    def attach_evaluation(self, run_id: str, metrics_path: Path, suite: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        metrics_path = metrics_path.resolve()
        metrics = read_json(metrics_path)
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

    def evaluate(self, run_id: str, suite: str) -> dict[str, Any]:
        manifest = self.load_manifest(run_id)
        policy = manifest["artifacts"].get("policy")
        if not policy:
            raise SystemExit(f"Run {run_id} has no exported ONNX policy")
        if manifest["task"] not in {"swizzle", "roller"}:
            raise SystemExit(f"No evaluator is registered for task {manifest['task']!r}")
        policy_path = Path(policy["path"])
        if not policy_path.is_file() or sha256(policy_path) != policy["sha256"]:
            raise SystemExit(f"Policy snapshot is missing or corrupt: {policy_path}")
        output = self.runs_dir / run_id / "evaluations" / f"{suite}.json"
        evaluator = LAB_ROOT / "tools" / "evaluate_swizzle.py"
        uv = LAB_ROOT / ".tools" / "uv" / "bin" / "uv"
        if not uv.is_file():
            raise SystemExit("DuckLab uv environment is missing; run ./scripts/bootstrap.sh")
        subprocess.run(
            [str(uv), "run", str(evaluator), str(policy_path), "--output", str(output)],
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
        rows = "".join(
            f"<tr><td>{html.escape(item['suite'])}</td><td>{html.escape(item['created_at'])}</td>"
            f"<td><a href='{html.escape(os.path.relpath(item['path'], self.runs_dir / run_id))}'>JSON</a></td></tr>"
            for item in manifest.get("evaluations", [])
        ) or "<tr><td colspan='3'>No evaluations attached</td></tr>"
        comparison_rows = "".join(
            f"<li><a href='{html.escape(os.path.relpath(path, self.runs_dir / run_id))}'>{html.escape(path.stem)}</a></li>"
            for path in sorted((self.runs_dir / run_id / "comparisons").glob("*.html"))
        ) or "<li>No comparisons generated</li>"
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
            f"<p>Iteration: {manifest.get('latest_iteration')} · ONNX export: {manifest.get('has_exported_policy')}</p>"
            f"<p class='mono'>{html.escape(manifest['source_run_dir'])}</p>"
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
        all_manifests = sorted(self.manifests(), key=lambda item: item["created_at"], reverse=True)
        upstream_source = git_revision(UPSTREAM)
        upstream_commit = str(upstream_source.get("commit") or "unknown")[:8]
        upstream_branch = str(upstream_source.get("branch") or "detached")
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
            state = "active" if newest.get("experiment_kind") == "training" and label in active_experiments else "finished/snapshot"
            # Retries and discovery passes can snapshot the same iteration
            # more than once. That is useful provenance, but it is not another
            # user-facing saved model.
            distinct_versions: dict[int | None, dict[str, Any]] = {}
            for manifest in sorted(group, key=lambda item: item.get("created_at", ""), reverse=True):
                distinct_versions.setdefault(manifest.get("latest_iteration"), manifest)
            snapshots = []
            for manifest in sorted(distinct_versions.values(), key=lambda item: item.get("latest_iteration") or -1, reverse=True):
                score = None
                for evaluation in manifest.get("evaluations", []):
                    try:
                        evaluation_data = read_json(Path(evaluation["path"]))
                        score = evaluation_data.get("policy_bench_score", {}).get("overall")
                        if score is None:
                            score = score_evaluation(evaluation_data, manifest["task"]).get("overall")
                    except (OSError, KeyError, TypeError, json.JSONDecodeError):
                        pass
                snapshots.append(
                    "<article class='saved-model'>"
                    "<div class='saved-model-title'>"
                    f"<strong>Iteration {manifest.get('latest_iteration')}</strong>"
                    f"<span class='stage-badge'>{html.escape(manifest['stage'])}</span></div>"
                    "<div class='saved-model-stats'>"
                    f"<span><small>Score</small>{score if score is not None else '—'}</span>"
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
            if state == "active":
                active_rows.append(
                    "<article class='run-card'>"
                    f"<div><span class='status-dot'></span><strong>{html.escape(label)}</strong> "
                    f"<span class='pill'>{html.escape(newest['task'])}</span>"
                    f"<p class='muted'>Training now · latest saved model: iteration {latest.get('latest_iteration')}</p></div>"
                    f"<button class='play' data-run-id='{html.escape(latest['run_id'])}' "
                    "data-label='Open saved model'>Open saved model</button></article>"
                )
            row = (
                "<article class='finished-card'>"
                "<div class='finished-card-top'><div>"
                f"<h3>{html.escape(label)}</h3>"
                f"<div class='run-tags'><span class='pill'>{html.escape(newest['task'])}</span>"
                f"<span class='kind-tag'>{'Smoke check' if newest.get('experiment_kind') == 'smoke' else 'Training run'}</span>"
                "<span class='complete-tag'>Finished</span></div></div>"
                "<div class='launch-cluster'>"
                f"<button class='play primary-action' data-run-id='{html.escape(latest['run_id'])}' data-label='Drive training arena'>Drive training arena</button>"
                f"<button class='deployment secondary' data-run-id='{html.escape(latest['run_id'])}' "
                f"{'disabled ' if not latest.get('has_exported_policy') or latest.get('task') not in {'roller', 'swizzle'} else ''}"
                "title='Run the exported ONNX through Pollen CPU MuJoCo'>Deployment check</button></div></div>"
                "<div class='run-stats'>"
                f"<div><small>Latest iteration</small><strong>{max(item.get('latest_iteration') or -1 for item in group):,}</strong></div>"
                f"<div><small>Saved models</small><strong>{len(distinct_versions)}</strong></div>"
                f"<div><small>Skill</small><strong>{html.escape(newest['task']).title()}</strong></div>"
                f"<div><small>Upstream</small><strong>{html.escape(str(latest.get('source', {}).get('upstream', {}).get('commit') or 'unknown')[:8])}</strong></div></div>"
                "<details class='saved-dropdown'><summary><span>Saved models</span>"
                f"<span class='summary-count'>{len(distinct_versions)}</span><span class='chevron'>⌄</span></summary>"
                "<div class='saved-list'>"
                + "".join(snapshots)
                + "</div></details></article>"
            )
            if state != "active":
                finished_rows.append(row)
        champions = "".join(
            f"<li><strong>{html.escape(task)}</strong>: "
            + ", ".join(f"{html.escape(stage)} = {html.escape(run_id)}" for stage, run_id in stages.items())
            + "</li>"
            for task, stages in registry.get("tasks", {}).items()
        ) or "<li>No promoted policies yet</li>"
        content = (
            "<header class='product-header'><div class='brand-lockup'>"
            "<div class='duck-mark' aria-hidden='true'>MD</div><div><p class='eyebrow'>DUCKLAB · POLICY BENCH</p>"
            "<h1>MicroDuck Control Room</h1><p class='tagline'>Train. Test. Promote.</p></div></div>"
            "<div class='header-side'><span class='local-badge'><i></i> Local &amp; open source</span>"
            f"<span class='stack-badge'>Pollen microduck_rl · {html.escape(upstream_branch)}@{html.escape(upstream_commit)}</span>"
            "<div class='header-status'><span id='system-status'>Checking system status…</span></div></div></header>"
            "<nav class='quick-nav' aria-label='Dashboard sections'><a href='#training'>Training</a><a href='#simulations'>Simulations</a><a href='#runs'>Finished runs</a><a href='#assistant'>Assistant</a></nav>"
            "<section id='training'><div class='section-title'><div><p class='eyebrow'>NOW</p><h2>Active training</h2></div></div><div class='panel' id='active-training'>"
            + ("".join(active_rows) or "<p id='active-empty'>No active training jobs.</p>")
            + "<div class='resource-control'><div><strong>Resource mode</strong><p id='resource-copy' class='muted'>Shared keeps local AI services online.</p></div><select id='resource-profile' aria-label='Training resource mode'><option value='shared'>Shared · vLLM stays online</option><option value='training-priority'>Training priority · pause vLLM</option></select></div>"
            + "<div class='progress' aria-label='Training progress'><span id='training-progress-bar'></span></div>"
            "<p id='training-progress' class='progress-copy'>Checking progress…</p>"
            "<div id='live-reward' class='live-curve' hidden><div class='curve-heading'><div><strong id='reward-title'>Recent mean reward</strong><span id='reward-range' class='muted'></span></div><button id='reward-scope' class='secondary curve-scope' type='button'>Entire run</button></div>"
            "<svg viewBox='0 0 720 170' role='img' aria-label='Recent training mean reward'><line x1='28' y1='145' x2='700' y2='145'></line><polyline id='reward-line' points=''></polyline></svg></div></div></section>"
            "<section id='simulations'><div class='section-title'><div><p class='eyebrow'>PLAYGROUND</p><h2>Open simulations</h2></div></div><div class='panel'>"
            "<div class='section-heading'><p class='muted'>Each model runs in its own arena. Open its controller only when you want to drive that model.</p>"
            "<button id='stop-all-viewers' class='secondary' type='button' disabled>Stop all</button></div>"
            "<div id='viewer-sessions' class='session-grid'><p>No simulations open.</p></div></div></section>"
            "<section id='runs'><div class='section-title'><div><p class='eyebrow'>LIBRARY</p><h2>Finished training runs</h2></div>"
            "<p class='section-note'>Open the latest model, or expand a run to inspect its saved history.</p></div>"
            "<div class='finished-grid'>"
            + "".join(finished_rows)
            + "</div></section>"
            f"<section><div class='section-title'><div><p class='eyebrow'>REGISTRY</p><h2>Promoted policies</h2></div></div><div class='panel'><ul>{champions}</ul></div></section>"
            + "<section id='assistant'><div class='section-title'><div><p class='eyebrow'>COPILOT</p><h2>DuckLab Assistant</h2></div></div><div class='panel assistant-panel'><div id='chat-log' class='chat-log'>"
            + "<p><strong>DuckLab:</strong> Tell me what you want MicroDuck to learn or ask what is running.</p>"
            + "</div><form id='chat-form'><input id='chat-input' autocomplete='off' placeholder='Example: train MicroDuck to skate backwards'>"
            + "<button type='submit'>Send</button></form><div id='chat-action'></div></div></section>"
            + "<script>"
            + "const TOKEN='__CONTROL_TOKEN__';"
            + "let rewardScope='recent',rewardSeries={recent:[],full:[],count:0};"
            + "async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Policy-Bench-Token':TOKEN},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');return j;}"
            + "function say(who,text){const p=document.createElement('p');const b=document.createElement('strong');b.textContent=who+': ';p.appendChild(b);p.appendChild(document.createTextNode(text));document.querySelector('#chat-log').appendChild(p);p.scrollIntoView();}"
            + "function shortRun(id){return id.length>54?id.slice(0,51)+'…':id;}"
            + "function sessionCard(v){const card=document.createElement('article');card.className='session-card';const info=document.createElement('div');const title=document.createElement('strong');title.textContent=v.label||shortRun(v.run_id);title.title=v.run_id;const ports=document.createElement('p');ports.className='muted';ports.textContent=(v.task||'policy')+(v.iteration!==null?' · saved iteration '+v.iteration:'')+' · arena '+v.viser_port+' · controller '+v.controller_port;info.append(title,ports);const actions=document.createElement('div');actions.className='session-actions';const drive=document.createElement('a');drive.href=v.controller_url;drive.target='_blank';drive.textContent='Drive arena';const arena=document.createElement('a');arena.href=v.viser_url;arena.target='_blank';arena.textContent='Arena only';const stop=document.createElement('button');stop.className='danger';stop.textContent='Stop';stop.onclick=async()=>{stop.disabled=true;try{await api('/api/stop-viewer',{run_id:v.run_id});await refreshStatus();}catch(e){alert(e.message);stop.disabled=false;}};actions.append(drive,arena,stop);card.append(info,actions);return card;}"
            + "function renderSessions(viewers){const box=document.querySelector('#viewer-sessions');box.replaceChildren();if(!viewers.length){const p=document.createElement('p');p.textContent='No simulations open.';box.appendChild(p);}else{viewers.forEach(v=>box.appendChild(sessionCard(v)));}const stopAll=document.querySelector('#stop-all-viewers');stopAll.disabled=!viewers.length;}"
            + "function drawReward(){const history=rewardScope==='full'?rewardSeries.full:rewardSeries.recent;const box=document.querySelector('#live-reward');if(!history||history.length<2){box.hidden=true;return;}box.hidden=false;document.querySelector('#reward-title').textContent=rewardScope==='full'?'Entire run · mean reward':'Recent mean reward';document.querySelector('#reward-scope').textContent=rewardScope==='full'?'Recent':'Entire run';const values=history.map(p=>p.reward),ordered=[...values].sort((a,b)=>a-b);const low=ordered[Math.floor((ordered.length-1)*.05)],high=ordered[Math.ceil((ordered.length-1)*.95)],span=high-low||1,clipped=values.filter(value=>value<low||value>high).length;const points=history.map((p,i)=>{const shown=Math.max(low,Math.min(high,p.reward));return(28+i*672/(history.length-1)).toFixed(1)+','+(145-(shown-low)*120/span).toFixed(1)}).join(' ');document.querySelector('#reward-line').setAttribute('points',points);const latest=history[history.length-1],sample=rewardScope==='full'&&rewardSeries.count>history.length?' · '+history.length+' plotted from '+rewardSeries.count+' points':'';document.querySelector('#reward-range').textContent=' · iteration '+history[0].iteration+' → '+latest.iteration+' · latest '+latest.reward.toFixed(2)+' · plotted range '+low.toFixed(2)+' to '+high.toFixed(2)+' · '+clipped+' outliers clipped'+sample;}function renderReward(progress){rewardSeries={recent:progress&&progress.reward_history||[],full:progress&&progress.reward_history_full||[],count:progress&&progress.reward_history_count||0};drawReward();}"
            + "async function openWhenReady(win,url){for(let i=0;i<120;i++){try{await fetch(url,{mode:'no-cors',cache:'no-store'});if(win&&!win.closed)win.location=url;return;}catch(e){await new Promise(r=>setTimeout(r,500));}}if(win&&!win.closed)win.location=url;}"
            + "async function playRun(button){const label=button.dataset.label||'Open simulation';const windowName='microduck-drive-'+button.dataset.runId.replace(/[^a-zA-Z0-9]/g,'-');const drive=window.open('about:blank',windowName);button.disabled=true;button.textContent='Starting simulation…';try{const result=await api('/api/play',{run_id:button.dataset.runId});await refreshStatus();await openWhenReady(drive,result.controller_url);}catch(error){if(drive&&!drive.closed)drive.close();alert(error.message);}finally{button.disabled=false;button.textContent=label;}}"
            + "async function deploymentCheck(button){const label=button.textContent;button.disabled=true;button.textContent='Checking ONNX…';try{const result=await api('/api/deployment-check',{run_id:button.dataset.runId});alert('Deployment check complete · score '+(result.score===null?'not scorable':result.score+'/100'));location.reload();}catch(error){alert(error.message);button.disabled=false;button.textContent=label;}}"
            + "document.querySelectorAll('.play').forEach(button=>button.addEventListener('click',()=>playRun(button)));"
            + "document.querySelectorAll('.deployment').forEach(button=>button.addEventListener('click',()=>deploymentCheck(button)));"
            + "document.querySelectorAll('.star').forEach(button=>button.addEventListener('click',async()=>{try{const starred=button.textContent.includes('Star');await api('/api/star',{run_id:button.dataset.runId,star:starred});location.reload();}catch(error){alert(error.message);}}));"
            + "document.querySelector('#stop-all-viewers').addEventListener('click',async()=>{try{await api('/api/stop-viewer',{});await refreshStatus();}catch(error){alert(error.message);}});"
            + "async function refreshStatus(){try{const r=await fetch('/api/status',{cache:'no-store'});const s=await r.json();const detected=s.training.detected.length;const p=s.training.progress;const pct=p&&p.total?Math.min(100,100*p.iteration/p.total):0;const t=detected?'Training running'+(p?' · iteration '+p.iteration+(p.total?' / '+p.total:'')+(p.eta?' · ETA '+p.eta:''):''):'No training running';const resource=s.resources||{profile:'shared',vllm_online:false};document.querySelector('#system-status').textContent=t+' · '+s.viewers.length+' simulation'+(s.viewers.length===1?'':'s')+' open · vLLM '+(resource.vllm_online?'online':'paused');document.querySelector('#resource-copy').textContent=resource.profile==='training-priority'?'Training priority active; vLLM is paused until training exits.':'Shared keeps vLLM and Hermes inference online during training.';document.querySelector('#training-progress').textContent=t+(p&&p.total?' · '+pct.toFixed(1)+'% complete':'');document.querySelector('#training-progress-bar').style.width=pct+'%';renderReward(p);renderSessions(s.viewers||[]);}catch(e){document.querySelector('#system-status').textContent='Status unavailable';}}"
            + "document.querySelector('#chat-form').addEventListener('submit',async event=>{event.preventDefault();const input=document.querySelector('#chat-input');const message=input.value.trim();if(!message)return;say('You',message);input.value='';document.querySelector('#chat-action').replaceChildren();try{const response=await api('/api/chat',{message});say('DuckLab',response.reply);if(response.kind==='confirm-training'){const profile=document.querySelector('#resource-profile');if(response.action.resource_profile)profile.value=response.action.resource_profile;const button=document.createElement('button');button.textContent='Confirm training launch';button.onclick=async()=>{button.disabled=true;try{const action={...response.action,resource_profile:profile.value};const result=await api('/api/train',action);say('DuckLab','Training started in '+result.resource_profile+' mode.');refreshStatus();}catch(error){say('DuckLab',error.message);button.disabled=false;}};document.querySelector('#chat-action').appendChild(button);}if(response.kind==='play'&&response.result){await refreshStatus();}}catch(error){say('DuckLab',error.message);}});"
            + "document.querySelector('#reward-scope').addEventListener('click',()=>{rewardScope=rewardScope==='recent'?'full':'recent';drawReward();});refreshStatus();setInterval(refreshStatus,5000);"
            + "</script>"
        )
        output = self.state_dir / "index.html"
        output.write_text(page("MicroDuck Policy Bench", content, show_title=False))
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
:root{{--bg:#0b0f14;--surface:#121922;--surface-2:#18222e;--line:#2a3948;--text:#f2f7fb;--muted:#99a9b8;--brand:#45d6a0;--brand-2:#5dbdff;--danger:#dc5d68}}
*{{box-sizing:border-box}}body{{font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1240px;margin:0 auto;padding:44px 24px 80px;background:radial-gradient(circle at 50% -20%,#162638 0,var(--bg) 42%);color:var(--text)}}
h1{{font-size:clamp(2rem,4vw,3.2rem);letter-spacing:-.045em;margin:0 0 8px}}h2{{font-size:1.22rem;letter-spacing:-.015em;margin:34px 0 10px}}p{{margin:7px 0}}a{{color:var(--brand-2);text-decoration:none}}a:hover{{text-decoration:underline}}
.lede{{font-size:1.05rem;color:var(--muted);margin-bottom:22px}}.summary-strip{{display:flex;align-items:center;min-height:48px;padding:12px 16px;border:1px solid #285345;border-radius:12px;background:#10251f;color:#c9f9e7}}
.panel{{background:color-mix(in srgb,var(--surface) 94%,transparent);border:1px solid var(--line);border-radius:14px;padding:18px;margin:10px 0;box-shadow:0 14px 35px rgba(0,0,0,.14)}}
.section-heading,.run-card,.session-card,.session-actions{{display:flex;align-items:center;gap:12px}}.section-heading,.run-card,.session-card{{justify-content:space-between}}.run-card{{padding:4px 0 14px}}.run-card+.run-card{{border-top:1px solid var(--line);padding-top:14px}}
.status-dot{{display:inline-block;width:9px;height:9px;margin-right:9px;border-radius:50%;background:var(--brand);box-shadow:0 0 0 5px rgba(69,214,160,.12)}}.pill,.badge{{display:inline-block;background:#203445;color:#cfeaff;border-radius:999px;padding:2px 9px;font-size:.78rem;margin-left:7px}}
button,.session-actions a{{appearance:none;border:0;border-radius:9px;padding:9px 13px;background:#197c5c;color:white;font:inherit;font-weight:650;cursor:pointer;white-space:nowrap}}button:hover,.session-actions a:hover{{filter:brightness(1.12);text-decoration:none}}button:disabled{{opacity:.48;cursor:not-allowed}}button.secondary{{background:#293847}}button.danger{{background:#552b32;color:#ffcdd2}}
.session-grid{{display:grid;gap:10px}}.session-card{{background:var(--surface-2);border:1px solid var(--line);border-radius:11px;padding:14px 15px}}.session-actions a{{background:#24445d}}.session-actions a:first-child{{background:#197c5c}}
.progress{{height:12px;background:#26323d;border-radius:99px;overflow:hidden;margin:15px 0 8px}}.progress span{{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--brand),#79e8ff);transition:width .5s}}.progress-copy{{color:#d5e2eb}}
.live-curve{{margin-top:15px;padding:12px 13px;background:#0d141b;border:1px solid var(--line);border-radius:10px}}.live-curve[hidden]{{display:none}}.curve-heading{{display:flex;justify-content:space-between;gap:12px;align-items:center}}.curve-heading>div{{min-width:0}}.curve-scope{{padding:6px 10px;font-size:.8rem}}.live-curve svg{{display:block;width:100%;height:auto;max-height:180px}}.live-curve line{{stroke:#30404e}}.live-curve polyline{{fill:none;stroke:var(--brand-2);stroke-width:2.5;vector-effect:non-scaling-stroke}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:var(--surface)}}table{{width:100%;border-collapse:collapse;min-width:920px}}th,td{{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#b7c7d5;font-size:.78rem;text-transform:uppercase;letter-spacing:.055em;background:#151e28}}tr:last-child td{{border-bottom:0}}tbody tr:hover,table tr:hover td{{background:rgba(93,189,255,.025)}}
details{{margin-top:9px;padding:9px 11px;background:#0e151c;border:1px solid var(--line);border-radius:9px;min-width:360px}}details summary{{cursor:pointer;font-weight:650;color:#d7e4ee}}details table{{margin-top:10px;min-width:670px;font-size:.9em}}td button{{margin:2px 4px 2px 0}}
form{{display:flex;gap:9px;margin-top:12px}}input,select{{min-width:0;background:#0b1117;color:var(--text);border:1px solid #435567;border-radius:9px;padding:11px 12px;font:inherit}}input{{flex:1}}input:focus,select:focus{{outline:2px solid rgba(93,189,255,.4);border-color:var(--brand-2)}}.chat-log{{max-height:280px;overflow:auto}}#chat-action{{margin-top:10px}}#chat-action a{{margin-left:10px}}
pre,.mono{{font-family:ui-monospace,SFMono-Regular,monospace;overflow:auto;background:#10171f;padding:13px;border-radius:9px}}.chart{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;margin:12px 0}}.chart svg{{display:block;width:100%;height:auto}}.muted{{color:var(--muted);font-size:.9em}}
html{{scroll-behavior:smooth}}body{{max-width:1320px;padding:30px 28px 90px;background:radial-gradient(circle at 78% -12%,rgba(32,102,123,.28),transparent 34%),radial-gradient(circle at 4% 8%,rgba(32,105,78,.2),transparent 28%),var(--bg)}}section{{scroll-margin-top:86px;margin-top:36px}}
.product-header{{position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:flex-end;gap:28px;min-height:230px;padding:34px 36px;border:1px solid #304656;border-radius:24px;background:linear-gradient(135deg,rgba(24,38,51,.98),rgba(13,23,31,.98));box-shadow:0 24px 70px rgba(0,0,0,.3)}}.product-header:after{{content:'';position:absolute;width:360px;height:360px;right:-120px;top:-190px;border-radius:50%;background:radial-gradient(circle,rgba(93,189,255,.24),transparent 65%);pointer-events:none}}.brand-lockup{{display:flex;align-items:center;gap:22px;z-index:1}}.duck-mark{{display:grid;place-items:center;flex:0 0 72px;height:72px;border-radius:20px;background:linear-gradient(145deg,var(--brand),#2a94aa);box-shadow:0 13px 38px rgba(69,214,160,.22);color:#07120e;font-weight:900;font-size:1.25rem;letter-spacing:-.08em}}.product-header h1{{font-size:clamp(2.4rem,5vw,4rem);line-height:1;letter-spacing:-.055em;margin:7px 0 12px}}.eyebrow{{margin:0;color:#72dbbb;font-size:.72rem;font-weight:800;letter-spacing:.14em}}.tagline{{margin:0;color:#b9c8d4;font-size:1.08rem}}.header-side{{display:flex;flex-direction:column;align-items:flex-end;gap:10px;z-index:1;max-width:460px}}.local-badge{{display:flex;align-items:center;gap:8px;color:#cce9df;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}}.local-badge i{{width:8px;height:8px;border-radius:50%;background:var(--brand);box-shadow:0 0 0 5px rgba(69,214,160,.12)}}.stack-badge{{color:#aac4d6;font:600 .78rem ui-monospace,SFMono-Regular,monospace}}.header-status{{padding:12px 15px;border:1px solid #335366;border-radius:11px;background:rgba(6,15,21,.6);color:#d9e9f2;text-align:right}}
.quick-nav{{position:sticky;top:12px;z-index:10;display:flex;gap:6px;width:max-content;max-width:100%;margin:14px auto 0;padding:6px;border:1px solid rgba(58,79,96,.9);border-radius:12px;background:rgba(13,19,26,.9);box-shadow:0 12px 35px rgba(0,0,0,.26);backdrop-filter:blur(14px);overflow:auto}}.quick-nav a{{padding:7px 12px;border-radius:8px;color:#b9c8d4;font-size:.84rem;font-weight:650;white-space:nowrap}}.quick-nav a:hover{{background:#1c2934;color:#fff;text-decoration:none}}
.section-title{{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:11px}}.section-title h2{{margin:0;font-size:1.35rem}}.section-title .eyebrow{{margin-bottom:3px}}.section-note{{color:var(--muted);font-size:.9rem;text-align:right;max-width:480px}}.panel{{margin:0;border-radius:16px;padding:20px;background:rgba(18,25,34,.94)}}
button,.session-actions a{{border:1px solid transparent;font-weight:700;transition:transform .15s ease,filter .15s ease,border-color .15s ease}}button:hover,.session-actions a:hover{{transform:translateY(-1px)}}button:disabled{{transform:none}}button.secondary{{background:#263644;border-color:#3a4c5c}}.primary-action{{padding:10px 16px}}
.finished-grid{{display:grid;gap:14px}}.finished-card{{overflow:hidden;border:1px solid var(--line);border-radius:16px;background:linear-gradient(145deg,rgba(20,29,39,.98),rgba(15,22,29,.98));box-shadow:0 12px 30px rgba(0,0,0,.12)}}.finished-card-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding:20px 21px 13px}}.finished-card h3{{font-size:1.08rem;margin:0;letter-spacing:-.015em}}.run-tags{{display:flex;align-items:center;flex-wrap:wrap;gap:7px;margin-top:9px}}.run-tags .pill{{margin:0}}.kind-tag,.complete-tag,.stage-badge{{padding:3px 8px;border-radius:999px;background:#252f39;color:#b6c4d0;font-size:.74rem;font-weight:700}}.complete-tag{{background:#143c31;color:#9ee8ce}}.launch-cluster{{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}}.run-stats{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid rgba(42,57,72,.65)}}.run-stats>div{{display:flex;flex-direction:column;gap:2px;padding:13px 21px;border-right:1px solid rgba(42,57,72,.65)}}.run-stats>div:last-child{{border:0}}small{{display:block;color:var(--muted);font-size:.69rem;text-transform:uppercase;letter-spacing:.07em;font-weight:750}}
.saved-dropdown{{margin:0;min-width:0;padding:0;border:0;border-top:1px solid var(--line);border-radius:0;background:#0d141b}}.saved-dropdown>summary{{display:flex;align-items:center;gap:9px;padding:13px 21px;cursor:pointer;color:#cfdae3;font-weight:700;list-style:none;user-select:none}}.saved-dropdown>summary::-webkit-details-marker{{display:none}}.saved-dropdown>summary:hover{{background:#111b24}}.summary-count{{display:grid;place-items:center;min-width:22px;height:22px;padding:0 6px;border-radius:999px;background:#243441;color:#b9d3e5;font-size:.72rem}}.chevron{{margin-left:auto;font-size:1.2rem;transition:transform .18s ease}}.saved-dropdown[open] .chevron{{transform:rotate(180deg)}}.saved-list{{display:grid;gap:8px;padding:0 12px 12px}}.saved-model{{display:grid;grid-template-columns:minmax(150px,1fr) minmax(250px,1.6fr) auto;align-items:center;gap:18px;padding:13px 14px;border:1px solid #263744;border-radius:10px;background:#131d25}}.saved-model-title{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}.stage-badge{{background:#203545;color:#bfe2fa}}.saved-model-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.saved-model-stats span{{font-weight:750}}.saved-model-actions{{display:flex;align-items:center;justify-content:flex-end;gap:10px}}.text-action{{font-weight:700;white-space:nowrap}}
.assistant-panel{{background:linear-gradient(145deg,#121b24,#101820)}}
.resource-control{{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:14px 0;padding:13px 14px;border:1px solid #2b4050;border-radius:11px;background:#101a22}}.resource-control select{{min-width:300px}}
@media(max-width:900px){{.product-header{{align-items:flex-start;flex-direction:column;min-height:0}}.header-side{{align-items:flex-start;max-width:none}}.header-status{{text-align:left}}.saved-model{{grid-template-columns:1fr}}.saved-model-actions{{justify-content:flex-start}}}}
@media(max-width:720px){{body{{padding:14px 13px 60px}}.product-header{{padding:25px 20px;border-radius:18px}}.brand-lockup{{align-items:flex-start;gap:14px}}.duck-mark{{flex-basis:52px;height:52px;border-radius:14px;font-size:.95rem}}.quick-nav{{justify-content:flex-start;margin-top:10px}}.section-title,.section-heading,.run-card,.session-card,.finished-card-top,.resource-control{{align-items:flex-start;flex-direction:column}}.section-note{{text-align:left}}.session-actions,.launch-cluster{{width:100%;flex-wrap:wrap}}.session-actions a,.session-actions button,.run-card>button,.launch-cluster button{{width:100%;text-align:center}}.resource-control select{{width:100%;min-width:0}}.run-stats{{grid-template-columns:1fr}}.run-stats>div{{border-right:0;border-bottom:1px solid var(--line)}}.saved-model-stats{{grid-template-columns:repeat(3,1fr)}}.saved-model-actions{{flex-wrap:wrap}}form{{flex-direction:column}}}}
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
            if not args.task or manifest["task"] == args.task
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
