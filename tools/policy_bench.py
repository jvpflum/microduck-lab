#!/usr/bin/env python3
"""Offline-first experiment registry and policy promotion workflow."""

from __future__ import annotations

import argparse
import hashlib
import html
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
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


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
            return read_json(existing)
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
            "stage": "experimental",
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
        subprocess.run(
            [sys.executable, str(evaluator), str(policy_path), "--output", str(output)],
            cwd=UPSTREAM,
            check=True,
        )
        return self.attach_evaluation(run_id, output, suite)

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
        body = page(
            manifest["run_id"],
            f"<p><span class='badge'>{html.escape(manifest['stage'])}</span> Task: {html.escape(manifest['task'])}</p>"
            f"<p>Iteration: {manifest.get('latest_iteration')} · ONNX export: {manifest.get('has_exported_policy')}</p>"
            f"<p class='mono'>{html.escape(manifest['source_run_dir'])}</p>"
            f"<h2>Evaluations</h2><table><tr><th>Suite</th><th>Created</th><th>Data</th></tr>{rows}</table>"
            f"<h2>Comparisons</h2><ul>{comparison_rows}</ul>"
            f"<h2>Manifest</h2><pre>{html.escape(json.dumps(manifest, indent=2, sort_keys=True))}</pre>",
        )
        output = self.runs_dir / run_id / "report.html"
        output.write_text(body)
        return output

    def render_dashboard(self) -> Path:
        registry = read_json(self.registry_path)
        rows = []
        for manifest in sorted(self.manifests(), key=lambda item: item["created_at"], reverse=True):
            run_dir = self.runs_dir / manifest["run_id"]
            report = run_dir / "report.html"
            if not report.exists():
                self.render_run_report(manifest["run_id"])
            rows.append(
                "<tr>"
                f"<td><a href='runs/{html.escape(manifest['run_id'])}/report.html'>{html.escape(manifest['run_id'])}</a></td>"
                f"<td>{html.escape(manifest['task'])}</td><td><span class='badge'>{html.escape(manifest['stage'])}</span></td>"
                f"<td>{manifest.get('latest_iteration')}</td><td>{len(manifest.get('evaluations', []))}</td>"
                f"<td>{'yes' if manifest.get('has_exported_policy') else 'no'}</td></tr>"
            )
        champions = "".join(
            f"<li><strong>{html.escape(task)}</strong>: "
            + ", ".join(f"{html.escape(stage)} = {html.escape(run_id)}" for stage, run_id in stages.items())
            + "</li>"
            for task, stages in registry.get("tasks", {}).items()
        ) or "<li>No promoted policies yet</li>"
        content = (
            "<p>Offline policy lineage, evaluation, comparison, and promotion.</p>"
            f"<h2>Registry</h2><ul>{champions}</ul>"
            "<h2>Runs</h2><table><tr><th>Run</th><th>Task</th><th>Stage</th><th>Iteration</th><th>Evals</th><th>ONNX</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        output = self.state_dir / "index.html"
        output.write_text(page("MicroDuck Policy Bench", content))
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


def page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;background:#101418;color:#edf2f7}}
a{{color:#65c7ff}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #34404b;text-align:left}}
.badge{{background:#1f6f50;border-radius:999px;padding:3px 9px}}pre,.mono{{font-family:ui-monospace,monospace;overflow:auto;background:#171d23;padding:12px}}</style></head>
<body><h1>{html.escape(title)}</h1>{body}</body></html>"""


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
