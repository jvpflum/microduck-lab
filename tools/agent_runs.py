#!/usr/bin/env python3
"""Publish generic coding-agent run receipts for the DuckLab dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT_DIR = LAB_ROOT / "policy-bench" / "agent-runs"
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
ACTIVE_STATES = {"planning", "running", "evaluating"}
ALL_STATES = ACTIVE_STATES | {"complete", "failed", "stopped"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != 1:
        raise ValueError("Agent run receipt schema_version must be 1")
    run_id = str(value.get("run_id", ""))
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only lowercase letters, numbers, dot, dash, or underscore")
    for field in ("project", "title", "goal", "status"):
        if not str(value.get(field, "")).strip():
            raise ValueError(f"Agent run receipt requires {field}")
    if value["status"] not in ALL_STATES:
        raise ValueError(f"status must be one of {', '.join(sorted(ALL_STATES))}")
    metrics = value.get("metrics", [])
    if not isinstance(metrics, list) or len(metrics) > 24:
        raise ValueError("metrics must be a list of at most 24 entries")
    for metric in metrics:
        if not isinstance(metric, dict) or not str(metric.get("name", "")).strip():
            raise ValueError("Each metric requires a name")
        metric_value = metric.get("value")
        if not isinstance(metric_value, (int, float, str)) or isinstance(metric_value, bool):
            raise ValueError("Metric values must be numbers or short strings")
    actions = value.get("actions", [])
    if not isinstance(actions, list) or len(actions) > 8:
        raise ValueError("actions must be a list of at most eight links")
    for action in actions:
        url = str(action.get("url", "")) if isinstance(action, dict) else ""
        if not url.startswith(("http://", "https://", "/")):
            raise ValueError("Action URLs must be HTTP(S) or dashboard-relative")
    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) > 16:
        raise ValueError("artifacts must be a list of at most 16 entries")
    normalized_artifacts = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not artifact.get("path"):
            raise ValueError("Each artifact requires a path")
        path = Path(str(artifact["path"])).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Artifact does not exist: {path}")
        expected = artifact.get("sha256")
        actual = sha256(path)
        if expected and expected != actual:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        normalized_artifacts.append({**artifact, "path": str(path), "sha256": actual})
    progress = value.get("progress")
    if progress is not None:
        if not isinstance(progress, dict):
            raise ValueError("progress must be an object")
        for field in ("current", "total"):
            if field in progress and not isinstance(progress[field], (int, float)):
                raise ValueError(f"progress.{field} must be numeric")
    return {
        **value,
        "run_id": run_id,
        "project": str(value["project"]).strip(),
        "title": str(value["title"]).strip(),
        "goal": str(value["goal"]).strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": normalized_artifacts,
    }


def publish_receipt(receipt_path: Path, output_dir: Path = DEFAULT_RECEIPT_DIR) -> Path:
    value = json.loads(receipt_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Agent run receipt must be a JSON object")
    value = validate_receipt(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{value['run_id']}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output)
    return output


def load_agent_runs(output_dir: Path = DEFAULT_RECEIPT_DIR) -> list[dict[str, Any]]:
    if not output_dir.is_dir():
        return []
    receipts = []
    for path in output_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                receipts.append(validate_receipt_for_read(value))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(receipts, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def validate_receipt_for_read(value: dict[str, Any]) -> dict[str, Any]:
    """Cheap, non-mutating validation for dashboard reads."""
    run_id = str(value.get("run_id", ""))
    if value.get("schema_version") != 1 or not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid receipt")
    if value.get("status") not in ALL_STATES:
        raise ValueError("Invalid receipt status")
    for field in ("project", "title", "goal"):
        if not str(value.get(field, "")).strip():
            raise ValueError("Incomplete receipt")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    args = parser.parse_args()
    output = publish_receipt(args.receipt, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
