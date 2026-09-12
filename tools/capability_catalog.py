#!/usr/bin/env python3
"""Load the declarative robot, program, task, evaluator, and viewer catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = LAB_ROOT / "config" / "robotics-capabilities.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported robotics capability catalog schema")
    for section in ("robots", "programs", "tasks"):
        if not isinstance(catalog.get(section), dict) or not catalog[section]:
            raise ValueError(f"Capability catalog requires a non-empty {section} object")
    for token, task in catalog["tasks"].items():
        if task.get("robot_id") not in catalog["robots"]:
            raise ValueError(f"Task {token!r} references an unknown robot")
        if task.get("program_id") not in catalog["programs"]:
            raise ValueError(f"Task {token!r} references an unknown program")
        for field in ("train_script", "play_task", "display_name"):
            if not task.get(field):
                raise ValueError(f"Task {token!r} is missing {field}")
        for relative in (task["train_script"], (task.get("evaluator") or {}).get("script")):
            if not relative:
                continue
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Task {token!r} contains an unsafe path")
    return catalog


def task_spec(task: str) -> dict[str, Any] | None:
    return load_catalog()["tasks"].get(task)


def task_has_evaluator(task: str | None) -> bool:
    spec = task_spec(str(task))
    return bool(spec and spec.get("evaluator"))


def training_tasks() -> dict[str, dict[str, Any]]:
    result = {}
    for token, spec in load_catalog()["tasks"].items():
        result[token] = {
            **spec,
            "train_script": LAB_ROOT / spec["train_script"],
        }
    return result


def program_specs() -> dict[str, dict[str, Any]]:
    return load_catalog()["programs"]
