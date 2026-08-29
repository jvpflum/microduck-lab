#!/usr/bin/env python3
"""Local control server for the MicroDuck Policy Bench dashboard."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import mimetypes
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import shutil
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from policy_bench import Bench, DEFAULT_STATE, FACTORY_ARENA_URL, LAB_ROOT, UPSTREAM, sha256


FACTORY_ARENA_DIST = LAB_ROOT / "upstream" / "microduck-simulator" / "app" / "dist"


TASKS = {
    "swizzle": {
        "play_task": "Mjlab-Velocity-Swizzle-MicroDuck",
        "train_script": LAB_ROOT / "scripts" / "train-swizzle.sh",
        "default_iterations": 8000,
    },
    "roller": {
        "play_task": "Mjlab-Velocity-Flat-MicroDuck-Rollers",
        "train_script": LAB_ROOT / "scripts" / "train-skate.sh",
        "default_iterations": 5000,
    },
    "walking": {
        "play_task": "Mjlab-Velocity-Flat-MicroDuck",
        "train_script": LAB_ROOT / "scripts" / "train-baseline.sh",
        "default_iterations": 4000,
    },
    "hop": {
        "play_task": "Mjlab-RollerHop-Flat-MicroDuck",
        "train_script": LAB_ROOT / "scripts" / "train-hop.sh",
        "default_iterations": 1500,
    },
    "backflip": {
        "play_task": "Mjlab-RollerBackflip-Flat-MicroDuck",
        "train_script": LAB_ROOT / "scripts" / "train-backflip.sh",
        "default_iterations": 2500,
    },
}

# The dashboard itself owns 8091. Each simulation gets an isolated Viser and
# controller pair so selecting one policy can never redirect an existing tab
# to another policy. Forward this bounded pool over SSH for remote use.
VIEWER_PORT_PAIRS = (
    (8080, 8090),
    (8081, 8092),
    (8082, 8093),
    (8083, 8094),
    (8084, 8095),
    (8085, 8096),
)

RESOURCE_PROFILES = {
    "shared": "Shared · keep vLLM and Hermes online",
    "training-priority": "Training priority · pause vLLM until training exits",
}
RESOURCE_MARKER = LAB_ROOT / "policy-bench" / "training-priority.json"
DEMONSTRATIONS_DIR = LAB_ROOT / "reports" / "demonstrations"


@dataclass
class ViewerSession:
    run_id: str
    label: str
    task: str
    iteration: int | None
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path
    viser_port: int
    controller_port: int
    started_at: float
    num_envs: int
    kind: str


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8")


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def running_training_processes() -> list[dict[str, Any]]:
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        # uv keeps a parent process around; report the actual trainer once.
        if "/uv/bin/uv run train" in command:
            continue
        if re.search(r"(?:^|/)train\s+Mjlab-", command):
            matches.append({"pid": int(entry.name), "command": command.strip()})
    return sorted(matches, key=lambda item: item["pid"])


def resource_status() -> dict[str, Any]:
    """Return cheap, local resource-mode status for the polled dashboard."""
    priority = RESOURCE_MARKER.is_file()
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.08):
            vllm_online = True
    except OSError:
        vllm_online = False
    return {
        "profile": "training-priority" if priority else "shared",
        "label": RESOURCE_PROFILES["training-priority" if priority else "shared"],
        "vllm_online": vllm_online,
    }


def cleanup_orphaned_dashboard_viewers(
    state_dir: Path = DEFAULT_STATE, proc_root: Path = Path("/proc")
) -> int:
    """Stop viewer groups left by an unclean dashboard termination.

    Only processes loading immutable Policy Bench snapshots are in scope; a
    viewer launched manually from an upstream log directory is left alone.
    """
    groups: set[int] = set()
    proc = proc_root
    if not proc.is_dir():
        return 0
    snapshot_root = str((state_dir / "runs").resolve())
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            pid = int(entry.name)
            process_group = os.getpgid(pid)
        except (OSError, PermissionError, ProcessLookupError):
            continue
        if "tools/play_viser_compat.py" in command and snapshot_root in command and process_group == pid:
            groups.add(process_group)
    for process_group in groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return len(groups)


def training_progress() -> dict[str, Any] | None:
    """Read the newest local training log without depending on W&B."""
    logs = sorted((LAB_ROOT / "reports").glob("train-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in logs:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        iterations = re.findall(r"(?:Iteration:|Learning iteration)\s*(\d+)", text)
        totals = re.findall(r"Learning iteration\s*\d+/(\d+)", text)
        eta = re.findall(r"ETA:\s*([^\n\r]+)", text)
        if iterations:
            full_history = [
                {"iteration": int(iteration), "reward": float(reward)}
                for iteration, reward in re.findall(
                    r"Learning iteration\s*(\d+)/\d+[\s\S]*?Mean reward:\s*([-+\d.eE]+)",
                    text,
                )
            ]
            # Preserve the complete training span without sending thousands of
            # SVG points over SSH every five seconds. Always retain both ends.
            if len(full_history) > 1_000:
                last = len(full_history) - 1
                indices = sorted({round(index * last / 999) for index in range(1_000)})
                full_history_for_chart = [full_history[index] for index in indices]
            else:
                full_history_for_chart = full_history
            latest_start = text.rfind("Learning iteration")
            latest_block = text[latest_start:] if latest_start >= 0 else text

            def latest_number(label: str) -> float | None:
                matches = re.findall(rf"{re.escape(label)}:\s*([-+\d.eE]+)", latest_block)
                return float(matches[-1]) if matches else None

            recent_values = [item["reward"] for item in full_history[-20:]]
            previous_values = [item["reward"] for item in full_history[-40:-20]]
            recent_mean = sum(recent_values) / len(recent_values) if recent_values else None
            previous_mean = sum(previous_values) / len(previous_values) if previous_values else None
            trend_delta = (
                recent_mean - previous_mean
                if recent_mean is not None and previous_mean is not None
                else None
            )
            volatility = None
            if recent_values:
                volatility = math.sqrt(
                    sum((value - recent_mean) ** 2 for value in recent_values) / len(recent_values)
                )
            trend_threshold = max(0.1, abs(previous_mean or 0.0) * 0.03)
            if len(full_history) < 20:
                trend = "warming up"
            elif trend_delta is not None and trend_delta > trend_threshold:
                trend = "improving"
            elif trend_delta is not None and trend_delta < -trend_threshold:
                trend = "regressing"
            else:
                trend = "steady"

            episode_rewards = {
                name: float(value)
                for name, value in re.findall(
                    r"Episode_Reward/([\w_]+):\s*([-+\d.eE]+)", latest_block
                )
            }
            nan_terminations = latest_number("Episode_Termination/nan_state") or 0.0
            verdict_tone = "good" if trend == "improving" else "neutral"
            verdict = {
                "improving": "The policy is learning; reward has risen over the last 20 iterations.",
                "regressing": "Recent reward is falling. Let it run briefly, then inspect a checkpoint if this continues.",
                "steady": "Reward is stable. This may be consolidation or the start of a plateau.",
                "warming up": "The run is still warming up; there is not enough history for a reliable trend.",
            }[trend]
            if nan_terminations > 0:
                verdict_tone = "bad"
                verdict = "Numerical failures are present. Do not promote this run until they are resolved."
            elif "hop_takeoff_velocity" in episode_rewards:
                takeoff = episode_rewards.get("hop_takeoff_velocity", 0.0)
                landing = episode_rewards.get("hop_landing", 0.0)
                stillness = episode_rewards.get("hop_landing_stillness", 0.0)
                if takeoff < 0.005:
                    verdict_tone = "watch"
                    verdict = (
                        "Reward is rising, but the takeoff signal is near zero. The policy may be "
                        "collecting shaping reward without a strong jump; inspect the latest checkpoint."
                    )
                elif landing < 0.1 or stillness < 0.01:
                    verdict_tone = "watch"
                    verdict = "Takeoff is emerging, but controlled landing is not learned yet. Keep training."

            checkpoints = sorted(
                (LAB_ROOT / "upstream" / "microduck_rl" / "logs" / "rsl_rl").glob(
                    "*/*/model_*.pt"
                ),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
            checkpoint_iteration = None
            if checkpoints:
                match = re.fullmatch(r"model_(\d+)\.pt", checkpoints[0].name)
                checkpoint_iteration = int(match.group(1)) if match else None
            return {
                "log": str(path),
                "iteration": int(iterations[-1]),
                "total": int(totals[-1]) if totals else None,
                "eta": eta[-1].strip() if eta else None,
                "reward_history": full_history[-80:],
                "reward_history_full": full_history_for_chart,
                "reward_history_count": len(full_history),
                "intelligence": {
                    "current_reward": full_history[-1]["reward"] if full_history else None,
                    "best_reward": max((item["reward"] for item in full_history), default=None),
                    "recent_mean": recent_mean,
                    "trend_delta": trend_delta,
                    "trend": trend,
                    "volatility": volatility,
                    "verdict": verdict,
                    "verdict_tone": verdict_tone,
                    "steps_per_second": latest_number("Steps per second"),
                    "total_steps": latest_number("Total steps"),
                    "mean_episode_length": latest_number("Mean episode length"),
                    "mean_action_std": latest_number("Mean action std"),
                    "value_loss": latest_number("Mean value loss"),
                    "surrogate_loss": latest_number("Mean surrogate loss"),
                    "nan_terminations": nan_terminations,
                    "latest_checkpoint_iteration": checkpoint_iteration,
                    "episode_rewards": episode_rewards,
                },
            }
    return None


def parse_training_request(message: str) -> dict[str, Any] | None:
    lowered = message.lower()
    if not re.search(r"\b(train|training|learn)\b", lowered):
        return None
    if "backflip" in lowered or "back flip" in lowered:
        task = "backflip"
    elif "hop" in lowered or "jump" in lowered:
        task = "hop"
    elif "roller" in lowered:
        task = "roller"
    elif "swizzle" in lowered or "skate" in lowered:
        task = "swizzle"
    elif "walk" in lowered:
        task = "walking"
    else:
        return {"error": "Tell me which skill to train: backflip, hop, swizzle, roller, or walking."}
    iteration_match = re.search(r"([\d,]+)\s*(?:iterations?|iters?)\b", lowered)
    environment_match = re.search(r"([\d,]+)\s*(?:environments?|envs?)\b", lowered)
    iterations = (
        int(iteration_match.group(1).replace(",", ""))
        if iteration_match
        else TASKS[task]["default_iterations"]
    )
    environments = (
        int(environment_match.group(1).replace(",", ""))
        if environment_match
        else 4096
    )
    if not 5 <= iterations <= 100_000:
        return {"error": "Iterations must be between 5 and 100,000."}
    if not 1 <= environments <= 8192:
        return {"error": "Parallel environments must be between 1 and 8,192."}
    priority_words = ("overnight", "maximum training", "max training", "training priority")
    resource_profile = (
        "training-priority" if any(word in lowered for word in priority_words) else "shared"
    )
    return {
        "task": task,
        "iterations": iterations,
        "environments": environments,
        "resource_profile": resource_profile,
    }


def codex_training_plan(message: str) -> dict[str, Any] | None:
    """Ask the local Codex CLI to classify a request, then validate its JSON.

    Codex is an interpreter here, never an executor: the returned task must be
    one of TASKS and numeric values are checked again before the UI can launch.
    Set DUCKLAB_CODEX=0 to force the deterministic parser (useful offline).
    """
    if os.environ.get("DUCKLAB_CODEX", "1").lower() in {"0", "false", "no"}:
        return None
    codex = os.environ.get("DUCKLAB_CODEX_BIN") or shutil.which("codex")
    if not codex:
        return None
    prompt = (
        "You are the MicroDuck training assistant. Return JSON only, with keys "
        "task, iterations, environments, resource_profile, supported, reply. task must be one of "
        "backflip, hop, swizzle, roller, walking, or custom. Choose custom for a skill that has "
        "no registered simulator task. Never invent a runnable task. Defaults are "
        "8000 iterations and 4096 environments. resource_profile must be shared "
        "or training-priority; use training-priority for overnight or maximum-throughput "
        "requests, otherwise shared. Keep reply under 240 characters.\n"
        f"User request: {message}"
    )
    try:
        result = subprocess.run(
            [codex, "exec", "--skip-git-repo-check", "--sandbox", "read-only", prompt],
            cwd=str(LAB_ROOT), capture_output=True, text=True, timeout=20, check=False,
        )
        raw = result.stdout.strip().splitlines()
        candidates = [line.strip().removeprefix("```json").removesuffix("```").strip() for line in raw]
        data = next((json.loads(line) for line in reversed(candidates) if line.startswith("{")), None)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("task") not in TASKS:
        return {"unsupported": True, "reply": data.get("reply") if isinstance(data, dict) else None}
    try:
        iterations = int(data.get("iterations", TASKS[data["task"]]["default_iterations"]))
        environments = int(data.get("environments", 4096))
    except (TypeError, ValueError):
        return None
    if not 5 <= iterations <= 100_000 or not 1 <= environments <= 8192:
        return None
    resource_profile = str(data.get("resource_profile", "shared"))
    if resource_profile not in RESOURCE_PROFILES:
        resource_profile = "shared"
    return {"task": data["task"], "iterations": iterations, "environments": environments,
            "resource_profile": resource_profile,
            "reply": str(data.get("reply") or "I mapped that to a validated training task.")}


class ProcessManager:
    def __init__(self, bench: Bench):
        self.bench = bench
        self.lock = threading.Lock()
        self.viewers: dict[str, ViewerSession] = {}
        self.training: subprocess.Popen[bytes] | None = None
        self.training_config: dict[str, Any] | None = None
        self.training_log = None
        self.deployment_checks: set[str] = set()
        self._last_training_discovery = 0.0
        self._training_candidates: list[dict[str, Any]] = []

    @staticmethod
    def _alive(process: subprocess.Popen[bytes] | None) -> bool:
        return process is not None and process.poll() is None

    def _reap_finished_locked(self) -> None:
        for run_id, session in list(self.viewers.items()):
            if session.process.poll() is not None:
                session.log_handle.close()
                del self.viewers[run_id]
        if self.training is not None and self.training.poll() is not None:
            self.training = None
            self.training_config = None
            if self.training_log:
                self.training_log.close()
                self.training_log = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._reap_finished_locked()
            training_alive = self._alive(self.training)
            detected_training = running_training_processes()
            now = time.monotonic()
            # Snapshot newly saved checkpoints while a real dashboard-owned
            # training job is active. A page refresh then exposes the latest
            # immutable checkpoint debugger without waiting for finalization.
            if (
                detected_training
                and self.bench.state_dir == DEFAULT_STATE.resolve()
                and now - self._last_training_discovery >= 30.0
            ):
                registered = self.bench.discover(UPSTREAM / "logs" / "rsl_rl")
                active_names = {
                    match.group(1)
                    for item in detected_training
                    if (match := re.search(r"--agent\.run-name\s+([^\s]+)", item.get("command", "")))
                }
                self._training_candidates = [
                    {
                        "run_id": item["run_id"],
                        "task": item["task"],
                        "iteration": item.get("latest_iteration"),
                        "label": item.get("experiment_label"),
                    }
                    for item in registered
                    if item.get("experiment_label") in active_names
                ]
                self._last_training_discovery = now
            viewers = [
                self._viewer_result(session, reused=True)
                for session in sorted(self.viewers.values(), key=lambda item: item.started_at)
            ]
            return {
                "viewers": viewers,
                # Kept for compatibility with older clients while the UI and
                # chat move to the explicit session list.
                "viewer": {
                    "running": bool(viewers),
                    "run_id": viewers[0]["run_id"] if viewers else None,
                    "pid": viewers[0]["pid"] if viewers else None,
                },
                "training": {
                    "managed_running": training_alive,
                    "config": self.training_config if training_alive else None,
                    "pid": self.training.pid if training_alive and self.training else None,
                    "detected": detected_training,
                    "progress": training_progress() if training_alive or detected_training else None,
                    "candidates": self._training_candidates if detected_training else [],
                },
                "resources": resource_status(),
            }

    def run_deployment_check(self, run_id: str) -> dict[str, Any]:
        manifest = self.bench.load_manifest(run_id)
        if manifest.get("task") not in {"roller", "swizzle", "hop"}:
            raise ValueError("Deployment Check is not configured for this policy type")
        if not manifest.get("artifacts", {}).get("policy"):
            raise ValueError("This saved model has no exported ONNX policy yet")
        with self.lock:
            if run_id in self.deployment_checks:
                raise ValueError("A Deployment Check is already running for this model")
            self.deployment_checks.add(run_id)
        try:
            suite = "hop-v1" if manifest.get("task") == "hop" else "skating-v1"
            record = self.bench.evaluate(run_id, suite)
        finally:
            with self.lock:
                self.deployment_checks.discard(run_id)
        evaluation = json.loads(Path(record["path"]).read_text())
        return {
            "run_id": run_id,
            "suite": record["suite"],
            "score": evaluation.get("policy_bench_score", {}).get("overall"),
            "report_url": f"/runs/{run_id}/report.html",
        }

    def launch_simulator(self, run_id: str) -> dict[str, Any]:
        """Open an immutable exported policy in Pollen's browser arena.

        This is the product-facing path.  Viser remains available internally
        as an engineering debugger, but the dashboard never routes Play to it.
        """
        manifest = self.bench.load_manifest(run_id)
        task = manifest.get("task")
        preview = {
            "walking": {"slot": "walk", "loco": "legs", "label": "Run"},
            "roller": {"slot": "drive", "loco": "rollers", "label": "Drive"},
            "swizzle": {"slot": "drive", "loco": "rollers", "label": "Swizzle"},
            "hop": {
                "slot": "crouch",
                "loco": "rollers",
                "label": "Hop",
                "period": "3.0",
                "end": "1.0",
            },
            "backflip": {
                "slot": "crouch",
                "loco": "rollers",
                "label": "Backflip",
                "period": "4.0",
                "end": "1.0",
            },
        }.get(task)
        if preview is None:
            raise ValueError(f"Interactive arena preview is not configured for task {task!r}")
        policy = manifest.get("artifacts", {}).get("policy")
        if not policy:
            raise ValueError(
                "This saved model has not exported its browser policy yet. "
                "Choose the newest saved model marked ready, or wait for the next export."
            )
        policy_path = Path(policy["path"])
        if not policy_path.is_file() or sha256(policy_path) != policy["sha256"]:
            raise ValueError("The policy snapshot is missing or failed its hash check")
        try:
            relative = policy_path.resolve().relative_to(self.bench.runs_dir.resolve())
        except ValueError as error:
            raise ValueError("The policy snapshot is outside Policy Bench storage") from error
        policy_url = "/runs/" + quote(relative.as_posix(), safe="/")
        params = {
            "boot": "1",
            "preview_policy": policy_url,
            "preview_slot": preview["slot"],
            "preview_loco": preview["loco"],
            "preview_label": preview["label"],
        }
        if "period" in preview:
            params["preview_period"] = preview["period"]
            params["preview_end"] = preview["end"]
        query = "&".join(f"{key}={quote(value, safe='/')}" for key, value in params.items())
        return {
            "run_id": run_id,
            "task": task,
            "iteration": manifest.get("latest_iteration"),
            "renderer": "pollen-browser-arena",
            "policy_sha256": policy["sha256"],
            "open_url": f"/factory/?{query}",
        }

    def launch_training_viewer(self, run_id: str) -> dict[str, Any]:
        """Render six sampled environments from the newest live checkpoint."""
        return self.launch_viewer(
            run_id,
            num_envs=6,
            kind="training-preview",
            replace_experiment=True,
        )

    def launch_viewer(
        self,
        run_id: str,
        *,
        num_envs: int = 1,
        kind: str = "engineering-debugger",
        replace_experiment: bool = False,
    ) -> dict[str, Any]:
        manifest = self.bench.load_manifest(run_id)
        task = manifest["task"]
        if task not in TASKS:
            raise ValueError(f"Interactive play is not configured for task {task!r}")
        checkpoint = manifest["artifacts"].get("checkpoint")
        if not checkpoint:
            raise ValueError("This candidate has no checkpoint to play")
        checkpoint_path = Path(checkpoint["path"])
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError("The checkpoint snapshot is missing or failed its hash check")
        with self.lock:
            self._reap_finished_locked()
            label = manifest.get("experiment_label") or Path(manifest["source_run_dir"]).name
            if replace_experiment:
                # A new live checkpoint supersedes the old preview for the
                # same training job. Keep at most one six-robot preview so a
                # long run cannot exhaust the bounded viewer port pool.
                for old_run_id, old_session in list(self.viewers.items()):
                    if (
                        old_run_id != run_id
                        and old_session.kind == "training-preview"
                        and old_session.label == label
                    ):
                        self._terminate_viewer(old_session)
                        del self.viewers[old_run_id]
            existing = self.viewers.get(run_id)
            if existing is not None and self._alive(existing.process):
                if existing.num_envs == num_envs and existing.kind == kind:
                    return self._viewer_result(existing, reused=True)
                self._terminate_viewer(existing)
                del self.viewers[run_id]
            reserved = {
                port
                for session in self.viewers.values()
                for port in (session.viser_port, session.controller_port)
            }
            pair = next(
                (
                    (viser_port, controller_port)
                    for viser_port, controller_port in VIEWER_PORT_PAIRS
                    if viser_port not in reserved
                    and controller_port not in reserved
                    and port_available(viser_port)
                    and port_available(controller_port)
                ),
                None,
            )
            if pair is None:
                raise ValueError(
                    "All viewer slots are occupied. Stop an existing simulation in Viewer sessions, "
                    "or close an older terminal-launched viewer."
                )
            viser_port, controller_port = pair
            safe_run_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_id)[:80]
            log_path = LAB_ROOT / "reports" / f"policy-bench-viewer-{safe_run_id}-{int(time.time())}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab", buffering=0)
            command = [
                str(LAB_ROOT / ".tools" / "uv" / "bin" / "uv"),
                "run",
                str(LAB_ROOT / "tools" / "play_viser_compat.py"),
                TASKS[task]["play_task"],
                "--checkpoint-file",
                str(checkpoint_path),
                "--num-envs",
                str(num_envs),
            ]
            command.extend(["--device", "cpu", "--viewer", "viser"])
            environment = os.environ.copy()
            environment["WANDB_MODE"] = "disabled"
            environment["DUCKLAB_VISER_PORT"] = str(viser_port)
            environment["DUCKLAB_GAMEPAD_PORT"] = str(controller_port)
            environment["DUCKLAB_VIEW_KIND"] = kind
            environment["DUCKLAB_VIEW_NUM_ENVS"] = str(num_envs)
            environment["DUCKLAB_DEMO_DIR"] = str(LAB_ROOT / "reports" / "demonstrations")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=UPSTREAM,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                log_handle.close()
                raise
            session = ViewerSession(
                run_id=run_id,
                label=label,
                task=task,
                iteration=manifest.get("latest_iteration"),
                process=process,
                log_handle=log_handle,
                log_path=log_path,
                viser_port=viser_port,
                controller_port=controller_port,
                started_at=time.time(),
                num_envs=num_envs,
                kind=kind,
            )
            self.viewers[run_id] = session
            return self._viewer_result(session, reused=False)

    @staticmethod
    def _viewer_result(session: ViewerSession, reused: bool) -> dict[str, Any]:
        result = {
            "run_id": session.run_id,
            "label": session.label,
            "task": session.task,
            "iteration": session.iteration,
            "reused": reused,
            "viser_url": f"http://localhost:{session.viser_port}",
            "controller_url": (
                f"http://localhost:{session.controller_port}"
                f"/?arena_port={session.viser_port}"
            ),
            "viser_port": session.viser_port,
            "controller_port": session.controller_port,
            "pid": session.process.pid,
            "log": str(session.log_path),
            "started_at": session.started_at,
            "num_envs": session.num_envs,
            "kind": session.kind,
        }
        result["open_url"] = result["viser_url"] if session.task == "hop" else result["controller_url"]
        return result

    @staticmethod
    def _terminate_viewer(session: ViewerSession) -> None:
        try:
            os.killpg(session.process.pid, signal.SIGTERM)
            session.process.wait(timeout=8)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(session.process.pid, signal.SIGKILL)
            session.process.wait(timeout=3)
        finally:
            session.log_handle.close()

    def stop_viewer(self, run_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            self._reap_finished_locked()
            if run_id:
                session = self.viewers.pop(run_id, None)
                if session is None:
                    return {"stopped": False, "message": "That simulation is not running."}
                self._terminate_viewer(session)
                return {"stopped": True, "run_id": run_id, "pid": session.process.pid}
            sessions = list(self.viewers.values())
            self.viewers.clear()
            for session in sessions:
                self._terminate_viewer(session)
            return {
                "stopped": bool(sessions),
                "count": len(sessions),
                "message": f"Stopped {len(sessions)} simulation{'s' if len(sessions) != 1 else ''}." if sessions else "No simulations are running.",
            }

    def start_training(self, config: dict[str, Any]) -> dict[str, Any]:
        task = config.get("task")
        iterations = config.get("iterations")
        environments = config.get("environments")
        resource_profile = config.get("resource_profile", "shared")
        if task not in TASKS:
            raise ValueError("Unknown training task")
        if not isinstance(iterations, int) or not 5 <= iterations <= 100_000:
            raise ValueError("Iterations must be between 5 and 100,000")
        if not isinstance(environments, int) or not 1 <= environments <= 8192:
            raise ValueError("Parallel environments must be between 1 and 8,192")
        if resource_profile not in RESOURCE_PROFILES:
            raise ValueError("Resource profile must be shared or training-priority")
        with self.lock:
            self._reap_finished_locked()
            detected = running_training_processes()
            if self._alive(self.training) or detected:
                raise ValueError("A training process is already running; concurrent full training is blocked.")
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            log_path = LAB_ROOT / "reports" / f"gui-train-{task}-{timestamp}.log"
            self.training_log = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment.update(
                {
                    "DUCKLAB_ITERATIONS": str(iterations),
                    "DUCKLAB_ENVS": str(environments),
                    "WANDB_MODE": "disabled",
                    "DUCKLAB_RESOURCE_PROFILE": resource_profile,
                }
            )
            self.training = subprocess.Popen(
                [str(TASKS[task]["train_script"])],
                cwd=LAB_ROOT,
                env=environment,
                stdout=self.training_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.training_config = dict(config)
            return {
                "started": True,
                "pid": self.training.pid,
                "task": task,
                "iterations": iterations,
                "environments": environments,
                "resource_profile": resource_profile,
                "log": str(log_path),
            }


class DashboardHandler(SimpleHTTPRequestHandler):
    server: "DashboardServer"

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the long-running dashboard terminal readable.

        The browser intentionally polls /api/status, so logging every 200
        response quickly buries useful training and viewer messages.
        """
        if args and "/api/" in str(args[0]):
            return
        super().log_message(format, *args)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Policy-Bench-Token", ""), self.server.control_token
        )

    def _read_json(self, max_bytes: int = 65_536) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > max_bytes:
            raise ValueError("Invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _serve_factory_arena(self) -> None:
        """Serve Pollen's built browser arena through the dashboard tunnel."""
        request_path = unquote(urlsplit(self.path).path)
        if request_path == "/factory":
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", "/factory/")
            self.end_headers()
            return
        relative = request_path.removeprefix("/factory/")
        target = (FACTORY_ARENA_DIST / relative).resolve()
        arena_root = FACTORY_ARENA_DIST.resolve()
        if target != arena_root and arena_root not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = target.read_bytes()
        if target.name == "index.html":
            payload = payload.replace(b'src="/', b'src="/factory/').replace(
                b'href="/', b'href="/factory/'
            )
            capture_script = (
                b"<script>globalThis.__DUCKLAB_CAPTURE_TOKEN__="
                + json.dumps(self.server.control_token).encode("utf-8")
                + b"</script>"
            )
            payload = payload.replace(b"</head>", capture_script + b"</head>")
            # Keep the pinned open-source arena intact while removing Pollen's
            # storefront CTA from DuckLab's locally served product surface.
            # :has() targets the complete Shop HUD plate, not only its anchor.
            ducklab_style = (
                b'<style id="ducklab-arena-overrides">'
                b'div:has(>div>a[href*="store.pollen-robotics.com"]){display:none!important}'
                b'#dark-wing-arena-brand{position:fixed;left:50%;top:18px;transform:translateX(-50%);z-index:2147483646;'
                b'display:flex;align-items:center;gap:9px;padding:8px 11px;border:1px solid rgba(163,126,255,.38);'
                b'border-radius:9px;background:rgba(12,8,23,.72);backdrop-filter:blur(10px);'
                b'box-shadow:0 8px 24px rgba(0,0,0,.28);color:#f6f2ff;font-family:inherit;'
                b'font-size:11px;font-weight:750;letter-spacing:.12em;line-height:1;pointer-events:none;'
                b'text-transform:uppercase}'
                b'#dark-wing-arena-brand i{display:block;width:8px;height:8px;border-radius:50%;'
                b'background:#8157ff;box-shadow:0 0 0 3px rgba(129,87,255,.18)}'
                b'#dark-wing-arena-brand em{color:#f3c969;font-style:normal;font-weight:800}'
                b'</style>'
            )
            payload = payload.replace(b"</head>", ducklab_style + b"</head>")
            dark_wing_brand = (
                b'<div id="dark-wing-arena-brand" aria-label="Dark Wing Duck Enterprise">'
                b'<i></i><span>Dark Wing Duck <em>Enterprise</em></span></div>'
            )
            payload = payload.replace(b"</body>", dark_wing_brand + b"</body>")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store" if target.name == "index.html" else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_bench_run_asset(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        relative = request_path.removeprefix("/runs/")
        root = self.server.bench.runs_dir.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path in {"/factory", "/factory/"} or request_path.startswith("/factory/"):
            self._serve_factory_arena()
            return
        if request_path.startswith("/runs/"):
            self._serve_bench_run_asset()
            return
        # Pollen's app intentionally uses page-relative fetch URLs. Some
        # browser APIs resolve those against the origin root after the bundle
        # is mounted below /factory/, so preserve the upstream paths as local
        # aliases instead of patching Pollen's source.
        if request_path.startswith(("/robot/", "/policies/", "/assets/", "/bundle/")):
            original_path = self.path
            self.path = f"/factory/{request_path.lstrip('/')}"
            try:
                self._serve_factory_arena()
            finally:
                self.path = original_path
            return
        if self.path == "/api/status":
            self._send_json(self.server.manager.status())
            return
        if self.path in {"/", "/index.html"}:
            active = set()
            for process in running_training_processes():
                match = re.search(r"--agent\.run-name\s+([^\s]+)", process["command"])
                if match:
                    active.add(match.group(1))
            path = self.server.bench.render_dashboard(active_experiments=active)
            payload = path.read_text().replace("__CONTROL_TOKEN__", self.server.control_token).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send_json({"error": "Invalid control token"}, HTTPStatus.FORBIDDEN)
            return
        try:
            body = self._read_json(2_500_000 if self.path == "/api/demonstrations" else 65_536)
            if self.path == "/api/demonstrations":
                self._send_json(self.server.save_demonstration(body), HTTPStatus.CREATED)
            elif self.path == "/api/play":
                self._send_json(self.server.manager.launch_simulator(str(body.get("run_id", ""))))
            elif self.path == "/api/watch-training":
                self._send_json(self.server.manager.launch_training_viewer(str(body.get("run_id", ""))))
            elif self.path == "/api/deployment-check":
                self._send_json(self.server.manager.run_deployment_check(str(body.get("run_id", ""))))
            elif self.path == "/api/stop-viewer":
                run_id = str(body.get("run_id", "")).strip() or None
                self._send_json(self.server.manager.stop_viewer(run_id))
            elif self.path == "/api/star":
                run_id = str(body.get("run_id", ""))
                self._send_json(self.server.bench.star(run_id) if body.get("star", True) else self.server.bench.unstar(run_id))
            elif self.path == "/api/train":
                self._send_json(self.server.manager.start_training(body))
            elif self.path == "/api/chat":
                self._send_json(self.server.chat(str(body.get("message", ""))))
            else:
                self._send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except (ValueError, SystemExit, json.JSONDecodeError, subprocess.CalledProcessError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)


class DashboardServer(ThreadingHTTPServer):
    # Lets the dashboard restart cleanly after Ctrl-C or an SSH reconnect.
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], bench: Bench):
        self.bench = bench
        self.control_token = secrets.token_urlsafe(32)
        self.manager = ProcessManager(bench)
        super().__init__(address, lambda *args, **kwargs: DashboardHandler(*args, directory=str(bench.state_dir), **kwargs))

    def save_demonstration(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist an arena state/action trajectory."""
        skill = re.sub(r"[^a-z0-9_-]+", "-", str(body.get("skill", "demo")).lower()).strip("-")
        frames = body.get("frames")
        if not skill or not isinstance(frames, list) or not 25 <= len(frames) <= 400:
            raise ValueError("Demonstration must contain 25-400 frames and a valid skill")
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Invalid demonstration frame")
            for field in ("qpos", "qvel", "action", "command"):
                values = frame.get(field)
                if not isinstance(values, list) or len(values) > 128 or not values:
                    raise ValueError(f"Invalid {field} trajectory")
                if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite value in {field} trajectory")
        DEMONSTRATIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = DEMONSTRATIONS_DIR / f"{skill}-{stamp}.json"
        suffix = 2
        while destination.exists():
            destination = DEMONSTRATIONS_DIR / f"{skill}-{stamp}-{suffix}.json"
            suffix += 1
        destination.write_text(json.dumps(body, separators=(",", ":")))
        return {
            "saved": True,
            "skill": skill,
            "frames": len(frames),
            "seconds": round(len(frames) / float(body.get("control_hz", 50)), 2),
            "path": str(destination.relative_to(LAB_ROOT)),
        }

    def chat(self, message: str) -> dict[str, Any]:
        text = message.strip()
        if not text:
            return {"kind": "message", "reply": "Tell me what you want to train, inspect, or play."}
        lowered = text.lower()
        if "status" in lowered or "running" in lowered:
            status = self.manager.status()
            detected = status["training"]["detected"]
            reply = "Training is running." if detected else "No training process is currently detected."
            if status["viewers"]:
                reply += f" {len(status['viewers'])} simulation(s) are open."
            return {"kind": "status", "reply": reply, "status": status}
        shipped_capabilities = ("walk", "walking", "skate", "skating", "roller", "reverse", "turn", "spin", "sit", "stand", "recover", "kick", "crouch")
        wants_custom_training = any(word in lowered for word in ("custom", "retrain", "improve", "beat", "new policy"))
        asks_for_capability = any(word in lowered for word in shipped_capabilities)
        if asks_for_capability and not wants_custom_training:
            return {
                "kind": "factory-play",
                "reply": "Pollen already ships that capability. Start in the factory playground with the native Xbox controls; train only if a named evaluation gate fails.",
                "url": FACTORY_ARENA_URL,
            }
        request = parse_training_request(text)
        if request is not None:
            if "error" in request:
                planned = codex_training_plan(text)
                if planned and planned.get("unsupported"):
                    return {"kind": "message", "reply": planned.get("reply") or
                            "I understand the goal, but that skill does not have a registered simulator task yet. Add the task and reward first, then I can train it."}
                if planned:
                    request = planned
                else:
                    return {"kind": "message", "reply": request["error"]}
            if "error" in request:
                return {"kind": "message", "reply": request["error"]}
            active = running_training_processes()
            warning = " A training process is already running, so launch will remain blocked." if active else ""
            return {
                "kind": "confirm-training",
                "reply": (
                    f"{request.get('reply', 'Ready to train.')} "
                    f"Configuration: {request['task']} for {request['iterations']:,} iterations "
                    f"with {request['environments']:,} parallel environments in "
                    f"{request.get('resource_profile', 'shared')} mode.{warning}"
                ),
                "action": request,
            }
        if "stop" in lowered and ("viewer" in lowered or "simulation" in lowered):
            result = self.manager.stop_viewer()
            return {"kind": "message", "reply": result.get("message", "Viewer stopped."), "result": result}
        play = re.search(r"\b(?:play|test|drive)\b.*?\b(?:iteration\s*)?(\d+)\b", lowered)
        if play:
            iteration = int(play.group(1))
            matches = [item for item in self.bench.manifests() if item.get("latest_iteration") == iteration]
            if len(matches) == 1:
                result = self.manager.launch_simulator(matches[0]["run_id"])
                return {"kind": "play", "reply": f"Launching iteration {iteration}.", "result": result}
            if not matches:
                return {"kind": "message", "reply": f"I could not find a registered iteration {iteration}. Run discovery first."}
            return {"kind": "message", "reply": f"More than one task has iteration {iteration}; use its Play button."}
        if "run" in lowered or "checkpoint" in lowered or "model" in lowered:
            manifests = self.bench.manifests()
            latest = max(manifests, key=lambda item: item["created_at"]) if manifests else None
            reply = f"There are {len(manifests)} registered candidates."
            if latest:
                reply += f" The newest is {latest['run_id']} at stage {latest['stage']}."
            return {"kind": "message", "reply": reply}
        return {
            "kind": "message",
            "reply": (
                "Tell me a goal in plain English, such as ‘train MicroDuck to skate backwards’. "
                "I’ll check Pollen’s shipped skills first, then map a genuinely new goal to a registered task and wait for confirmation. "
                "I also understand play, status, and stop viewer."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    cleaned = cleanup_orphaned_dashboard_viewers(args.state_dir)
    if cleaned:
        print(f"Cleaned up {cleaned} stale dashboard viewer session(s).", flush=True)
    bench = Bench(args.state_dir)
    bench.initialize()
    bench.render_dashboard()
    server = DashboardServer(("127.0.0.1", args.port), bench)
    print(f"Policy Bench control center: http://127.0.0.1:{args.port}", flush=True)
    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.stop_viewer()
        server.server_close()


if __name__ == "__main__":
    main()
