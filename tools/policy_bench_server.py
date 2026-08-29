#!/usr/bin/env python3
"""Local control server for the MicroDuck Policy Bench dashboard."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import shutil
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from policy_bench import Bench, DEFAULT_STATE, LAB_ROOT, UPSTREAM, sha256


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
}


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
        if re.search(r"(?:^|/)train\s+Mjlab-", command):
            matches.append({"pid": int(entry.name), "command": command.strip()})
    return sorted(matches, key=lambda item: item["pid"])


def parse_training_request(message: str) -> dict[str, Any] | None:
    lowered = message.lower()
    if not re.search(r"\b(train|training|learn)\b", lowered):
        return None
    if "roller" in lowered:
        task = "roller"
    elif "swizzle" in lowered or "skate" in lowered:
        task = "swizzle"
    elif "walk" in lowered:
        task = "walking"
    else:
        return {"error": "Tell me which skill to train: swizzle, roller, or walking."}
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
    return {"task": task, "iterations": iterations, "environments": environments}


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
        "task, iterations, environments, supported, reply. task must be one of "
        "swizzle, roller, walking, or custom. Choose custom for a skill that has "
        "no registered simulator task. Never invent a runnable task. Defaults are "
        "8000 iterations and 4096 environments. Keep reply under 240 characters.\n"
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
    return {"task": data["task"], "iterations": iterations, "environments": environments,
            "reply": str(data.get("reply") or "I mapped that to a validated training task.")}


class ProcessManager:
    def __init__(self, bench: Bench):
        self.bench = bench
        self.lock = threading.Lock()
        self.viewer: subprocess.Popen[bytes] | None = None
        self.viewer_run_id: str | None = None
        self.viewer_log = None
        self.training: subprocess.Popen[bytes] | None = None
        self.training_config: dict[str, Any] | None = None
        self.training_log = None

    @staticmethod
    def _alive(process: subprocess.Popen[bytes] | None) -> bool:
        return process is not None and process.poll() is None

    def _reap_finished_locked(self) -> None:
        if self.viewer is not None and self.viewer.poll() is not None:
            self.viewer = None
            self.viewer_run_id = None
            if self.viewer_log:
                self.viewer_log.close()
                self.viewer_log = None
        if self.training is not None and self.training.poll() is not None:
            self.training = None
            self.training_config = None
            if self.training_log:
                self.training_log.close()
                self.training_log = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._reap_finished_locked()
            viewer_alive = self._alive(self.viewer)
            training_alive = self._alive(self.training)
            return {
                "viewer": {
                    "running": viewer_alive,
                    "run_id": self.viewer_run_id if viewer_alive else None,
                    "pid": self.viewer.pid if viewer_alive and self.viewer else None,
                },
                "training": {
                    "managed_running": training_alive,
                    "config": self.training_config if training_alive else None,
                    "pid": self.training.pid if training_alive and self.training else None,
                    "detected": running_training_processes(),
                },
            }

    def launch_viewer(self, run_id: str) -> dict[str, Any]:
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
            if self._alive(self.viewer):
                if self.viewer_run_id == run_id:
                    return self._viewer_result(run_id, reused=True)
                raise ValueError("Another dashboard-managed viewer is running. Stop it before playing a different run.")
            unavailable = [port for port in (8080, 8090) if not port_available(port)]
            if unavailable:
                raise ValueError(
                    "Viewer ports are already occupied: "
                    + ", ".join(str(port) for port in unavailable)
                    + ". Stop the older Viser/controller session first."
                )
            log_path = LAB_ROOT / "reports" / f"policy-bench-viewer-{int(time.time())}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.viewer_log = log_path.open("ab", buffering=0)
            command = [
                str(LAB_ROOT / ".tools" / "uv" / "bin" / "uv"),
                "run",
                str(LAB_ROOT / "tools" / "play_viser_compat.py"),
                TASKS[task]["play_task"],
                "--checkpoint-file",
                str(checkpoint_path),
                "--num-envs",
                "1",
                "--device",
                "cpu",
                "--viewer",
                "viser",
            ]
            environment = os.environ.copy()
            environment["WANDB_MODE"] = "disabled"
            environment["DUCKLAB_GAMEPAD_PORT"] = "8090"
            self.viewer = subprocess.Popen(
                command,
                cwd=UPSTREAM,
                env=environment,
                stdout=self.viewer_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.viewer_run_id = run_id
            return self._viewer_result(run_id, reused=False, log_path=log_path)

    def _viewer_result(self, run_id: str, reused: bool, log_path: Path | None = None) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "reused": reused,
            "viser_url": "http://localhost:8080",
            "controller_url": "http://localhost:8090",
            "pid": self.viewer.pid if self.viewer else None,
            "log": str(log_path) if log_path else None,
        }

    def stop_viewer(self) -> dict[str, Any]:
        with self.lock:
            self._reap_finished_locked()
            if not self._alive(self.viewer):
                self.viewer = None
                self.viewer_run_id = None
                return {"stopped": False, "message": "No dashboard-managed viewer is running."}
            assert self.viewer is not None
            pid = self.viewer.pid
            try:
                os.killpg(pid, signal.SIGTERM)
                self.viewer.wait(timeout=8)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                self.viewer.wait(timeout=3)
            self.viewer = None
            self.viewer_run_id = None
            if self.viewer_log:
                self.viewer_log.close()
                self.viewer_log = None
            return {"stopped": True, "pid": pid}

    def start_training(self, config: dict[str, Any]) -> dict[str, Any]:
        task = config.get("task")
        iterations = config.get("iterations")
        environments = config.get("environments")
        if task not in TASKS:
            raise ValueError("Unknown training task")
        if not isinstance(iterations, int) or not 5 <= iterations <= 100_000:
            raise ValueError("Iterations must be between 5 and 100,000")
        if not isinstance(environments, int) or not 1 <= environments <= 8192:
            raise ValueError("Parallel environments must be between 1 and 8,192")
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
                "log": str(log_path),
            }


class DashboardHandler(SimpleHTTPRequestHandler):
    server: "DashboardServer"

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

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65_536:
            raise ValueError("Invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            self._send_json(self.server.manager.status())
            return
        if self.path in {"/", "/index.html"}:
            path = self.server.bench.render_dashboard()
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
            body = self._read_json()
            if self.path == "/api/play":
                self._send_json(self.server.manager.launch_viewer(str(body.get("run_id", ""))))
            elif self.path == "/api/stop-viewer":
                self._send_json(self.server.manager.stop_viewer())
            elif self.path == "/api/star":
                run_id = str(body.get("run_id", ""))
                self._send_json(self.server.bench.star(run_id) if body.get("star", True) else self.server.bench.unstar(run_id))
            elif self.path == "/api/train":
                self._send_json(self.server.manager.start_training(body))
            elif self.path == "/api/chat":
                self._send_json(self.server.chat(str(body.get("message", ""))))
            else:
                self._send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except (ValueError, SystemExit, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], bench: Bench):
        self.bench = bench
        self.manager = ProcessManager(bench)
        self.control_token = secrets.token_urlsafe(32)
        super().__init__(address, lambda *args, **kwargs: DashboardHandler(*args, directory=str(bench.state_dir), **kwargs))

    def chat(self, message: str) -> dict[str, Any]:
        text = message.strip()
        if not text:
            return {"kind": "message", "reply": "Tell me what you want to train, inspect, or play."}
        lowered = text.lower()
        if "status" in lowered or "running" in lowered:
            status = self.manager.status()
            detected = status["training"]["detected"]
            reply = "Training is running." if detected else "No training process is currently detected."
            if status["viewer"]["running"]:
                reply += f" Viewer is playing {status['viewer']['run_id']}."
            return {"kind": "status", "reply": reply, "status": status}
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
                    f"with {request['environments']:,} parallel environments.{warning}"
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
                result = self.manager.launch_viewer(matches[0]["run_id"])
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
                "I’ll map it to a registered task, show the plan, and wait for your confirmation. "
                "I also understand play, status, and stop viewer."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    bench = Bench(args.state_dir)
    bench.initialize()
    bench.render_dashboard()
    server = DashboardServer(("127.0.0.1", args.port), bench)
    print(f"Policy Bench control center: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.stop_viewer()
        server.server_close()


if __name__ == "__main__":
    main()
