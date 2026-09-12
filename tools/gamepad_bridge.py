"""Local HTTP bridge from the browser Gamepad API to mjlab command tensors."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit


class ControllerOwnershipError(ValueError):
    """Raised when a background controller tab tries to overwrite the owner."""


class ViewerControls(Protocol):
    def request_reset(self) -> None: ...
    def request_toggle_pause(self) -> None: ...
    def request_resume(self) -> None: ...


@dataclass(frozen=True)
class GamepadCommand:
    armed: bool
    connected: bool
    stale: bool
    command_x: float
    heading: float
    emergency_stop: bool
    updated_at: float
    gamepad_id: str
    mapping: str
    axes: tuple[float, ...]
    client_id: str

    @property
    def override(self) -> bool:
        return self.armed


class GamepadState:
    def __init__(self, timeout_s: float = 0.5) -> None:
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._armed = False
        self._connected = False
        self._command_x = 0.0
        self._heading = 0.0
        self._emergency_stop = False
        self._updated_at = 0.0
        self._buttons = {"reset": False, "pause": False}
        self._gamepad_id = ""
        self._mapping = ""
        self._axes: tuple[float, ...] = ()
        self._client_id = ""
        self._client_seen_at = 0.0
        self._viewer: ViewerControls | None = None

    def bind_viewer(self, viewer: ViewerControls) -> None:
        with self._lock:
            self._viewer = viewer

    def update(self, payload: dict[str, Any]) -> None:
        callbacks: list[str] = []
        now = time.monotonic()
        raw_axes = payload.get("axes", [])
        if not isinstance(raw_axes, list):
            raise ValueError("axes must be an array")
        axes = tuple(max(-1.0, min(1.0, float(value))) for value in raw_axes[:16])
        client_id = str(payload.get("client_id") or "legacy")[:120]
        takeover = bool(payload.get("takeover", False))
        with self._lock:
            owner_expired = now - self._client_seen_at > 1.5
            if self._client_id and self._client_id != client_id and not owner_expired and not takeover:
                raise ControllerOwnershipError(
                    "Another controller tab owns this simulator. Click Arm controller in this tab to take control."
                )
            self._client_id = client_id
            self._client_seen_at = now
            was_moving = (
                self._armed
                and self._connected
                and not self._emergency_stop
                and (abs(self._command_x) > 1e-3 or abs(self._heading) > 1e-3)
            )
            self._armed = bool(payload.get("armed", False))
            self._connected = bool(payload.get("connected", False))
            self._emergency_stop = bool(payload.get("emergency_stop", False))
            self._command_x = max(-1.0, min(1.0, float(payload.get("command_x", 0.0))))
            self._heading = max(-1.0, min(1.0, float(payload.get("heading", 0.0))))
            is_moving = (
                self._armed
                and self._connected
                and not self._emergency_stop
                and (abs(self._command_x) > 1e-3 or abs(self._heading) > 1e-3)
            )
            if is_moving and not was_moving:
                callbacks.append("resume")
            self._gamepad_id = str(payload.get("gamepad_id", ""))[:240]
            self._mapping = str(payload.get("mapping", ""))[:40]
            self._axes = axes
            self._updated_at = now

            for name in ("reset", "pause"):
                pressed = bool(payload.get(name, False))
                if pressed and not self._buttons[name]:
                    callbacks.append(name)
                self._buttons[name] = pressed
            viewer = self._viewer

        if viewer is not None:
            for name in callbacks:
                if name == "reset":
                    viewer.request_reset()
                elif name == "pause":
                    viewer.request_toggle_pause()
                elif name == "resume":
                    viewer.request_resume()

    def snapshot(self, now: float | None = None) -> GamepadCommand:
        current = time.monotonic() if now is None else now
        with self._lock:
            stale = self._armed and current - self._updated_at > self.timeout_s
            safe_zero = stale or not self._connected or self._emergency_stop
            return GamepadCommand(
                armed=self._armed,
                connected=self._connected,
                stale=stale,
                command_x=0.0 if safe_zero else self._command_x,
                heading=0.0 if safe_zero else self._heading,
                emergency_stop=self._emergency_stop,
                updated_at=self._updated_at,
                gamepad_id=self._gamepad_id,
                mapping=self._mapping,
                axes=self._axes,
                client_id=self._client_id,
            )


class GamepadBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8090,
        timeout_s: float = 0.5,
        page_path: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.state = GamepadState(timeout_s=timeout_s)
        self.page_path = page_path or Path(__file__).with_name("gamepad_controller.html")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def bind_viewer(self, viewer: ViewerControls) -> None:
        self.state.bind_viewer(viewer)

    def start(self) -> None:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                request_path = urlsplit(self.path).path
                if request_path == "/":
                    self.send_bytes(bridge.page_path.read_bytes(), "text/html; charset=utf-8")
                elif request_path == "/api/state":
                    state = asdict(bridge.state.snapshot())
                    self.send_bytes(json.dumps(state).encode(), "application/json")
                else:
                    self.send_bytes(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if urlsplit(self.path).path != "/api/state":
                    self.send_bytes(b"not found\n", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", "0")), 16_384)
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("payload must be an object")
                    if not str(payload.get("client_id", "")).strip():
                        raise ControllerOwnershipError(
                            "This controller tab is outdated. Close it and reopen the Xbox controller from Policy Bench."
                        )
                    bridge.state.update(payload)
                except ControllerOwnershipError as exc:
                    self.send_bytes(str(exc).encode(), "text/plain", HTTPStatus.CONFLICT)
                    return
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self.send_bytes(str(exc).encode(), "text/plain", HTTPStatus.BAD_REQUEST)
                    return
                self.send_bytes(b'{"ok":true}', "application/json")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="microduck-gamepad-http",
            daemon=True,
        )
        self._thread.start()
        print(f"Gamepad controller: http://localhost:{self.port}")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
