"""Launch mjlab Viser with fixed-range sliders and browser gamepad control."""

import atexit
import os
import sys
import time
from pathlib import Path

# Keep the gamepad HTTP thread responsive while the CPU viewer performs its
# physics/inference loop. The default interpreter switch interval is too coarse
# on the Spark for timely controller packets.
sys.setswitchinterval(0.001)

from gamepad_bridge import GamepadBridge

import viser
from viser._gui_api import GuiApi


_ViserServer = viser.ViserServer


def _argument_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


_task_id = next((argument for argument in sys.argv[1:] if argument.startswith("Mjlab-")), "Mjlab policy")
_checkpoint_file = _argument_value("--checkpoint-file")
_checkpoint_name = Path(_checkpoint_file).name if _checkpoint_file else "unknown checkpoint"


def _viser_server_on_session_port(*args, **kwargs):
    kwargs.setdefault("port", int(os.environ.get("DUCKLAB_VISER_PORT", "8080")))
    kwargs["label"] = "Dark Wing · Pollen microduck_rl"
    return _ViserServer(*args, **kwargs)


# mjlab constructs its server internally. Route it to the port assigned by the
# dashboard before importing mjlab.scripts.play.
viser.ViserServer = _viser_server_on_session_port


_add_slider = GuiApi.add_slider


def _add_slider_clamped(self, label, *, min, max, step, initial_value, **kwargs):
    # mjlab creates dashboard controls for fixed command axes whose configured
    # maximum is zero. Viser requires the initial value to be within its GUI
    # slider bounds. Clamp only the displayed control; simulation ranges stay
    # unchanged.
    initial_value = min if initial_value < min else max if initial_value > max else initial_value
    return _add_slider(
        self,
        label,
        min=min,
        max=max,
        step=step,
        initial_value=initial_value,
        **kwargs,
    )


GuiApi.add_slider = _add_slider_clamped

bridge = GamepadBridge(
    port=int(os.environ.get("DUCKLAB_GAMEPAD_PORT", "8090")),
    timeout_s=float(os.environ.get("DUCKLAB_GAMEPAD_TIMEOUT", "5.0")),
)
_last_gamepad_debug = 0.0
atexit.register(bridge.close)

# The roller and swizzle tasks both use this command type. Preserve its normal
# training/play update, then apply the controller override only while armed.
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly


def _apply_gamepad(self):
    global _last_gamepad_debug
    command = bridge.state.snapshot()
    if command.override:
        self.vel_command_b[:, 0] = command.command_x
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = command.heading
        now = time.monotonic()
        if now - _last_gamepad_debug >= 1.0:
            root_xy = self.robot.data.root_link_pos_w[0, :2].detach().cpu().tolist()
            print(
                f"[gamepad] applied x={command.command_x:+.3f} heading={command.heading:+.3f} "
                f"connected={command.connected} stale={command.stale} "
                f"root=({root_xy[0]:+.3f},{root_xy[1]:+.3f})",
                flush=True,
            )
            _last_gamepad_debug = now


_compute_velocity_command = VelocityCommandCommandOnly.compute


def _compute_velocity_with_gamepad(self, dt):
    # mjlab applies its Viser joystick sliders at the end of compute(). Apply
    # the browser controller after that step so static slider values cannot
    # overwrite an armed Xbox command.
    _compute_velocity_command(self, dt)
    _apply_gamepad(self)


VelocityCommandCommandOnly.compute = _compute_velocity_with_gamepad

import mjlab.scripts.play as play_module

_ViserPlayViewer = play_module.ViserPlayViewer


class GamepadViserPlayViewer(_ViserPlayViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._server.gui.add_markdown(
            "### Dark Wing Duck Enterprise\n"
            f"**Pollen microduck_rl · Mjlab**  \nTask: `{_task_id}`  \nCheckpoint: `{_checkpoint_name}`"
        )
        bridge.start()
        bridge.bind_viewer(self)

    def close(self):
        try:
            super().close()
        finally:
            bridge.close()

    def request_resume(self):
        """Resume on the rising edge of a drive command, without toggling a running sim."""
        if self._is_paused:
            self.request_toggle_pause()


play_module.ViserPlayViewer = GamepadViserPlayViewer
main = play_module.main


if __name__ == "__main__":
    main()
