"""Launch mjlab Viser with fixed-range sliders and browser gamepad control."""

import atexit
import os
import sys
import time

# Keep the gamepad HTTP thread responsive while the CPU viewer performs its
# physics/inference loop. The default interpreter switch interval is too coarse
# on the Spark for timely controller packets.
sys.setswitchinterval(0.001)

from gamepad_bridge import GamepadBridge

from viser._gui_api import GuiApi


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
            print(
                f"[gamepad] applied x={command.command_x:+.3f} heading={command.heading:+.3f} "
                f"connected={command.connected} stale={command.stale}",
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
        bridge.start()
        bridge.bind_viewer(self)

    def close(self):
        try:
            super().close()
        finally:
            bridge.close()


play_module.ViserPlayViewer = GamepadViserPlayViewer
main = play_module.main


if __name__ == "__main__":
    main()
