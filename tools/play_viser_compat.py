"""Launch mjlab Viser with fixed-range sliders and browser gamepad control."""

import atexit
import os
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np

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
_num_envs = int(_argument_value("--num-envs") or "1")
_training_preview = os.environ.get("DUCKLAB_VIEW_KIND") == "training-preview"


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
        step_dt = float(self.env.unwrapped.step_dt)
        self._demo_samples = deque(maxlen=max(10, int(6.0 / step_dt) + 2))
        self._demo_lock = Lock()
        self._server.gui.add_markdown(
            "### Dark Wing Duck Enterprise\n"
            f"**{'Live training snapshot' if _training_preview else 'Pollen microduck_rl · Mjlab'}**  \n"
            f"Task: `{_task_id}`  \nCheckpoint: `{_checkpoint_name}`"
            + (f"  \nRobots shown: **{_num_envs}**" if _training_preview else "")
        )
        with self._server.gui.add_folder("Demonstration recorder"):
            self._demo_label = self._server.gui.add_dropdown(
                "Skill",
                options=["Backflip", "Front flip", "Hop", "Other"],
                initial_value="Backflip",
            )
            self._demo_status = self._server.gui.add_markdown(
                "Recorder armed. Perform the move, then save the last five seconds."
            )
            save_demo = self._server.gui.add_button("Save last attempt")

            @save_demo.on_click
            def _(_) -> None:
                self._save_demo()
        bridge.start()
        bridge.bind_viewer(self)

    def setup(self):
        if _training_preview and _num_envs > 1:
            # The task's native camera convention uses a negative elevation,
            # which puts Viser below the floor and tracks only env 0. Frame the
            # complete training sample from above instead.
            self.cfg.distance = 3.0
            self.cfg.azimuth = 45.0
            self.cfg.elevation = 32.0
        super().setup()
        if _training_preview and _num_envs > 1:
            self._scene.camera_tracking_enabled = False

    def _execute_step(self):
        success = super()._execute_step()
        if success:
            self._capture_demo_sample()
        return success

    def _capture_demo_sample(self):
        env = self.env.unwrapped
        env_idx = int(getattr(getattr(self, "_scene", None), "env_idx", 0))
        sim = env.sim
        action = getattr(env.action_manager, "action", None)
        sample = {
            "env_idx": env_idx,
            "sim_time": float(self._step_count * env.step_dt),
            "qpos": sim.data.qpos[env_idx].detach().cpu().numpy().copy(),
            "qvel": sim.data.qvel[env_idx].detach().cpu().numpy().copy(),
            "action": (
                action[env_idx].detach().cpu().numpy().copy()
                if action is not None
                else np.empty(0, dtype=np.float32)
            ),
        }
        with self._demo_lock:
            self._demo_samples.append(sample)

    def _save_demo(self):
        env_idx = int(getattr(getattr(self, "_scene", None), "env_idx", 0))
        with self._demo_lock:
            samples = [sample for sample in self._demo_samples if sample["env_idx"] == env_idx]
        if len(samples) < 5:
            self._demo_status.content = "Not enough motion recorded yet. Run the simulation, then try again."
            return
        # Keep the last five seconds even though the ring has a one-second
        # guard band for slow viewers.
        cutoff = samples[-1]["sim_time"] - 5.0
        samples = [sample for sample in samples if sample["sim_time"] >= cutoff]
        safe_label = self._demo_label.value.lower().replace(" ", "-")
        output_dir = Path(
            os.environ.get("DUCKLAB_DEMO_DIR", str(Path.cwd() / "demonstrations"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = output_dir / f"{safe_label}-{stamp}-{_checkpoint_name.removesuffix('.pt')}.npz"
        np.savez_compressed(
            output,
            qpos=np.stack([sample["qpos"] for sample in samples]),
            qvel=np.stack([sample["qvel"] for sample in samples]),
            action=np.stack([sample["action"] for sample in samples]),
            sim_time=np.asarray([sample["sim_time"] for sample in samples]),
            env_idx=np.asarray(env_idx),
            task=np.asarray(_task_id),
            checkpoint=np.asarray(_checkpoint_name),
            skill=np.asarray(self._demo_label.value),
        )
        duration = samples[-1]["sim_time"] - samples[0]["sim_time"]
        self._demo_status.content = (
            f"Saved **{self._demo_label.value}** · {duration:.1f}s · {len(samples)} frames  \n"
            f"`{output.name}`"
        )
        print(f"[demo] saved {output}", flush=True)

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
