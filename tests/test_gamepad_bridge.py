import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.gamepad_bridge import GamepadState


class GamepadStateTests(unittest.TestCase):
    def test_disarmed_state_does_not_override(self):
        state = GamepadState()
        state.update({"armed": False, "connected": True, "command_x": 0.4})
        command = state.snapshot()
        self.assertFalse(command.override)

    def test_armed_connected_command_is_clamped(self):
        state = GamepadState()
        state.update(
            {"armed": True, "connected": True, "command_x": 4.0, "heading": -4.0}
        )
        command = state.snapshot()
        self.assertTrue(command.override)
        self.assertEqual(command.command_x, 1.0)
        self.assertEqual(command.heading, -1.0)

    def test_disconnect_estop_and_timeout_force_zero(self):
        state = GamepadState(timeout_s=0.01)
        state.update({"armed": True, "connected": False, "command_x": 0.5})
        self.assertEqual(state.snapshot().command_x, 0.0)
        state.update(
            {
                "armed": True,
                "connected": True,
                "command_x": 0.5,
                "emergency_stop": True,
            }
        )
        self.assertEqual(state.snapshot().command_x, 0.0)
        state.update({"armed": True, "connected": True, "command_x": 0.5})
        time.sleep(0.02)
        command = state.snapshot()
        self.assertTrue(command.stale)
        self.assertEqual(command.command_x, 0.0)

    def test_reset_and_pause_are_rising_edge_actions(self):
        viewer = Mock()
        state = GamepadState()
        state.bind_viewer(viewer)
        payload = {"armed": True, "connected": True, "reset": True, "pause": True}
        state.update(payload)
        state.update(payload)
        self.assertEqual(viewer.request_reset.call_count, 1)
        self.assertEqual(viewer.request_toggle_pause.call_count, 1)
        state.update({**payload, "reset": False, "pause": False})
        state.update(payload)
        self.assertEqual(viewer.request_reset.call_count, 2)
        self.assertEqual(viewer.request_toggle_pause.call_count, 2)

    def test_controller_page_is_self_contained(self):
        page = Path(__file__).parents[1] / "tools/gamepad_controller.html"
        text = page.read_text()
        self.assertIn("navigator.getGamepads", text)
        self.assertIn("/api/state", text)
        self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main()
