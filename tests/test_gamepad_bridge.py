import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.gamepad_bridge import ControllerOwnershipError, GamepadState


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

    def test_controller_diagnostics_are_bounded_and_reported(self):
        state = GamepadState()
        state.update({
            "armed": True,
            "connected": True,
            "gamepad_id": "Xbox Wireless Controller",
            "mapping": "standard",
            "axes": [0.25, -0.5, 4.0],
        })
        command = state.snapshot()
        self.assertEqual(command.gamepad_id, "Xbox Wireless Controller")
        self.assertEqual(command.mapping, "standard")
        self.assertEqual(command.axes, (0.25, -0.5, 1.0))

    def test_active_controller_tab_cannot_be_overwritten_without_takeover(self):
        state = GamepadState()
        state.update({"client_id": "tab-a", "armed": True, "connected": True, "command_x": 0.4})
        with self.assertRaisesRegex(ControllerOwnershipError, "Another controller tab"):
            state.update({"client_id": "tab-b", "armed": True, "connected": True, "command_x": 0.0})
        self.assertEqual(state.snapshot().command_x, 0.4)
        state.update({"client_id": "tab-b", "takeover": True, "armed": True, "connected": True, "command_x": -0.3})
        command = state.snapshot()
        self.assertEqual(command.client_id, "tab-b")
        self.assertEqual(command.command_x, -0.3)

    def test_controller_page_is_self_contained(self):
        page = Path(__file__).parents[1] / "tools/gamepad_controller.html"
        text = page.read_text()
        self.assertIn("navigator.getGamepads", text)
        self.assertIn("/api/state", text)
        self.assertIn("driveSource", text)
        self.assertIn("Hold to test forward", text)
        self.assertIn("Raw axes", text)
        self.assertNotIn("https://", text)


if __name__ == "__main__":
    unittest.main()
