"""Exercise adapter math without launching Isaac Lab or requiring a GPU.

Load only the pure definitions from the standalone remote replay script; its
top-level imports and launcher must never execute during these contract tests.
"""

import ast
import math
from pathlib import Path
import unittest

import numpy as np


source = ast.parse(Path(__file__).with_name("v89_replay.py").read_text())
definitions = {"History", "inverse_rotate_one", "tilt_from_quat", "tilt_from_isaac_quat", "command_for"}
module = ast.Module(body=[node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions], type_ignores=[])
scope = {"np": np, "math": math, "TERM_WIDTHS": (3, 3, 14, 14, 14, 3, 4, 6), "HISTORY_LENGTH": 256}
exec(compile(module, "replay_pure_contract", "exec"), scope)


class ReplayContractTests(unittest.TestCase):
    def test_isaac_identity_and_yaw_do_not_count_as_a_fall(self):
        tilt = scope["tilt_from_isaac_quat"]
        for angle in (0, math.pi / 2, math.pi, 4 * math.pi):
            q = np.array([0, 0, math.sin(angle / 2), math.cos(angle / 2)])
            self.assertAlmostEqual(tilt(q), 0)
            self.assertAlmostEqual(tilt(-q), 0)

    def test_isaac_roll_and_pitch_report_physical_tilt(self):
        for axis in (0, 1):
            for degrees in (30, 45, 90, 180):
                angle = math.radians(degrees)
                q = np.zeros(4)
                q[axis], q[3] = math.sin(angle / 2), math.cos(angle / 2)
                self.assertAlmostEqual(scope["tilt_from_isaac_quat"](q), angle)

    def test_projected_gravity_is_body_frame(self):
        q = np.array([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
        actual = scope["inverse_rotate_one"](q, np.array([0, 0, -1]))
        np.testing.assert_allclose(actual, [0, -1, 0], atol=1e-7)

    def test_term_major_history_has_correct_age_and_term_boundaries(self):
        history = scope["History"]()
        for time in range(258):
            packed = history.pack(np.arange(61, dtype=np.float32) + 1000 * time)
        # Last 256 chronological samples must be times 2 through 257.
        for channel, width, packed_start in ((0, 3, 0), (3, 3, 768), (6, 14, 1536), (20, 14, 5120), (34, 14, 8704), (48, 3, 12288), (51, 4, 13056), (55, 6, 14080)):
            term = packed[packed_start:packed_start + 256 * width].reshape(256, width)
            expected = np.arange(channel, channel + width)[None, :] + np.arange(2, 258)[:, None] * 1000
            np.testing.assert_array_equal(term, expected)

    def test_new_history_does_not_retain_prior_episode(self):
        old = scope["History"]()
        old.pack(np.full(61, 7))
        fresh = scope["History"]()
        np.testing.assert_array_equal(fresh.pack(np.zeros(61)), np.zeros(15616))
        np.testing.assert_array_equal(old.pack(np.full(61, 7)), np.full(15616, 7))

    def test_launch_and_stop_boundaries(self):
        command = scope["command_for"]
        for step, expected in ((0, 0), (99, 0), (100, .8), (249, .8), (250, 0), (499, 0)):
            self.assertAlmostEqual(float(command(step)[0]), expected, places=6)
            np.testing.assert_array_equal(command(step)[1:], np.zeros(12))


if __name__ == "__main__":
    unittest.main()
