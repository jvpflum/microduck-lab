"""CPU-only geometry/calibration tests; these do not establish native RTX support."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from isaac_workshop.head_depth import CameraSpec, HeadDepthRecorder, clock_progress, depth_geometry, rotation_xyzw, head_depth_observation, _png

try:
    import torch
except ImportError:
    torch = None


class HeadDepthTests(unittest.TestCase):
    def setUp(self):
        self.spec = CameraSpec(width=8, height=6)
        self.k = np.array(self.spec.manifest()["requested_intrinsics"])
        # Camera optical Z points down, ROS X points world X; camera at z=1m.
        self.q_down = (1., 0., 0., 0.)
        self.pos = (0., 0., 1.)

    def test_exact_floor_is_metric_optical_depth_at_all_pixels(self):
        depth = np.ones((6, 8), dtype=np.float32)
        result, valid = depth_geometry(depth, self.k, self.pos, self.q_down, self.spec)
        self.assertTrue(valid.all())
        self.assertEqual(result["floor_consistent_pixels"], 48)
        self.assertEqual(result["behind_floor_pixels"], 0)

    def test_closer_obstacles_are_not_floor_failures(self):
        depth = np.ones((6, 8), dtype=np.float32)
        depth[:, :4] = .5
        result, _ = depth_geometry(depth, self.k, self.pos, self.q_down, self.spec)
        self.assertEqual(result["floor_consistent_pixels"], 24)
        self.assertEqual(result["behind_floor_pixels"], 0)

    def test_wrong_depth_scale_places_surface_behind_floor(self):
        result, _ = depth_geometry(np.full((6, 8), 1.2), self.k, self.pos, self.q_down, self.spec)
        self.assertEqual(result["behind_floor_pixels"], 48)

    def test_radial_range_is_not_accepted_as_optical_depth(self):
        rows, cols = np.indices((6, 8))
        radial = np.sqrt(1 + ((cols-self.k[0, 2])/self.k[0, 0])**2 + ((rows-self.k[1, 2])/self.k[1, 1])**2)
        result, _ = depth_geometry(radial, self.k, self.pos, self.q_down, self.spec)
        self.assertGreater(result["behind_floor_fraction"], .5)

    def test_background_infinity_is_distinct_from_nan(self):
        depth = np.ones((6, 8), dtype=np.float32)
        depth[0, 0], depth[0, 1], depth[0, 2] = np.inf, np.nan, -np.inf
        result, valid = depth_geometry(depth, self.k, self.pos, self.q_down, self.spec)
        self.assertEqual(result["positive_infinity_pixels"], 1)
        self.assertEqual(result["nan_pixels"], 1)
        self.assertEqual(result["negative_infinity_pixels"], 1)
        self.assertEqual(valid.sum(), 45)

    def test_camera_mount_points_forward_and_down_from_neutral_head(self):
        # Measured USD chain at neutral: jaw +X -> robot +Z, jaw -Z -> robot +X.
        head_to_root = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
        forward = head_to_root @ rotation_xyzw(self.spec.quaternion_xyzw) @ np.array([1., 0., 0.])
        np.testing.assert_allclose(forward, [math.cos(math.radians(20)), 0, -math.sin(math.radians(20))], atol=1e-7)

    def test_wrong_quaternion_or_shapes_fail(self):
        with self.assertRaises(ValueError):
            depth_geometry(np.ones((6, 8)), self.k, self.pos, (0, 0, 0, 0), self.spec)
        with self.assertRaises(ValueError):
            depth_geometry(np.ones((8, 6)), self.k, self.pos, self.q_down, self.spec)

    def test_clock_requires_actual_progress_and_explicit_reset_epoch(self):
        old = (0, 10, .2)
        self.assertFalse(clock_progress(old, 0, 10, .2))
        self.assertTrue(clock_progress(old, 0, 11, .22))
        self.assertFalse(clock_progress(old, 0, 1, 0, reset=True))
        self.assertTrue(clock_progress(old, 1, 1, 0, reset=True))
        self.assertFalse(clock_progress(old, 1, 0, -1, reset=True))

    def test_png_has_real_image_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"frame.png"
            _png(p, np.zeros((6, 8, 3), dtype=np.uint8))
            data = p.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(int.from_bytes(data[16:20], "big"), 8)
            self.assertEqual(int.from_bytes(data[20:24], "big"), 6)

    def test_summary_accepts_background_and_nearer_obstacles(self):
        # Exercise the actual final gate with recorded geometry, not fake native support.
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HeadDepthRecorder.__new__(HeadDepthRecorder)
            recorder.output = Path(tmp)
            recorder.spec = self.spec
            recorder.manifest = {"test_fixture": True}
            recorder.final = None
            depth = np.ones((6, 8))
            depth[:, :3] = .5
            depth[:, 7] = np.inf
            geometry, _ = depth_geometry(depth, self.k, self.pos, self.q_down, self.spec)
            recorder.rows = [{"warmup": False, "geometry": geometry,
                              "capture_fields_valid": True, "rgb_spatially_nonuniform": True,
                              "frame_and_timestamp_progress": True, "rigid_head_attachment_verified": True}]
            (recorder.output/"sensor-trace.jsonl").write_text("fixture\n")
            summary = recorder.finalize()
            self.assertTrue(summary["collected_successfully"])
            self.assertFalse(summary["raw_depth_all_finite"])
            self.assertEqual(summary["floor_consistent_pixels"], 24)

    def test_warmup_only_cannot_qualify_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HeadDepthRecorder.__new__(HeadDepthRecorder)
            recorder.output = Path(tmp)
            recorder.spec, recorder.manifest, recorder.final = self.spec, {}, None
            recorder.rows = [{"warmup": True}]
            (recorder.output/"sensor-trace.jsonl").write_text("fixture\n")
            self.assertFalse(recorder.finalize()["collected_successfully"])




if __name__ == "__main__":
    unittest.main()
