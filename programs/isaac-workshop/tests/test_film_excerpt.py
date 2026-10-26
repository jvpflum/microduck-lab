"""Synthetic camera checks; no raw poses, licensed assets or native renders."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from isaac_workshop import film_cameras as original
from isaac_workshop import film_excerpt_cameras as excerpt

ROOT = Path(__file__).resolve().parents[1]


class ExcerptCameraTests(unittest.TestCase):
    def setUp(self):
        self.plan = excerpt.load_plan(ROOT / 'film-excerpt-shot-plan.json')
        self.original = original.load_plan(ROOT / 'film-shot-plan.json')
        self.roots = {f'duck_{i}': [i * .2, -10., .15] for i in range(6)}

    def test_separate_plans_reject_wrong_horizon(self):
        with self.assertRaises(ValueError):
            excerpt.load_plan(ROOT / 'film-shot-plan.json')
        with self.assertRaises(ValueError):
            original.load_plan(ROOT / 'film-excerpt-shot-plan.json')

    def test_three_hundred_frames_five_contiguous_shots(self):
        counts = {}
        for i in range(300):
            row = excerpt.camera_sample(self.plan, i, self.roots)
            counts[row['shot']] = counts.get(row['shot'], 0) + 1
            self.assertEqual(row['shot_frame'], i % 60)
            self.assertFalse(row['actor_extrapolation'])
        self.assertEqual(list(counts.values()), [60] * 5)
        self.assertEqual(self.plan['source_control_indices'],
                         {'start': 0, 'stride': 2, 'count': 300, 'source_control_dt_s': .02})
        self.assertEqual(self.plan['preview_frames'], [29, 89, 149, 209, 269])
        self.assertEqual(self.plan['seconds'], 12)

    def test_camera_geometry_and_focus_preserved_at_shot_endpoints(self):
        for i in range(5):
            for last in (False, True):
                old = original.camera_sample(self.original, i * 100 + (99 if last else 0), self.roots)
                new = excerpt.camera_sample(self.plan, i * 60 + (59 if last else 0), self.roots)
                for key in ('eye_cm', 'target_cm', 'focal_length_mm', 'measured_focus_roots', 'measured_focus_cm'):
                    self.assertEqual(new[key], old[key])
        self.assertEqual(self.plan['shots'][1]['eye_offset_start_cm'], [0, 110, 24])

    def test_out_of_range_and_unmeasured_actor_rejected(self):
        for i in (-1, 300, 499, True):
            with self.assertRaises(ValueError): excerpt.camera_sample(self.plan, i, self.roots)
        missing = dict(self.roots); missing.pop('duck_5')
        with self.assertRaises(ValueError): excerpt.camera_sample(self.plan, 0, missing)

    def test_changed_shot_boundaries_rejected(self):
        for field, value in [('start', 59), ('stop', 121)]:
            plan = copy.deepcopy(self.plan); plan['shots'][1][field] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'plan.json'; path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError): excerpt.load_plan(path)


if __name__ == '__main__': unittest.main()
