"""Synthetic CPU fixtures only: no simulation, assets, policies or native film."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from isaac_workshop import film_cameras as cameras
from isaac_workshop.film_motion import body_matrix, command_at, previous_action, stage_world_matrix


PLAN = Path(__file__).resolve().parents[1] / 'film-shot-plan.json'


class FilmMotionTests(unittest.TestCase):
    def test_schedule_all_controls_and_boundaries(self):
        commands = [command_at(i) for i in range(1000)]
        self.assertEqual(sum(v == (.15, 0., 0.) for v in commands), 600)
        for i, expected in [(0, .15), (399, .15), (400, 0.), (599, 0.),
                            (600, .15), (799, .15), (800, 0.), (1000, 0.)]:
            self.assertEqual(command_at(i), (expected, 0., 0.))

    def test_schedule_invalid_index(self):
        for value in [-1, 1001, 0.5, True]:
            with self.assertRaises(ValueError):
                command_at(value)

    def test_per_actor_history_never_uses_global_action(self):
        class Array:
            def __init__(self, shape): self.shape = shape
        values = {f'joint_pos_{i}': Array((1, 14)) for i in range(6)}
        class Manager:
            action = Array((1, 84))
            def get_term(self, name):
                return type('Term', (), {'raw_actions': values[name]})()
        env = type('Env', (), {'action_manager': Manager()})()
        for name, value in values.items():
            self.assertIs(previous_action(env, name), value)
        values['joint_pos_0'] = Array((1, 84))
        with self.assertRaises(ValueError): previous_action(env, 'joint_pos_0')

    def test_body_matrix_rotation_translation(self):
        q = [1., 2., 3., 0., 0., math.sqrt(.5), math.sqrt(.5)]
        matrix = body_matrix(q)
        self.assertAlmostEqual(matrix[0][0], 0.)
        self.assertAlmostEqual(matrix[0][1], -1.)
        self.assertAlmostEqual(matrix[1][0], 1.)
        self.assertEqual([matrix[i][3] for i in range(3)], [1., 2., 3.])

    def test_invalid_pose_rejected(self):
        for q in ([0.] * 6, [0.] * 7, [0., 0., 0., 0., 0., math.nan, 1.]):
            with self.assertRaises(ValueError): body_matrix(q)

    def test_basis_and_translation_scaled(self):
        m = stage_world_matrix([1., 2., 3., 0., 0., 0., 1.], .01)
        self.assertEqual(m, [100., 0., 0., 0., 0., 100., 0., 0.,
                             0., 0., 100., 0., 100., 200., 300., 1.])

    def test_rotated_local_vertex_in_stage_units(self):
        q = [1., 2., 3., 0., 0., math.sqrt(.5), math.sqrt(.5)]
        flat = stage_world_matrix(q, .01)
        rows = [flat[i:i + 4] for i in range(0, 16, 4)]
        vertex = [.1, 0., 0., 1.]
        world = [sum(vertex[k] * rows[k][j] for k in range(4)) for j in range(4)]
        for actual, expected in zip(world, [100., 210., 300., 1.]):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_stage_units(self):
        for unit in [0., -1., math.inf, math.nan, True, 'cm']:
            with self.assertRaises(ValueError):
                stage_world_matrix([0., 0., 0., 0., 0., 0., 1.], unit)


class FilmCameraTests(unittest.TestCase):
    def setUp(self):
        self.plan = cameras.load_plan(PLAN)
        self.roots = {f'duck_{i}': [i * .2, -10., .15] for i in range(6)}

    def test_all_frames_use_measured_roots(self):
        counts = {}
        for frame in range(500):
            sample = cameras.camera_sample(self.plan, frame, self.roots)
            counts[sample['shot']] = counts.get(sample['shot'], 0) + 1
            self.assertFalse(sample['actor_extrapolation'])
        self.assertEqual(list(counts.values()), [100] * 5)

    def test_camera_follows_root_translation_only(self):
        initial = cameras.camera_sample(self.plan, 149, self.roots)
        shifted = {k: [v[0] + .25, v[1], v[2]] for k, v in self.roots.items()}
        actual = cameras.camera_sample(self.plan, 149, shifted)
        delta = .25 / .009999999776482582
        self.assertAlmostEqual(actual['eye_cm'][0] - initial['eye_cm'][0], delta)
        self.assertAlmostEqual(actual['target_cm'][0] - initial['target_cm'][0], delta)
        self.assertEqual(actual['focal_length_mm'], initial['focal_length_mm'])

    def test_missing_or_nonfinite_roots_rejected(self):
        invalid = [dict(list(self.roots.items())[:-1]), copy.deepcopy(self.roots)]
        invalid[1]['duck_0'][0] = math.nan
        for roots in invalid:
            with self.assertRaises(ValueError): cameras.camera_sample(self.plan, 0, roots)

    def test_out_of_range_frame_rejected(self):
        for frame in [-1, 500, True]:
            with self.assertRaises(ValueError): cameras.camera_sample(self.plan, frame, self.roots)

    def test_shot_gap_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['shots'][1]['start'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'plan.json'
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError): cameras.load_plan(path)


if __name__ == '__main__':
    unittest.main()
