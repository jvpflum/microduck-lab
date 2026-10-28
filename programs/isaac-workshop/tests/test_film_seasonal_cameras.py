"""Synthetic camera math tests; actor poses remain input-only."""
import copy
from pathlib import Path
import unittest
from isaac_workshop import film_seasonal_cameras as c
class CameraTests(unittest.TestCase):
 def setUp(self):
  self.plan=c.load_plan(Path(__file__).resolve().parents[1]/'film-seasonal-shot-plan.json')
  self.roots={f'duck_{i}':[i*.2,-10,.15] for i in range(6)}
 def test_full_partition_and_input_immutable(self):
  original=copy.deepcopy(self.roots);samples=[c.camera_sample(self.plan,i,self.roots) for i in range(300)]
  self.assertEqual(len({x['shot'] for x in samples}),5);self.assertEqual(self.roots,original)
  self.assertTrue(all(x['actor_extrapolation'] is False for x in samples))
  for i in (-1,300,True):
   with self.assertRaises(ValueError):c.camera_sample(self.plan,i,self.roots)
 def test_tight_shot_z_is_camera_only(self):
  for frame in (89,209):
   a=c.camera_sample(self.plan,frame,self.roots);moved=copy.deepcopy(self.roots);moved['duck_0'][2]+=.05;b=c.camera_sample(self.plan,frame,moved)
   self.assertEqual(a['eye_cm'],b['eye_cm']);self.assertEqual(a['target_cm'],b['target_cm']);self.assertNotEqual(a['measured_focus_cm'],b['measured_focus_cm'])
 def test_wide_room_view_fixed_world(self):
  a=c.camera_sample(self.plan,269,self.roots);moved={n:[v[0]+1,v[1]+1,v[2]+1] for n,v in self.roots.items()};b=c.camera_sample(self.plan,269,moved)
  self.assertEqual(a['eye_cm'],b['eye_cm']);self.assertEqual(a['target_cm'],b['target_cm']);self.assertEqual(a['focal_length_mm'],21)
if __name__=='__main__':unittest.main()
