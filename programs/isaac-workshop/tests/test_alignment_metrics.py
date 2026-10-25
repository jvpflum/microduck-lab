import unittest
import numpy as np
from isaac_workshop.alignment_metrics import *

class Tests(unittest.TestCase):
 def setUp(self):
  self.plane=np.array([[[-10,-10,2],[10,-10,2],[10,10,2]],[[-10,-10,2],[10,10,2],[-10,10,2]]],float)
 def test_optical_depth_not_radial(self):
  rays=camera_rays(np.eye(3),np.eye(3),[[0,0],[1,0]],0.)
  np.testing.assert_allclose(optical_depth_to_triangles([0,0,0],rays,self.plane),[2,2])
  self.assertAlmostEqual(np.linalg.norm(rays[1]*2),math.sqrt(8))
 def test_current_vs_previous_pose_has_distinct_residual(self):
  rays=np.array([[0,0,1.]])
  actual=optical_depth_to_triangles([0,0,.1],rays,self.plane)
  self.assertEqual(depth_residual(actual,actual)['absolute_error_median_m'],0)
  self.assertAlmostEqual(depth_residual(actual,optical_depth_to_triangles([0,0,0],rays,self.plane))['absolute_error_median_m'],.1)
 def test_pixel_conventions_explicit(self):
  a=camera_rays(np.eye(3),np.eye(3),[[0,0]],0.);b=camera_rays(np.eye(3),np.eye(3),[[0,0]],.5)
  np.testing.assert_array_equal(a,[[0,0,1]]);np.testing.assert_array_equal(b,[[.5,.5,1]])
  with self.assertRaises(ValueError):camera_rays(np.eye(3),np.eye(3),[[0,0]],.25)
 def test_bounds_and_no_false_hit(self):
  r=optical_depth_to_triangles([0,0,0],[[0,0,-1],[0,0,1]],self.plane,far=1)
  self.assertTrue(np.isinf(r).all())
 def test_rotation_no_scale_or_translation(self):
  old=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
  new=tilted_camera(old,15);np.testing.assert_allclose(new.T@new,np.eye(3),atol=1e-15)
  self.assertAlmostEqual(np.linalg.det(new),1);self.assertLess(new[2,2],0)
 def test_bad_rays_fail(self):
  for r in (np.eye(3)*2,np.diag([1,1,-1])):
   with self.assertRaises(ValueError):camera_rays(np.eye(3),r,[[0,0]],0.)
 def test_floor_coverage_preserves_unknown(self):
  r=floor_reference_coverage([0,0,.3],[[0,0,-1],[0,0,.1],[0,0,-.01]],0)
  self.assertEqual(r['floor_reference_in_range_fraction'],1/3);self.assertEqual(r['upward_rays'],1);self.assertEqual(r['downward_beyond_far'],1)
 def test_missing_is_not_zero_error(self):
  r=depth_residual([np.inf,1],[1,np.inf]);self.assertIsNone(r['absolute_error_median_m']);self.assertEqual(r['both_finite'],0)
if __name__=='__main__':unittest.main()
