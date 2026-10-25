"""CPU interface guards; CUDA startup_coupon verifies the actual actuator chronology."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
SPEC=importlib.util.spec_from_file_location('fresh_env_contract',Path(__file__).with_name('env.py'))
module=importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name]=module
SPEC.loader.exec_module(module)


class EnvironmentContract(unittest.TestCase):
    def fixture(self):
        return [np.full((2,3),1.),np.full((2,3),2.),np.full((2,14),3.),
                np.full((2,14),4.),np.full((2,14),5.),np.array([.3,2.])]

    def test_actor_has_exact61_layout_and_onlyvx_nonzero(self):
        obs=module.compose_actor_numpy(*self.fixture())
        self.assertEqual(obs.shape,(2,61));self.assertEqual(obs.dtype,np.float32)
        np.testing.assert_array_equal(obs[:,34:48],5.)
        np.testing.assert_array_equal(obs[:,20:34],4.)
        np.testing.assert_allclose(obs[:,48],[.3,2.],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(obs[:,49:],0.)

    def test_actual_previous_action_not_replaced_with_joint_state(self):
        fields=self.fixture();fields[4]=np.arange(28).reshape(2,14)
        obs=module.compose_actor_numpy(*fields)
        np.testing.assert_array_equal(obs[:,34:48],fields[4])
        self.assertFalse(np.array_equal(obs[:,34:48],obs[:,6:20]))

    def test_nonfinite_actor_not_sanitized_silently(self):
        fields=self.fixture();fields[0][0,0]=np.nan
        with self.assertRaises(ValueError):module.compose_actor_numpy(*fields)

    def test_lagged_velocity_port_keeps_given_prior_not_current_joint_value(self):
        fields=self.fixture();fields[3][0]=np.linspace(-3,3,14)
        obs=module.compose_actor_numpy(*fields)
        np.testing.assert_allclose(obs[0,20:34],fields[3][0],rtol=0,atol=1e-7)

    def test_named_joints14_exclude_unactuated_wheels(self):
        self.assertEqual(len(module.JOINT_NAMES),14)
        self.assertEqual(len(set(module.JOINT_NAMES)),14)
        self.assertTrue(all('passive' not in name for name in module.JOINT_NAMES))
        self.assertEqual(module.JOINT_NAMES[5:9],('neck_pitch','head_pitch','head_yaw','head_roll'))

    def test_original_stand_defaults_preserved_without_rounding_recipe(self):
        np.testing.assert_allclose(module.DEFAULT_POSE[[1,2,3,4]],[-.08726646259971647,-.457924,-.004940,.452984],atol=0,rtol=0)
        np.testing.assert_allclose(module.DEFAULT_POSE[[10,11,12,13]],-module.DEFAULT_POSE[[1,2,3,4]],atol=0,rtol=0)

    def test_control_has_four_physics_five_drives_and_last_forward(self):
        schedule=module.control_schedule()
        self.assertEqual(schedule.count('physics'),4)
        self.assertEqual(schedule.count('drive'),5)
        self.assertEqual(schedule[-2:],('drive','forward'))


if __name__=='__main__':unittest.main(verbosity=2)
