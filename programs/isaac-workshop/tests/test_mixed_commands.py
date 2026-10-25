import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from isaac_workshop import mixed_commands as mc

HERE=Path(__file__).resolve().parent


class Tensors:
    @staticmethod
    def tensor(value,device=None,dtype=None):return np.array(value,dtype=dtype)


class NativeCommandFixture:
    def __init__(self,cfg,env):
        self._env=env;self.num_envs=env.num_envs;self.device='cpu'
        self.vel_command_b=np.zeros((env.num_envs,3),dtype=np.float32)
        self.is_standing_env=np.ones(env.num_envs,dtype=bool)
        self.is_heading_env=np.ones(env.num_envs,dtype=bool)


class CommandTests(unittest.TestCase):
    def make(self):
        env=SimpleNamespace(num_envs=1,episode_length_buf=np.array([0]))
        return mc.command_type(NativeCommandFixture,Tensors)(None,env),env

    def test_phase_boundaries_and_signs(self):
        for generation,twist in enumerate(mc.TWISTS):
            self.assertEqual(mc.command_at(generation,0),twist)
            self.assertEqual(mc.command_at(generation,199),twist)
            self.assertEqual(mc.command_at(generation,200),(0,0,0))
            self.assertEqual(mc.command_at(generation,300),(0,0,0))
        self.assertEqual(mc.command_at(3,0),mc.TWISTS[0])

    def test_reset_writes_start_before_native_counter_is_cleared(self):
        term,env=self.make()
        term._resample_command([0]);env.episode_length_buf[0]=299
        term._update_command();self.assertTrue(term.is_standing_env[0])
        # Exact pinned order: command manager reset first, episode counter zero later.
        term._resample_command(np.array([0]))
        np.testing.assert_allclose(term.vel_command_b,[[0,0,.3]])
        self.assertFalse(term.is_standing_env[0]);self.assertFalse(term.is_heading_env[0])
        env.episode_length_buf[0]=0;term._update_command()
        np.testing.assert_allclose(term.vel_command_b,[[0,0,.3]])

    def test_early_reset_advances_case_and_empty_reset_does_not(self):
        term,env=self.make();term._resample_command(slice(None))
        env.episode_length_buf[0]=7;term._resample_command([0]);term._resample_command([])
        self.assertEqual(term.episode_generation,1)
        env.episode_length_buf[0]=0;term._update_command()
        np.testing.assert_allclose(term.vel_command_b,[[0,0,.3]])

    def test_stop_and_right_masks(self):
        term,env=self.make()
        for _ in range(3):term._resample_command([0])
        np.testing.assert_allclose(term.vel_command_b,[[0,0,-.3]])
        env.episode_length_buf[0]=200;term._update_command()
        self.assertTrue(term.is_standing_env[0]);self.assertFalse(term.is_heading_env[0])
        np.testing.assert_array_equal(term.vel_command_b,[[0,0,0]])

    def test_invalid_lifecycle_and_worlds_fail(self):
        term,env=self.make()
        with self.assertRaises(ValueError):term._update_command()
        with self.assertRaises(RuntimeError):term._resample_command([1])
        with self.assertRaises(RuntimeError):mc.command_type(NativeCommandFixture,Tensors)(None,SimpleNamespace(num_envs=2))
        for generation,step in ((-1,0),(0,-1),(0,301),(True,1),(0,False)):
            with self.subTest(generation=generation,step=step),self.assertRaises(ValueError):mc.command_at(generation,step)

    def observations(self,value):
        result={'policy':[[0.]*61],'critic':[[0.]*76]}
        result['policy'][0][48:51]=value;result['critic'][0][51:54]=value
        return result

    def test_actual_command_slots_and_no_input_mutation(self):
        obs=self.observations([0,0,.3]);before=copy.deepcopy(obs)
        out=mc.verify_observations(obs,[[0,0,.3]],(0,0,.3))
        self.assertEqual(out['policy'],[0,0,.3]);self.assertEqual(obs,before)

    def test_command_tensor_alone_cannot_pass_missing_yaw_observation(self):
        obs=self.observations([0,0,.3]);obs['policy'][0][50]=0
        with self.assertRaisesRegex(RuntimeError,'policy command'):mc.verify_observations(obs,[[0,0,.3]],(0,0,.3))

    def test_critic_stale_schedule_nonfinite_and_shape_fail(self):
        for mutation in ('critic','schedule','nonfinite','shape'):
            obs=self.observations([0,0,.3]);expected=(0,0,.3)
            if mutation=='critic':obs['critic'][0][53]=0
            if mutation=='schedule':expected=(0,0,-.3)
            if mutation=='nonfinite':obs['policy'][0][0]=float('nan')
            if mutation=='shape':obs['critic'][0].append(0)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):mc.verify_observations(obs,[[0,0,.3]],expected)

    def test_actual_exposure_balanced_only_when_observed(self):
        totals=mc.empty_exposure();contact={'nonsole_or_nonfloor_magnitude_integral_n_s':.02,'maximum_nonsole_or_nonfloor_force_n':3.}
        for generation in range(6):
            for step in range(300):
                exposure={'case':mc.case_at(generation),'generation':generation,'episode_control_before_step':step,'phase':'active' if step<200 else 'stop'}
                mc.record_exposure(totals,exposure,step==299,contact,{'time_out':step==299})
        for row in totals.values():
            self.assertEqual((row['active_controls'],row['stop_controls'],row['completed_timeouts']),(400,200,2))
            self.assertAlmostEqual(row['nonsole_or_nonfloor_magnitude_integral_n_s'],12.)

    def test_partial_failure_is_not_counted_as_complete_balanced_episode(self):
        totals=mc.empty_exposure();contact={'nonsole_or_nonfloor_magnitude_integral_n_s':0.,'maximum_nonsole_or_nonfloor_force_n':0.}
        for step in range(7):
            mc.record_exposure(totals,{'case':'walk','generation':0,'episode_control_before_step':step,'phase':'active'},step==6,contact,{'fell_over':step==6})
        self.assertEqual(totals['walk']['active_controls'],7)
        self.assertEqual(totals['walk']['early_terminal_episodes'],1)
        self.assertEqual(totals['walk']['completed_timeouts'],0)
        self.assertEqual(totals['left_turn']['episode_generations'],[])

    def test_behavior_measures_yaw_velocity_and_stop_from_actual_state(self):
        episodes={};mc.begin_behavior(episodes,1,[0,0,.15],3.13)
        exp={'generation':1,'phase':'active','episode_control_before_step':0}
        mc.record_behavior(episodes,exp,[0,0,.15],-3.13,.02,[0,0,0],[0,0,.3],{})
        self.assertAlmostEqual(episodes['1']['yaw_change_rad'],2*np.pi-6.26)
        self.assertEqual(episodes['1']['active_yaw_tracking_rms_rad_s'],0)
        exp.update(phase='stop',episode_control_before_step=200)
        mc.record_behavior(episodes,exp,[.01,0,.15],-3.13,.02,[.02,0,0],[0,0,0],{'fell_over':True})
        self.assertAlmostEqual(episodes['1']['stop_displacement_m'],.01)
        self.assertAlmostEqual(episodes['1']['stop_speed_rms_m_s'],.02)
        self.assertTrue(episodes['1']['ended']);self.assertFalse(episodes['1']['behavior_qualified'])
        with self.assertRaises(RuntimeError):mc.record_behavior(episodes,exp,[0,0,.15],0,0,[0]*3,[0]*3,{})




if __name__=='__main__':unittest.main()
