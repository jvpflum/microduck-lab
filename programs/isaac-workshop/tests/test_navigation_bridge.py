import math
import unittest
import numpy as np
from isaac_workshop.navigation_bridge import CameraPacket, Contract, NavigationBridge, bounded_command, camera_counterfactual, goal_in_body, image_channels


class Tests(unittest.TestCase):
    def setUp(self):
        self.contract=Contract(height=4,width=6)
        self.bridge=NavigationBridge(self.contract);self.bridge.begin_episode(0)
        self.calls=[]
    def packet(self,step=0,**changes):
        values=dict(episode_id=0,control_step=step,sensor_frame=step+1,sensor_timestamp_s=.02*step,simulation_time_s=.02*step,depth_m=np.ones((4,6),np.float32))
        values.update(changes);return CameraPacket(**values)
    def predict(self,obs):
        self.calls.append(obs);return [1.,.5]
    def advance(self,packet,**changes):
        args=dict(locomotor_observation=np.zeros(61),goal_body_m=(1.,0.),predict=self.predict,goal_provenance='simulator_pose_scaffold');args.update(changes)
        return self.bridge.advance(packet,**args)
    def test_actual_clock_drives_ten_hz_commands_on_fifty_hz_stream(self):
        rows=[self.advance(self.packet(i))[1] for i in range(11)]
        self.assertEqual([r['control_step'] for r in rows if r['decision']],[0,5,10]);self.assertEqual(len(self.calls),3)
        self.assertEqual(rows[-1]['command_body_twist'],[.15,0.,.2])
    def test_repeated_frame_stops_on_nondecision_interval(self):
        self.advance(self.packet());command,row=self.advance(self.packet(1,sensor_frame=1))
        self.assertEqual(command,(0.,0.,0.));self.assertTrue(row['latched_stop']);self.assertEqual(len(self.calls),1)
    def test_failure_is_latched_and_never_hidden_by_later_good_frame(self):
        self.advance(self.packet(depth_m=np.full((4,6),np.inf,np.float32)))
        self.assertTrue(self.advance(self.packet(1))[1]['latched_stop']);self.assertEqual(len(self.calls),0)
        self.bridge.begin_episode(1)
        self.assertIsNone(self.advance(self.packet(0,episode_id=1))[1]['fault'])
    def test_no_episode_splice_or_implicit_reset(self):
        self.advance(self.packet());self.assertTrue(self.advance(self.packet(1,episode_id=1))[1]['latched_stop'])
        with self.assertRaises(ValueError):self.bridge.begin_episode(0)
    def test_stale_future_nonfinite_and_skipped_clocks_reject(self):
        for changes in ({'simulation_time_s':.2},{'sensor_timestamp_s':.1},{'sensor_timestamp_s':math.nan}):
            self.setUp();self.assertTrue(self.advance(self.packet(**changes))[1]['latched_stop'])
        self.setUp();self.advance(self.packet());self.assertTrue(self.advance(self.packet(2))[1]['latched_stop'])
    def test_raw_metric_depth_preserved_and_misses_explicit(self):
        depth=np.ones((4,6),np.float32);depth[0,:3]=[np.inf,.03,5.]
        before=depth.copy();image=image_channels(self.packet(depth_m=depth),self.contract)
        np.testing.assert_array_equal(depth,before);np.testing.assert_array_equal(image[:,0,:3],[[1,0,1],[0,1,0]])
    def test_nan_negative_depth_never_becomes_a_safe_background(self):
        for bad in (math.nan,-math.inf,-.1):
            depth=np.ones((4,6),np.float32);depth[0,0]=bad
            with self.assertRaises(ValueError):image_channels(self.packet(depth_m=depth),self.contract)
    def test_close_and_unknown_frontal_depth_cannot_increase_forward_command(self):
        for bad in (0.,.01,math.inf,5.):
            self.setUp();before,_=self.advance(self.packet())
            raw=np.ones((4,6),np.float32);raw[1:4,1:5]=bad
            after,row=self.advance(self.packet(1,depth_m=raw))
            self.assertGreater(before[0],0);self.assertEqual(after,(0.,0.,0.))
            self.assertTrue(row['latched_stop']);self.assertEqual(len(self.calls),1)
    def test_actor_cannot_receive_map_or_mutate_original_state(self):
        state=np.arange(61,dtype=np.float32);original=state.copy();self.advance(self.packet(),locomotor_observation=state)
        self.assertEqual(set(self.calls[0]),{'policy','head_depth','goal_body_m'});np.testing.assert_array_equal(state,original)
        self.assertFalse(self.calls[0]['policy'].flags.writeable)
    def test_bad_policy_output_fails_without_silent_clipping(self):
        for value in ([2,0],[0,2],[-1,0],[0,math.nan],[1,0,0]):
            with self.assertRaises(ValueError):bounded_command(value,self.contract)
    def test_destination_coordinates_and_provenance(self):
        np.testing.assert_allclose(goal_in_body([0,0],math.pi/2,[0,1]),[1,0],atol=1e-6)
        self.assertTrue(self.advance(self.packet(),goal_provenance='unlabelled_truth')[1]['latched_stop'])
    def test_counterfactual_detects_blind_policy_without_promoting_camera_claim(self):
        actual={'head_depth':np.zeros((2,4,6)), 'policy':np.zeros(61), 'goal_body_m':np.array([1.,0.])}
        report=camera_counterfactual(lambda x:[.5,0],actual,np.ones((2,4,6)))
        self.assertTrue(report['same_state_and_goal']);self.assertTrue(report['different_depth'])
        self.assertEqual(report['maximum_command_fraction_difference'],0);self.assertFalse(report['task_success_proven'])
    def test_counterfactual_changes_only_image_and_detects_response(self):
        actual={'head_depth':np.zeros((2,4,6)), 'policy':np.zeros(61), 'goal_body_m':np.array([1.,0.])}
        report=camera_counterfactual(lambda x:[float(x['head_depth'][0].mean()),0],actual,np.ones((2,4,6)))
        self.assertEqual(report['maximum_command_fraction_difference'],1);self.assertTrue(report['same_state_and_goal'])


if __name__=='__main__':unittest.main()
