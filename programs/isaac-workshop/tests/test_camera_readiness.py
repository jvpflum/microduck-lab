import copy
import unittest
import numpy as np
from isaac_workshop import camera_readiness as cr


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.contract = cr.Contract()
        self.valid = np.full((48,64), .5, dtype=np.float32)

    def run_frames(self, frames, state=None, mutate=None, timestamps=None, indices=None):
        receipt = {}; state = {'physics':0,'observation':'same'} if state is None else state
        def capture(i):
            if mutate: mutate(state,i)
            return frames[min(i-1,len(frames)-1)], {
                'sensor_frame':indices[i-1] if indices else i+4,
                'sensor_timestamp_s':timestamps[i-1] if timestamps else 0.}
        cr.bounded_warmup(lambda:copy.deepcopy(state),capture,receipt,self.contract)
        return receipt

    def test_two_unknown_startup_frames_then_actual_two_ready(self):
        result=self.run_frames([np.zeros_like(self.valid),np.zeros_like(self.valid),self.valid,self.valid])
        self.assertEqual(result['render_pumps'],4)
        self.assertEqual([r['ready']for r in result['attempts']],[False,False,True,True])
        self.assertTrue(result['frozen_state_verified']); self.assertFalse(result['unknown_depth_forward_authorized'])

    def test_entire_bounded_budget_unknown_fails_before_action(self):
        calls=[];receipt={}
        def capture(i):calls.append(i);return np.zeros_like(self.valid),{'sensor_frame':i,'sensor_timestamp_s':0.}
        with self.assertRaisesRegex(RuntimeError,'no action authorized'):
            cr.bounded_warmup(lambda:{'physics':0},capture,receipt,self.contract)
        self.assertEqual(len(calls),32);self.assertFalse(receipt['ready'])

    def test_ready_streak_resets_after_unknown(self):
        r=self.run_frames([self.valid,np.zeros_like(self.valid),self.valid,self.valid])
        self.assertEqual([x['consecutive_ready']for x in r['attempts']],[1,0,1,2])

    def test_near_and_unknown_frontal_never_authorize_forward(self):
        for bad in (0.,.02,np.nan,-np.inf):
            raw=self.valid.copy();raw[20,30]=bad
            self.assertFalse(cr.readiness(raw,self.contract)['ready'])
        raw=self.valid.copy();raw[12:39,12:52]=np.inf
        self.assertFalse(cr.readiness(raw,self.contract)['ready'])
        self.assertTrue(cr.readiness(self.valid,self.contract)['ready'])

    def test_physics_or_observation_drift_fails(self):
        for name in ('physics','observation'):
            with self.assertRaisesRegex(RuntimeError,'changed physics'):
                self.run_frames([self.valid],mutate=lambda state,i:state.update({name:i}))

    def test_repeated_capture_or_clock_advance_fails(self):
        for params in ({'indices':[5,5]},{'timestamps':[0.,.02]}):
            with self.assertRaisesRegex(RuntimeError,'zero-time stamp differs'):
                self.run_frames([self.valid],**params)

    def test_initial_zero_time_sample_need_not_be_frame_one(self):
        self.assertTrue(self.run_frames([self.valid],indices=[13,14])['ready'])
