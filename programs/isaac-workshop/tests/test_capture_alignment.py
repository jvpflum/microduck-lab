import copy
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from isaac_workshop import capture_alignment as ca

HERE=Path(__file__).resolve().parent


class AlignmentTests(unittest.TestCase):
    def test_transaction_prepares_before_pump_then_reads_same_pose(self):
        calls=[];pose={'xyz':[1,2,3]};row={}
        def prepare():calls.append('prepare');return copy.deepcopy(pose)
        def pump():calls.append('pump');return True
        def read():calls.append('read');return {'pose':copy.deepcopy(pose)}
        ca.capture_transaction(lambda:{'clock':.2},prepare,pump,read,row)
        self.assertEqual(calls,['prepare','pump','prepare','read'])
        self.assertTrue(row['frozen_state_verified']);self.assertEqual(row['explicit_pumps'],1)

    def test_physics_drift_or_camera_pose_drift_rejects_before_read(self):
        for field in ('physics','pose'):
            state={'physics':0,'pose':0};reads=[]
            def pump():state[field]+=1;return True
            with self.assertRaisesRegex(RuntimeError,'drift'):
                ca.capture_transaction(lambda:{'physics':state['physics']},lambda:{'pose':state['pose']},pump,lambda:reads.append(1),{})
            self.assertFalse(reads)

    def test_read_pose_drift_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Annotator read'):
            ca.capture_transaction(lambda:{'clock':0},lambda:{'x':0},lambda:True,lambda:{'pose':{'x':1}}, {})



    def test_geometry_diagnostic_distinguishes_prior_pose_without_relabel(self):
        triangles=np.array([[[-100,-100,2],[100,-100,2],[100,100,2]],[[-100,-100,2],[100,100,2],[-100,100,2]]])
        pose={'position_world_m':[0,0,.1],'quaternion_ros_xyzw':[0,0,0,1]}
        previous={'position_world_m':[0,0,0],'quaternion_ros_xyzw':[0,0,0,1]}
        pre={'depth':np.full((48,64),2.),'intrinsics':np.eye(3),'pose':pose}
        post={'depth':np.full((48,64),1.9),'intrinsics':np.eye(3),'pose':pose}
        rows=ca.residual_record(pre,post,previous,triangles)
        scores={(r['image'],r['pose_hypothesis']):r['absolute_error_median_m']for r in rows}
        self.assertAlmostEqual(scores['pre','current_pump_pose'],.1)
        self.assertAlmostEqual(scores['pre','previous_control_pump_pose'],0.)
        self.assertAlmostEqual(scores['post','current_pump_pose'],0.)
        np.testing.assert_array_equal(pre['depth'],np.full((48,64),2.))

    def fixture(self,root,pump_effect=None,wrong_stamp=False):
        class Timeline:
            auto=True
            def is_auto_updating(self):return self.auto
            def set_auto_update(self,value):self.auto=value
            def commit(self):pass
        class Settings:
            def __init__(self):self.value=True
            def get(self,key):return self.value
            def set(self,key,value):self.value=value
        obj=ca.SameStateCapture.__new__(ca.SameStateCapture)
        obj.timeline=Timeline();obj.settings=Settings();obj.output=root;obj.frames=root/'frames';obj.frames.mkdir()
        obj.stream=gzip.open(root/'capture-alignment.jsonl.gz','xt',encoding='utf-8');obj.rows=[];obj.receipt={'blocked_solver_call_attempts':0};obj.closed=False
        obj.base=types.SimpleNamespace(device='cpu');obj.wp=types.SimpleNamespace(synchronize_device=lambda device:None)
        obj.manager=types.SimpleNamespace(step=lambda:None,_run_solver_substeps=lambda:None)
        state={'physics':0,'obs':'unchanged'};frame=[3]
        obj.snapshot=lambda:copy.deepcopy(state)
        pose={'position_world_m':[0,0,0],'quaternion_ros_xyzw':[0,0,0,1],'intrinsics':np.eye(3).tolist()}
        obj.pose=lambda:copy.deepcopy(pose)
        obj.camera=types.SimpleNamespace(_renderer=types.SimpleNamespace(update_transforms=lambda:None,update_geometries=lambda:None),
            _update_poses=lambda frame_op:None,_invalidate_buffers=lambda:None,
            update=lambda dt,force_recompute:frame.__setitem__(0,frame[0]+1))
        def sample():return {'depth':np.full((48,64),.5,np.float32),'rgb':np.zeros((48,64,3),np.uint8),
            'intrinsics':np.eye(3),'pose':copy.deepcopy(pose),'sensor_frame':frame[0],
            'sensor_timestamp_s':.02 if wrong_stamp and frame[0]>3 else 0.}
        obj.sample=sample
        def pump():
            if pump_effect:pump_effect(obj,state)
            return True
        obj.visualizer=types.SimpleNamespace(_pump_kit_app_update=pump)
        return obj,state

    def test_actual_capture_method_restores_hooks_and_retains_both_raw_images(self):
        with tempfile.TemporaryDirectory()as tmp:
            obj,state=self.fixture(Path(tmp));step=obj.manager.step;solver=obj.manager._run_solver_substeps
            try:
                obj.capture(0)
                self.assertIs(obj.manager.step,step);self.assertIs(obj.manager._run_solver_substeps,solver)
                self.assertTrue(obj.timeline.auto);self.assertIs(obj.settings.value,True)
                self.assertTrue(obj.rows[0]['successful']);self.assertTrue(obj.rows[0]['restored'])
                with np.load(obj.output/obj.rows[0]['raw_pair_file'],allow_pickle=False)as data:
                    self.assertEqual(data['pre_sensor_frame'],3);self.assertEqual(data['post_sensor_frame'],4)
                    self.assertEqual(data['pre_sensor_timestamp_s'],data['post_sensor_timestamp_s'])
                self.assertEqual(state,{'physics':0,'obs':'unchanged'})
                obj.stream.close()
                with gzip.open(obj.output/'capture-alignment.jsonl.gz','rt') as stream:rows=[json.loads(x)for x in stream]
                self.assertEqual(rows,obj.rows)
            finally:obj.stream.close()

    def test_physics_call_rejected_settings_restored_and_failure_retained(self):
        with tempfile.TemporaryDirectory()as tmp:
            obj,state=self.fixture(Path(tmp),lambda obj,state:obj.manager.step());step=obj.manager.step
            try:
                with self.assertRaisesRegex(RuntimeError,'Solver call prohibited'):obj.capture(0)
                self.assertIs(obj.manager.step,step);self.assertTrue(obj.timeline.auto);self.assertIs(obj.settings.value,True)
                self.assertEqual(obj.receipt['blocked_solver_call_attempts'],1);self.assertFalse(obj.rows[0]['successful'])
                self.assertTrue((obj.output/obj.rows[0]['raw_pair_file']).is_file())
            finally:obj.stream.close()

    def test_sampling_clock_drift_fails_without_hiding_raw_pair(self):
        with tempfile.TemporaryDirectory()as tmp:
            obj,state=self.fixture(Path(tmp),wrong_stamp=True)
            try:
                with self.assertRaisesRegex(RuntimeError,'sampling clock'):obj.capture(0)
                self.assertFalse(obj.rows[0]['successful']);self.assertTrue(obj.rows[0]['restored'])
                self.assertTrue((obj.output/obj.rows[0]['raw_pair_file']).exists())
            finally:obj.stream.close()


if __name__=='__main__':unittest.main()
