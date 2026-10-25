import ast
import hashlib
import math
from pathlib import Path
import unittest
import numpy as np
from isaac_workshop.navigation_bridge import CameraPacket, Contract, NavigationBridge
from isaac_workshop.probe_controller import DepthStopProbe, depth_stop_evidence

HERE=Path(__file__).resolve().parent


def fixture(block=False):
    h,w=48,64;f=w/(2*math.tan(math.radians(87)/2));k=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
    c,s=math.cos(math.radians(20)),math.sin(math.radians(20))
    pose=np.eye(4);pose[:3,:3]=np.array([[0,-s,c],[-1,0,0],[0,-c,-s]]);pose[:3,3]=[.05,0,.10]
    yy,xx=np.indices((h,w));rays=np.stack(((xx+.5-w/2)/f,(yy+.5-h/2)/f,np.ones_like(xx)),axis=-1)@pose[:3,:3].T
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=(-.125-pose[2,3])/rays[...,2]
        depth=np.where((depth>=.03)&(depth<5),depth,np.inf)
        front=(.30-pose[0,3])/rays[...,0]
        points=front[...,None]*rays+pose[:3,3]
    hit=(front>=.03)&(front<depth)&(np.abs(points[...,1])<.044)&(points[...,2]>=-.125)&(points[...,2]<=-.060)
    if block:depth=np.where(hit,front,depth)
    return depth.astype(np.float32),k,pose


class ProbeTests(unittest.TestCase):
    def test_depth_plane_is_clear_and_actual_camera_block_stops(self):
        clear,k,pose=fixture();blocked,_,_=fixture(True)
        self.assertFalse(depth_stop_evidence(clear,k,pose)['stop'])
        result=depth_stop_evidence(blocked,k,pose)
        self.assertTrue(result['stop']);self.assertEqual(result['reason'],'measured_depth_obstacle')
        self.assertGreaterEqual(result['hazard_pixels'],3)
    def test_same_state_goal_changed_depth_changes_command(self):
        commands=[]
        for blocked in (False,True):
            depth,k,pose=fixture(blocked);policy=DepthStopProbe();policy.set_camera_geometry(k,pose)
            bridge=NavigationBridge();bridge.begin_episode(0)
            commands.append(bridge.advance(CameraPacket(0,0,1,0.,0.,depth),np.zeros(61),[.9,0],policy,goal_provenance='simulator_pose_scaffold')[0])
        self.assertEqual(commands,[(.15,0.,0.),(0.,0.,0.)])
    def test_unknown_front_and_too_near_never_authorize_forward(self):
        for value in (np.inf,0.,.02):
            depth,k,pose=fixture();depth[12:39,12:52]=value
            policy=DepthStopProbe();policy.set_camera_geometry(k,pose)
            bridge=NavigationBridge();bridge.begin_episode(0)
            command,row=bridge.advance(CameraPacket(0,0,1,0.,0.,depth),np.zeros(61),[.9,0],policy,goal_provenance='simulator_pose_scaffold')
            self.assertEqual(command,(0.,0.,0.));self.assertTrue(row['latched_stop'])
    def test_floor_unknown_fails_conservatively(self):
        depth,k,pose=fixture();result=depth_stop_evidence(np.full_like(depth,np.inf),k,pose)
        self.assertTrue(result['stop']);self.assertEqual(result['reason'],'floor_not_observed')
    def test_supported_goal_stops_but_no_success_inferred(self):
        depth,k,pose=fixture();policy=DepthStopProbe();policy.set_camera_geometry(k,pose)
        image=np.stack((np.where(np.isfinite(depth),(depth-.03)/4.97,1),np.isfinite(depth)))
        self.assertEqual(policy({'head_depth':image,'policy':np.zeros(61),'goal_body_m':np.array([.08,0])}),[0.,0.])


if __name__=='__main__':unittest.main()
