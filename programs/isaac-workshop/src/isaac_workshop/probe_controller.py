"""Conservative live-depth stop probe. No learned navigation is claimed.

The only geometric input is the measured camera-to-body calibration. A local
horizontal floor estimate comes from the depth cloud itself. No source obstacle
positions, collider bounds, global map or planned bypass enter this controller.
"""
from __future__ import annotations
import numpy as np


def depth_stop_evidence(depth_m, intrinsics, camera_to_body, *, near_m=.03, far_m=5.):
    depth=np.asarray(depth_m,dtype=np.float64);k=np.asarray(intrinsics,dtype=np.float64)
    transform=np.asarray(camera_to_body,dtype=np.float64)
    if (depth.ndim!=2 or k.shape!=(3,3) or transform.shape!=(4,4)
            or not np.isfinite(k).all() or not np.isfinite(transform).all()
            or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(transform[3],[0,0,0,1],atol=1e-6)
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5)):
        raise ValueError('Invalid camera geometry')
    valid=np.isfinite(depth)&(depth>=near_m)&(depth<far_m)
    if np.isnan(depth).any() or np.isneginf(depth).any() or (depth<near_m).any():
        return {'stop':True,'reason':'invalid_or_too_close_depth','floor_estimate_m':None,'hazard_pixels':0}
    yy,xx=np.indices(depth.shape)
    rays=np.stack(((xx+.5-k[0,2])/k[0,0],(yy+.5-k[1,2])/k[1,1],np.ones_like(xx)),axis=-1)
    points=(rays*np.where(valid,depth,0)[...,None])@transform[:3,:3].T+transform[:3,3]
    # A yaw-aligned, gravity-up body frame is supplied. Fit only a lower depth
    # quantile, not source floor truth; warped planks can cause conservative stops.
    floor_candidates=valid&(points[...,0]>.10)&(points[...,0]<1.2)&(np.abs(points[...,1])<.4)
    if int(floor_candidates.sum())<24:
        return {'stop':True,'reason':'floor_not_observed','floor_estimate_m':None,'hazard_pixels':0}
    floor=float(np.quantile(points[...,2][floor_candidates],.10))
    hazard=valid&(points[...,0]>.05)&(points[...,0]<.35)&(np.abs(points[...,1])<.17)&(points[...,2]>floor+.025)&(points[...,2]<floor+.40)
    count=int(hazard.sum())
    return {'stop':count>=3,'reason':'measured_depth_obstacle' if count>=3 else 'depth_corridor_clear',
            'floor_estimate_m':floor,'floor_candidate_pixels':int(floor_candidates.sum()),
            'hazard_pixels':count,'minimum_hazard_forward_m':float(points[...,0][hazard].min()) if count else None,
            'parameters':{'forward_m':[.05,.35],'half_width_m':.17,'height_above_depth_floor_m':[.025,.40],
                          'minimum_hazard_pixels':3,'floor_quantile':.10,'pixel_center_offset':.5},
            'limitations':'Depth-derived horizontal plane and fixed envelope; not a general collision-safety proof.'}


class DepthStopProbe:
    def __init__(self):
        self.intrinsics=None;self.camera_to_body=None;self.stopped=False;self.last=None
    def set_camera_geometry(self,intrinsics,camera_to_body):
        self.intrinsics=np.asarray(intrinsics).copy();self.camera_to_body=np.asarray(camera_to_body).copy()
    def __call__(self,obs):
        image=obs['head_depth'];depth=image[0]*(5.-.03)+.03
        depth=np.where(image[1]>.5,depth,np.inf)
        evidence=depth_stop_evidence(depth,self.intrinsics,self.camera_to_body)
        if evidence['stop']:self.stopped=True
        distance=float(np.linalg.norm(obs['goal_body_m']))
        if distance<=.1:self.stopped=True;evidence['goal_radius_entered']=True
        self.last=evidence
        # No steering/avoidance implied: the first trial isolates camera-caused
        # stopping over the unchanged forward-walking policy.
        return [0. if self.stopped else 1.,0.]
