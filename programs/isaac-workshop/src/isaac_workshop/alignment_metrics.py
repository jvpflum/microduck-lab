"""Source-geometry depth diagnostics only; never supplies privileged rays to policy."""
import math
import numpy as np


def rotation_xyzw(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,)or not np.isfinite(q).all()or abs(float(q@q)-1)>1e-5:raise ValueError('Unit finite quaternion required')
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def camera_rays(intrinsics,rotation,pixels_xy,pixel_offset):
    k=np.asarray(intrinsics,float);r=np.asarray(rotation,float);p=np.asarray(pixels_xy,float)
    if k.shape!=(3,3)or r.shape!=(3,3)or p.ndim!=2 or p.shape[1]!=2:raise ValueError('Camera dimensions')
    if not all(np.isfinite(v).all()for v in (k,r,p))or k[0,0]<=0 or k[1,1]<=0:raise ValueError('Nonfinite camera/intrinsics')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5)or not math.isclose(np.linalg.det(r),1,abs_tol=1e-5):raise ValueError('Proper rotation required')
    if pixel_offset not in(0.,.5):raise ValueError('Explicit pixel convention required')
    camera=np.column_stack(((p[:,0]+pixel_offset-k[0,2])/k[0,0],(p[:,1]+pixel_offset-k[1,2])/k[1,1],np.ones(len(p))))
    # Keep optical z=1: ray parameter is distance_to_image_plane, NOT radial metres.
    return camera@r.T


def optical_depth_to_triangles(origin,rays,triangles,near=.03,far=5.):
    o=np.asarray(origin,float);d=np.asarray(rays,float);t=np.asarray(triangles,float)
    if o.shape!=(3,)or d.ndim!=2 or d.shape[1]!=3 or t.ndim!=3 or t.shape[1:]!=(3,3):raise ValueError('Ray/triangle dimensions')
    if not all(np.isfinite(a).all()for a in(o,d,t))or not 0<near<far:raise ValueError('Finite ray data required')
    result=np.full(len(d),np.inf);e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];v=o-t[:,0]
    for i,ray in enumerate(d):
        p=np.cross(ray,e2);det=np.einsum('ij,ij->i',e1,p);nonparallel=np.abs(det)>1e-12
        inv=np.divide(1.,det,out=np.zeros_like(det),where=nonparallel)
        u=np.einsum('ij,ij->i',v,p)*inv;q=np.cross(v,e1);w=q@ray*inv;distance=np.einsum('ij,ij->i',e2,q)*inv
        hit=nonparallel&(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)&(distance>=near)&(distance<far)
        if hit.any():result[i]=float(np.min(distance[hit]))
    return result


def depth_residual(measured,expected):
    measured=np.asarray(measured,float);expected=np.asarray(expected,float)
    if measured.shape!=expected.shape or measured.ndim!=1:raise ValueError('Depth dimensions')
    finite=np.isfinite(measured)&np.isfinite(expected);err=np.abs(measured[finite]-expected[finite])
    return {'rays':len(measured),'both_finite':int(finite.sum()),'measured_miss_expected_hit':int((~np.isfinite(measured)&np.isfinite(expected)).sum()),
        'measured_hit_expected_miss':int((np.isfinite(measured)&~np.isfinite(expected)).sum()),
        'absolute_error_median_m':float(np.median(err))if len(err)else None,'absolute_error_p90_m':float(np.quantile(err,.9))if len(err)else None,
        'signed_error_median_m':float(np.median(measured[finite]-expected[finite]))if len(err)else None}


def tilted_camera(rotation,additional_down_degrees):
    if not math.isfinite(additional_down_degrees)or not 0<=additional_down_degrees<=25:raise ValueError('Bounded physical rotation required')
    a=math.radians(additional_down_degrees);c,s=math.cos(a),math.sin(a)
    # ROS right/down/forward. Physical rotation about camera +X, no pixel warp.
    return np.asarray(rotation)@np.array([[1,0,0],[0,c,s],[0,-s,c]])


def floor_reference_coverage(origin,rays,floor_z,near=.03,far=5.):
    o=np.asarray(origin,float);r=np.asarray(rays,float)
    if o.shape!=(3,)or r.ndim!=2 or r.shape[1]!=3 or not np.isfinite(floor_z):raise ValueError('Floor reference dimensions')
    with np.errstate(divide='ignore',invalid='ignore'):distance=(floor_z-o[2])/r[:,2]
    eligible=(r[:,2]<-1e-9)&(distance>=near)&(distance<far)
    return {'rays':len(r),'floor_reference_in_range_fraction':float(eligible.mean()),'upward_rays':int((r[:,2]>=0).sum()),
        'downward_beyond_far':int(((r[:,2]<0)&(distance>=far)).sum()),'static_plane_only':True}
