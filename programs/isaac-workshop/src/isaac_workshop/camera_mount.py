"""Explicit external virtual-camera mounting and native USD pose readback.

Corrects the inherited origin behind CAD head surfaces; not hardware calibration.
No robot mesh, joint, clipping range, field of view or guard is changed.
"""
import math

OFFSET=(.0155,-.0000913778,-.08748)
PITCH_DEGREES=125.
HEAD_SUFFIX='/Robot/Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft/HeadDepth'


def make_spec(perception):
    return perception.CameraSpec(width=64,height=48,offset_pos_m=OFFSET,pitch_about_head_y_deg=PITCH_DEGREES)


def explain_camera(receipt):
    receipt.update(extrinsics_provenance='Source-reviewed external virtual camera: original origin moved30mm head-forward to clear rigid head geometry, physical optical axis35degrees down at neutral; not hardware calibration',
        calibration_status='Engineering virtual external mount; all19rigid CAD head meshes checked, dynamic body self-occlusion unqualified',
        combined_revision={'pixel_center_offset':.5,'head_forward_translation_added_m':.03,'downward_pitch_added_degrees':15},
        unchanged={'resolution':[64,48],'horizontal_fov_degrees':87,'near_far_m':[.03,5],'frontal_valid_fraction':.8})
    return receipt


def expected_usd_local_matrix():
    # CameraCfg world convention: +X forward,+Z up. USD OpenGL: -Z forward,+Y up.
    # R_world=Ry125; OpenGL columns in world are (0,-1,0),(0,0,1),(-1,0,0).
    a=math.radians(PITCH_DEGREES);c,s=math.cos(a),math.sin(a)
    columns=((0.,-1.,0.),(s,0.,c),(-c,0.,s))
    return [list(x)+( [0.] ) for x in columns]+[list(OFFSET)+[1.]]


def validate_local_matrix(matrix):
    expected=expected_usd_local_matrix()
    if len(matrix)!=4 or any(len(x)!=4 for x in matrix)or not all(math.isfinite(float(x))for row in matrix for x in row):
        raise RuntimeError('Native camera local matrix shape/finite differs')
    error=max(abs(float(matrix[i][j])-expected[i][j])for i in range(4)for j in range(4))
    if error>2e-6:raise RuntimeError('Native physical camera mounting differs')
    return {'expected_local_usd_matrix':expected,'actual_local_usd_matrix':matrix,'maximum_absolute_matrix_difference':error,'tolerance':2e-6,'verified':True}


def verify_native(camera):
    from pxr import Usd,UsdGeom
    if len(camera._sensor_prims)!=1:raise RuntimeError('Exactly one mounted camera required')
    prim=camera._sensor_prims[0].GetPrim()
    if not str(prim.GetPath()).endswith(HEAD_SUFFIX):raise RuntimeError('Native camera attachment path differs')
    if tuple(camera.cfg.offset.pos)!=OFFSET or camera.cfg.offset.convention!='world':
        raise RuntimeError('Native mounting config differs')
    if camera.cfg.width!=64 or camera.cfg.height!=48 or tuple(camera.cfg.spawn.clipping_range)!=(.03,5.):
        raise RuntimeError('Native dimensions or clipping range changed')
    m=UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    if UsdGeom.Xformable(prim).GetResetXformStack():raise RuntimeError('Camera must retain physical head parenting')
    result=validate_local_matrix([[float(m[i][j])for j in range(4)]for i in range(4)])
    data=camera.data
    K=data.intrinsic_matrices.torch[0].detach().cpu().tolist()
    f=64/(2*math.tan(math.radians(87)/2));expected=[[f,0.,32.],[0.,f,24.],[0.,0.,1.]]
    difference=max(abs(K[i][j]-expected[i][j])for i in range(3)for j in range(3))
    if difference>1e-5:raise RuntimeError('Actual native intrinsics differ')
    actual_clip=[float(x)for x in camera._sensor_prims[0].GetClippingRangeAttr().Get()]
    stage_units=float(UsdGeom.GetStageMetersPerUnit(prim.GetStage()))
    if abs(stage_units-1.)>1e-12 or len(actual_clip)!=2 or max(abs(a-b)for a,b in zip(actual_clip,(.03,5.)))>1e-6:
        raise RuntimeError('Actual native clipping range or metre stage differs')
    result.update(camera_prim=str(prim.GetPath()),head_parent=str(prim.GetParent().GetPath()),
        actual_intrinsics=K,expected_intrinsics=expected,maximum_intrinsics_difference=difference,
        clipping_range_config_m=list(camera.cfg.spawn.clipping_range),
        clipping_range_usd_authored=actual_clip,stage_meters_per_unit=stage_units,
        no_robot_geometry_modified=True,hardware_calibrated=False)
    return result
