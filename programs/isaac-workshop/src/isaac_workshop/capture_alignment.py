"""One source-bound render-only acquisition at the current physical state.

The controller receives only actual camera pixels. Exact source-floor rays are
an after-episode diagnostic, never a command or a replacement camera image.
"""
from __future__ import annotations
import base64
import hashlib
import gzip
import json
import math
from pathlib import Path
import threading
import zlib
import numpy as np
from . import camera_readiness
from .navigation_bridge import digest
from .alignment_metrics import camera_rays, rotation_xyzw, optical_depth_to_triangles, depth_residual

FLOOR_MANIFEST_SHA = '4a61926d91c96047b9331530a9d29aa33b9b4894875584c30d2f58c0945e6f8a'
PIXELS = np.array([(x,y) for y in (32,36,40) for x in (16,24,32,40,48)], dtype=float)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_floor(runtime_assets, here=None):
    here = Path(here) if here else Path(__file__).parent
    manifest = here/'floor-geometry.json'
    if sha(manifest) != FLOOR_MANIFEST_SHA:
        raise RuntimeError('Floor diagnostic manifest differs')
    cfg = json.loads(manifest.read_text())
    expected = {r['source_path']:r['geometry_sha256'] for r in cfg['surfaces']}
    actual = {r['source_path']:r['source_surface_sha256'] for r in runtime_assets['meshes'] if r['object_id']=='floor'}
    if actual != expected:
        raise RuntimeError('Diagnostic floor differs from admitted native source surfaces')
    encoded = here/cfg['encoded_file']
    if encoded.name != 'floor-triangles.f64.zlib.b64' or sha(encoded) != cfg['encoded_sha256']:
        raise RuntimeError('Floor diagnostic data differs')
    inflater = zlib.decompressobj()
    data = inflater.decompress(base64.b64decode(encoded.read_bytes(),validate=True),cfg['binary_bytes']+1)
    if (not inflater.eof or inflater.unused_data or inflater.unconsumed_tail or len(data)!=cfg['binary_bytes']
            or hashlib.sha256(data).hexdigest()!=cfg['binary_sha256']):
        raise RuntimeError('Bounded floor diagnostic decode failed')
    triangles = np.frombuffer(data,dtype='<f8').reshape(-1,3,3)
    if len(triangles)!=cfg['triangles'] or not np.isfinite(triangles).all():
        raise RuntimeError('Floor triangle contract differs')
    return triangles, {'manifest_sha256':FLOOR_MANIFEST_SHA,'surfaces':cfg['surfaces'],
                       'triangle_count':len(triangles),'diagnostic_only_not_policy_input':True}


def capture_transaction(snapshot, prepare, pump, read, row):
    """Testable sequence. Settings/hook lease is held by the native caller."""
    before = snapshot()
    row['frozen_before'] = before
    pose = prepare()
    row['pose_before_explicit_pump'] = pose
    if snapshot()!=before:
        raise RuntimeError('Capture preparation changed frozen state')
    if pump() is not True:
        raise RuntimeError('Native explicit Kit pump unavailable')
    row['explicit_pumps'] = 1
    row['pose_after_explicit_pump'] = prepare()
    if snapshot()!=before or row['pose_after_explicit_pump']!=pose:
        raise RuntimeError('State/pose drift during render-only pump')
    sample = read()
    row['pose_after_annotator_read'] = sample['pose']
    row['frozen_after'] = snapshot()
    if row['frozen_after']!=before or sample['pose']!=pose:
        raise RuntimeError('Annotator read changed frozen state or camera pose')
    row['frozen_state_verified'] = True
    row['pose_at_pump_unchanged_through_read'] = True
    return sample


def residual_record(pre, post, prior_pose, triangles):
    """Optical depth, explicit half-pixel centers; no changes to controller rays."""
    records = []
    for label,sample in (('pre',pre),('post',post)):
        measured = sample['depth'][PIXELS[:,1].astype(int),PIXELS[:,0].astype(int)]
        for hypothesis,pose in (('current_pump_pose',post['pose']),('previous_control_pump_pose',prior_pose)):
            if pose is None:
                continue
            rays = camera_rays(sample['intrinsics'],rotation_xyzw(pose['quaternion_ros_xyzw']),PIXELS,.5)
            expected = optical_depth_to_triangles(pose['position_world_m'],rays,triangles)
            records.append({'image':label,'pose_hypothesis':hypothesis,**depth_residual(measured,expected)})
    return records


class SameStateCapture:
    def __init__(self, base, source, output):
        import warp as wp
        import omni.timeline
        from isaaclab.app.settings_manager import get_settings_manager
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Same-state capture requires native main thread')
        self.base,self.output,self.wp = base,Path(output),wp
        self.camera=base.scene.sensors['head_depth'];self.robot=base.scene['robot']
        views=[v for v in base.sim.visualizers if type(v).__module__+'.'+type(v).__name__ ==
               'isaaclab_visualizers.kit.kit_visualizer.KitVisualizer']
        if len(views)!=1 or not self.camera.is_initialized:
            raise RuntimeError('One initialized Kit visualizer and camera required')
        self.visualizer=views[0];self.manager=base.sim.backend_manager
        self.timeline=omni.timeline.get_timeline_interface();self.settings=get_settings_manager()
        contract=camera_readiness.verify_source_contract(source,self.camera,self.visualizer,base.sim.render_context)
        self.frames=self.output/'aligned-camera-frames';self.frames.mkdir(exist_ok=False)
        self.stream=gzip.open(self.output/'capture-alignment.jsonl.gz','xt',encoding='utf-8')
        self.rows=[];self.receipt={'schema':'same-state-native-camera-capture-r4','source_contract':contract,
            'camera_mount_unchanged':False,'controller_unchanged':False,'capture_sequence_unchanged_from_r4':True,'frontal_guard_unchanged':.8,
            'native_gpu_acquisition_timestamp_available':False,'sensor_counter_is_not_completion_proof':True,
            'controller_projection':'Half-pixel centers calibrated from R3 exact native floor ray replay; same convention used by controller and offline diagnostic',
            'complete':False,'raw_pairs':0,'blocked_solver_call_attempts':0}
        self.closed=False

    def array(self,value):
        if hasattr(value,'torch'):value=value.torch
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        elif hasattr(value,'numpy'):value=value.numpy()
        a=np.ascontiguousarray(value)
        if not np.isfinite(a).all():raise RuntimeError('Nonfinite frozen native state')
        return a

    def record(self,value):
        a=self.array(value)
        return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest(a)}

    def pose(self):
        data=self.camera._data
        return {'position_world_m':self.array(data.pos_w)[0].tolist(),
                'quaternion_ros_xyzw':self.array(data.quat_w_ros)[0].tolist(),
                'intrinsics':self.array(data.intrinsic_matrices)[0].tolist()}

    def snapshot(self):
        b,m,v,t=self.base,self.manager,self.visualizer,self.timeline
        self.wp.synchronize_device(b.device)
        state=m.get_state_0()
        clocks={'physics_counter':int(b._sim_step_counter),'physics_count_api':int(b.sim.get_physics_step_count()),
                'control_counter':int(b.common_step_counter),'newton_s':float(m._sim_time),
                'visualizer_s':float(v._sim_time),'timeline_s':float(t.get_current_time()),
                'timeline_playing':bool(t.is_playing())}
        if not all(math.isfinite(x) for x in clocks.values()):raise RuntimeError('Nonfinite native clock')
        return {'clocks':clocks,'body_q':self.record(state.body_q),'body_qd':self.record(state.body_qd),
            'robot':{n:self.record(getattr(self.robot.data,n)) for n in
                ('root_link_pos_w','root_link_quat_w','root_link_lin_vel_b','root_link_ang_vel_b',
                 'body_link_pos_w','body_link_quat_w','body_link_lin_vel_w','body_link_ang_vel_w','joint_pos','joint_vel')},
            'episode_length':self.record(b.episode_length_buf),
            'command':self.record(b.command_manager.get_command('base_velocity')),
            'observations':{k:self.record(x) for k,x in b.obs_buf.items()},
            'camera_sampling_clock':self.record(self.camera._timestamp),
            'video':[{'steps':int(x._step_count),'frames':len(x._frames)} for x in b.video_recorders]}

    def sample(self):
        self.wp.synchronize_device(self.base.device)
        data=self.camera.data
        depth=data.output['distance_to_image_plane'].torch[0,...,0].detach().cpu().numpy().copy()
        rgb=data.output['rgb'].torch[0].detach().cpu().numpy().copy()
        if depth.shape!=(48,64) or rgb.shape[:2]!=depth.shape or rgb.dtype!=np.uint8:
            raise RuntimeError('Same-state camera dimensions/type differ')
        if np.isnan(depth).any() or np.isneginf(depth).any() or (depth<0).any():
            raise RuntimeError('Invalid metric depth; positive infinity remains unknown')
        pose=self.pose()
        return {'depth':depth,'rgb':rgb,'intrinsics':np.asarray(pose['intrinsics']), 'pose':pose,
                'sensor_frame':int(self.camera.frame.torch[0]),
                'sensor_timestamp_s':float(self.wp.to_torch(self.camera._timestamp_last_update)[0])}

    def capture(self,control_step):
        if type(control_step)is not int or control_step!=len(self.rows) or control_step>=100:
            raise RuntimeError('Exact consecutive bounded capture controls required')
        row={'control_step':control_step,'explicit_pumps':0,'restored':False,'successful':False}
        baseline=None;restore=[];post=None;pre=None
        t,s,m=self.timeline,self.settings,self.manager
        original_auto=bool(t.is_auto_updating());original_play=s.get('/app/player/playSimulations')
        if type(original_play)is not bool:raise RuntimeError('Native physics-player bool required')
        row.update(timeline_auto_update_before=original_auto,physics_player_before=original_play)
        def reject(*args,**kwargs):
            self.receipt['blocked_solver_call_attempts']+=1
            raise RuntimeError('Solver call prohibited during same-state capture')
        try:
            pre=self.sample();baseline=self.snapshot()
            row['pre_sample']={k:v for k,v in pre.items() if k not in ('depth','rgb','intrinsics')}
            for name in ('step','_run_solver_substeps'):
                original=getattr(m,name);restore.append((name,name in vars(m),original));setattr(m,name,reject)
            t.set_auto_update(False);t.commit();s.set('/app/player/playSimulations',False)
            def frozen_snapshot():
                if t.is_auto_updating() or s.get('/app/player/playSimulations')is not False:
                    raise RuntimeError('Capture freeze lost')
                return self.snapshot()
            if frozen_snapshot()!=baseline:raise RuntimeError('Freeze changed native state')
            def prepare():
                self.camera._renderer.update_transforms();self.camera._renderer.update_geometries()
                self.camera._update_poses(frame_op=0)
                self.wp.synchronize_device(self.base.device)
                return self.pose()
            def read():
                self.camera._invalidate_buffers();self.camera.update(0.,force_recompute=True)
                return self.sample()
            post=capture_transaction(frozen_snapshot,prepare,self.visualizer._pump_kit_app_update,read,row)
            if (post['sensor_frame']!=pre['sensor_frame']+1 or not math.isfinite(post['sensor_timestamp_s'])
                    or post['sensor_timestamp_s']!=pre['sensor_timestamp_s']
                    or not math.isclose(post['sensor_timestamp_s'],control_step*.02,rel_tol=0,abs_tol=1e-5)):
                raise RuntimeError('Render-only sample advanced sampling clock or wrong frame count')
            row['post_sample']={k:v for k,v in post.items() if k not in ('depth','rgb','intrinsics')}
            row['successful']=True
        except BaseException as exc:
            row['error']=str(exc);self.receipt['error']=str(exc)
            raise
        finally:
            try:
                s.set('/app/player/playSimulations',original_play);t.set_auto_update(original_auto);t.commit()
                row['state_after_restore']=self.snapshot()
                if (s.get('/app/player/playSimulations')is not original_play or bool(t.is_auto_updating())!=original_auto
                        or baseline is not None and row['state_after_restore']!=baseline):
                    row['successful']=False
                    raise RuntimeError('Same-state capture restoration drift')
                row['restored']=True
            finally:
                for name,was_instance,original in reversed(restore):
                    if was_instance:setattr(m,name,original)
                    else:delattr(m,name)
                row['solver_methods_restored']=True
                payload={}
                for label,sample in (('pre',pre),('post',post)):
                    if sample is None:continue
                    for key in ('depth','rgb','intrinsics','sensor_frame','sensor_timestamp_s'):
                        payload[label+'_'+key]=sample[key]
                    payload[label+'_position_world_m']=sample['pose']['position_world_m']
                    payload[label+'_quaternion_ros_xyzw']=sample['pose']['quaternion_ros_xyzw']
                if payload:
                    path=self.frames/f'{control_step:06d}.npz'
                    with path.open('xb')as stream:np.savez_compressed(stream,**payload)
                    row.update(raw_pair_file=path.relative_to(self.output).as_posix(),raw_pair_sha256=sha(path))
                self.rows.append(row);self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush()
                self.receipt['raw_pairs']=sum(bool(x.get('pre_sample') and x.get('post_sample'))for x in self.rows)
                self.persist()

    def persist(self):
        (self.output/'capture-alignment-summary.json').write_text(json.dumps(self.receipt,indent=2,allow_nan=False)+'\n')

    def finish(self):
        if self.closed:return self.receipt
        self.closed=True;self.stream.close()
        try:
            assets=json.loads((self.output/'attic-runtime-assets/runtime-assets.json').read_text())
            triangles,pins=load_floor(assets)
            previous=None;records=[]
            for row in self.rows:
                if not row.get('successful') or not row.get('restored'):continue
                path=self.output/row['raw_pair_file']
                if sha(path)!=row['raw_pair_sha256']:raise RuntimeError('Owned raw capture changed')
                with np.load(path,allow_pickle=False)as data:
                    samples={label:{'depth':data[label+'_depth'],'intrinsics':data[label+'_intrinsics'],
                        'pose':{'position_world_m':data[label+'_position_world_m'].tolist(),
                                'quaternion_ros_xyzw':data[label+'_quaternion_ros_xyzw'].tolist()}}
                             for label in ('pre','post')}
                    records.append({'control_step':row['control_step'],'hypotheses':residual_record(samples['pre'],samples['post'],previous,triangles)})
                    previous=samples['post']['pose']
            current=[h for r in records for h in r['hypotheses'] if h['image']=='post' and h['pose_hypothesis']=='current_pump_pose']
            evidence={'schema':'sampled-floor-acquisition-diagnostic-r4','source':pins,'pixel_center_offset':.5,
                'pixels_xy':PIXELS.tolist(),'records':records,'controller_projection_changed':False,
                'limits':['Only15 fixed floor-reference rays; other objects can occlude them.','No measured GPU timestamp or full-image correctness.','These oracle rays are never supplied to the controller.']}
            path=self.output/'capture-floor-residuals.json';path.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
            self.receipt.update(source_floor_diagnostic_sha256=sha(path),source_floor_diagnostic_complete=len(records)==len(self.rows),
                post_current_pose_max_median_error_m=max((h['absolute_error_median_m'] for h in current if h['absolute_error_median_m']is not None),default=None),
                sampled_floor_current_pose_matches=bool(current) and all(h['both_finite']==15 and h['absolute_error_median_m']<=.001 for h in current),
                sampled_geometry_tolerance_m=.001,
                geometry_match_scope='Median of15 fixed near-floor rays per image; does not establish full image or obstacle reconstruction')
        except BaseException as exc:
            self.receipt.update(diagnostic_error=str(exc),source_floor_diagnostic_complete=False,sampled_floor_current_pose_matches=False)
        self.receipt.update(complete=bool(self.rows) and all(r.get('successful') and r.get('restored') and r.get('solver_methods_restored')for r in self.rows),
            controls=len(self.rows),explicit_pumps=sum(r.get('explicit_pumps',0)for r in self.rows),
            trace_sha256=sha(self.output/'capture-alignment.jsonl.gz'),trace_file='capture-alignment.jsonl.gz',trace_encoding='complete gzip UTF-8 JSONL; no rows omitted')
        self.persist()
        return self.receipt
