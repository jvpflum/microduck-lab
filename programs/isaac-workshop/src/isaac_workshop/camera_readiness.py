"""Source-bound RTX startup sampling at zero physics time, before any action.

Only the head camera's capture buffers/frame count may change. The first
locomotor observation is deliberately left untouched; normal command refresh
later refreshes the depth observation group without advancing history.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading
import numpy as np
from .navigation_bridge import CameraPacket, Contract, digest, image_channels

MAX_PUMPS = 32
REQUIRED_CONSECUTIVE = 2


def readiness(raw, contract):
    """Use the existing interlock, without disguising near/unknown pixels."""
    raw = np.asarray(raw)
    result = {'ready': False, 'raw_depth_sha256': digest(raw)}
    try:
        image = image_channels(CameraPacket(0, 0, 1, 0., 0., raw), contract)
        y0, y1 = int(contract.height*.25), math.ceil(contract.height*.8)
        x0, x1 = int(contract.width*.2), math.ceil(contract.width*.8)
        frontal = float(image[1, y0:y1, x0:x1].mean())
        result.update(valid_fraction=float(image[1].mean()), frontal_valid_fraction=frontal)
        if frontal < contract.minimum_frontal_valid_fraction:
            raise ValueError('Unknown frontal depth requires stop')
        result.update(ready=True, reason=None)
    except (ValueError, TypeError) as exc:
        result['reason'] = str(exc)
    return result


def bounded_warmup(snapshot, capture, receipt, contract):
    """Testable control flow; capture must pump then sample at dt=0."""
    baseline = snapshot()
    receipt.update(max_render_pumps=MAX_PUMPS, required_consecutive_ready=REQUIRED_CONSECUTIVE,
                   state_before=baseline, attempts=[], ready=False, physics_advanced=False)
    previous_frame = None
    streak = 0
    for attempt in range(1, MAX_PUMPS+1):
        if snapshot() != baseline:
            raise RuntimeError('State drift before camera warmup capture')
        depth, metadata = capture(attempt)
        after = snapshot()
        row = {'attempt': attempt, **metadata, **readiness(depth, contract),
               'state_unchanged': after == baseline}
        receipt['attempts'].append(row)
        if after != baseline:
            raise RuntimeError('Camera warmup changed physics, clocks, commands or observations')
        frame, stamp = metadata['sensor_frame'], metadata['sensor_timestamp_s']
        if (type(frame) is not int or frame < 1 or previous_frame is not None and frame != previous_frame+1
                or not math.isfinite(stamp) or stamp != 0.):
            raise RuntimeError('Camera warmup capture index or zero-time stamp differs')
        previous_frame = frame
        streak = streak+1 if row['ready'] else 0
        row['consecutive_ready'] = streak
        if streak == REQUIRED_CONSECUTIVE:
            receipt.update(ready=True, render_pumps=attempt, state_after=after,
                           frozen_state_verified=True, unknown_depth_forward_authorized=False)
            return
    raise RuntimeError('Camera readiness unknown after bounded render-only warmup; no action authorized')


def verify_source_contract(source, camera, visualizer, context):
    """Exact pinned public source, newline-normalized for native Windows."""
    from isaaclab.sensors.sensor_base import SensorBase
    from isaaclab_physx.renderers import isaac_rtx_renderer_utils
    manifest_path = Path(__file__).with_name('warmup-source-contract.json')
    manifest = json.loads(manifest_path.read_text())
    objects = {'camera': type(camera), 'sensor_base': SensorBase,
               'render_context': type(context), 'rtx_renderer': type(camera._renderer),
               'rtx_utils': isaac_rtx_renderer_utils, 'kit_visualizer': type(visualizer)}
    checked = {}
    for name, obj in objects.items():
        path = Path(inspect.getfile(obj)).resolve()
        expected = (Path(source)/manifest['sources'][name]['path']).resolve()
        if path != expected or not path.is_relative_to(Path(source).resolve()):
            raise RuntimeError('Warmup native source path differs: '+name)
        actual = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        if actual != manifest['sources'][name]['normalized_text_sha256']:
            raise RuntimeError('Warmup native source hash differs: '+name)
        checked[name] = {'path': str(path), 'normalized_text_sha256': actual}
    return {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'sources': checked}


def ensure_camera_ready(base, source, output, receipt, contract=None):
    """Render without env.step/reset, solver calls, history or command changes."""
    import warp as wp
    import omni.timeline
    from isaaclab.app.settings_manager import get_settings_manager
    contract = contract or Contract()
    receipt.update(schema='native-render-only-camera-readiness-r3', ready=False, action_authorized=False,
                   native_validated=False, no_policy_or_optimizer_called=True,
                   sensor_frame_count_is_not_gpu_completion_counter=True,
                   scope='Startup capture readiness only; not learned navigation or obstacle qualification')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Camera warmup must run on native main thread')
    camera = base.scene.sensors['head_depth']
    views = [v for v in base.sim.visualizers if type(v).__module__+'.'+type(v).__name__ ==
             'isaaclab_visualizers.kit.kit_visualizer.KitVisualizer']
    if len(views) != 1 or not camera.is_initialized:
        raise RuntimeError('One initialized Kit visualizer and native camera required')
    visualizer = views[0]
    receipt['source_contract'] = verify_source_contract(source, camera, visualizer, base.sim.render_context)
    timeline = omni.timeline.get_timeline_interface()
    settings = get_settings_manager()
    manager = base.sim.backend_manager
    robot = base.scene['robot']
    output = Path(output)
    raw_dir = output/'readiness-frames'; raw_dir.mkdir(exist_ok=False)
    receipt_path = output/'camera-readiness.json'

    def array(value):
        if hasattr(value, 'torch'): value = value.torch
        if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'): value = value.numpy()
        result = np.ascontiguousarray(value)
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite frozen native state')
        return result

    def record(value):
        a = array(value)
        return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': digest(a)}

    def snapshot():
        wp.synchronize_device(base.device)
        state = manager.get_state_0()
        clocks = {'physics_counter': int(base._sim_step_counter),
                  'physics_count_api': int(base.sim.get_physics_step_count()),
                  'control_counter': int(base.common_step_counter),
                  'newton_s': float(manager._sim_time), 'visualizer_s': float(visualizer._sim_time),
                  'timeline_s': float(timeline.get_current_time()), 'timeline_playing': bool(timeline.is_playing())}
        if not all(math.isfinite(v) for v in clocks.values()): raise RuntimeError('Nonfinite clock')
        return {'clocks': clocks, 'body_q': record(state.body_q), 'body_qd': record(state.body_qd),
                'robot': {n: record(getattr(robot.data, n)) for n in (
                    'root_link_pos_w','root_link_quat_w','root_link_lin_vel_b','root_link_ang_vel_b',
                    'body_link_pos_w','body_link_quat_w','body_link_lin_vel_w','body_link_ang_vel_w','joint_pos','joint_vel')},
                'episode_length': record(base.episode_length_buf),
                'command': record(base.command_manager.get_command('base_velocity')),
                'observations': {k: record(v) for k,v in base.obs_buf.items()},
                'camera_sampling_clock': record(camera._timestamp),
                'camera_pose': {n: record(getattr(camera._data,n)) for n in ('pos_w','quat_w_ros','intrinsic_matrices')},
                'video': [{'steps': int(v._step_count), 'frames': len(v._frames)} for v in base.video_recorders]}

    def persist():
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    original_auto = bool(timeline.is_auto_updating())
    original_play_flag = settings.get('/app/player/playSimulations')
    if type(original_play_flag) is not bool:
        raise RuntimeError('Explicit native physics-player bool required before warmup')
    baseline = snapshot()
    initial = {'sensor_timestamp_s': array(camera._timestamp).tolist(),
               'native_command_body_twist': array(base.command_manager.get_command('base_velocity')).tolist(),
               'actor_command_slots': array(base.obs_buf['policy'])[:,48:51].tolist(),
               'critic_command_slots': array(base.obs_buf['critic'])[:,51:54].tolist()}
    initial['checks'] = {
        'sensor_sampling_clock_zero': bool(np.all(np.asarray(initial['sensor_timestamp_s']) == 0)),
        'native_command_zero': bool(np.all(np.asarray(initial['native_command_body_twist']) == 0)),
        'actor_command_zero': bool(np.all(np.asarray(initial['actor_command_slots']) == 0)),
        'critic_command_zero': bool(np.all(np.asarray(initial['critic_command_slots']) == 0))}
    receipt['initial_preconditions'] = initial
    persist()
    if not all(initial['checks'].values()):
        receipt['error'] = 'Warmup initial zero-state preconditions failed: '+str(initial['checks'])
        persist()
        raise RuntimeError(receipt['error'])
    restore = []
    receipt['blocked_solver_call_attempts'] = 0
    def reject_solver(*args, **kwargs):
        receipt['blocked_solver_call_attempts'] += 1
        raise RuntimeError('Physics invocation forbidden during camera readiness')
    try:
        for name in ('step','_run_solver_substeps'):
            original = getattr(manager,name)
            restore.append((name,name in vars(manager),original))
            setattr(manager,name,reject_solver)
        timeline.set_auto_update(False); timeline.commit()
        settings.set('/app/player/playSimulations',False)
        if timeline.is_auto_updating() or settings.get('/app/player/playSimulations') is not False or snapshot()!=baseline:
            raise RuntimeError('Camera warmup freeze did not preserve initial state')
        receipt['timeline_auto_update_before'] = original_auto
        receipt['physics_player_before'] = original_play_flag
        def capture(attempt):
            if timeline.is_auto_updating() or settings.get('/app/player/playSimulations') is not False:
                raise RuntimeError('Render-only freeze lost before capture')
            # Publish fixed geometry/poses before the explicit fresh Kit pump.
            camera._renderer.update_transforms(); camera._renderer.update_geometries()
            camera._update_poses(frame_op=0)
            if visualizer._pump_kit_app_update() is not True:
                raise RuntimeError('Native Kit render-only pump unavailable')
            # update(0, force_recompute=True) alone does not invalidate cached data.
            camera._invalidate_buffers()
            camera.update(0., force_recompute=True)
            wp.synchronize_device(base.device)
            if timeline.is_auto_updating() or settings.get('/app/player/playSimulations') is not False:
                raise RuntimeError('Render-only freeze lost during capture')
            data = camera.data
            depth = data.output['distance_to_image_plane'].torch[0,...,0].detach().cpu().numpy().copy()
            rgb = data.output['rgb'].torch[0].detach().cpu().numpy().copy()
            frame = int(camera.frame.torch[0]); stamp = float(wp.to_torch(camera._timestamp_last_update)[0])
            if rgb.shape[:2] != depth.shape or rgb.dtype != np.uint8:
                raise RuntimeError('Native RGB/depth startup dimensions differ')
            path = raw_dir/f'{attempt:03d}.npz'
            np.savez_compressed(path,depth_m=depth,rgb=rgb,sensor_frame=frame,sensor_timestamp_s=stamp,
                intrinsics=array(data.intrinsic_matrices),camera_pos_w=array(data.pos_w),camera_quat_ros=array(data.quat_w_ros))
            return depth, {'sensor_frame':frame,'sensor_timestamp_s':stamp,
                           'explicit_render_pump':True,'sampling_dt_s':0.,
                           'raw_frame_file':path.relative_to(output).as_posix(),
                           'raw_frame_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        bounded_warmup(snapshot,capture,receipt,contract)
        receipt['native_validated'] = True
    except BaseException as exc:
        receipt.update(ready=False,error=str(exc),action_authorized=False)
        raise
    finally:
        try:
            settings.set('/app/player/playSimulations',original_play_flag)
            timeline.set_auto_update(original_auto); timeline.commit()
            if settings.get('/app/player/playSimulations') is not original_play_flag or bool(timeline.is_auto_updating())!=original_auto or snapshot()!=baseline:
                receipt.update(ready=False,restored=False)
                raise RuntimeError('Warmup restoration changed native state/settings')
            receipt.update(restored=True,final_state_unchanged=True)
        finally:
            for name,was_instance,original in reversed(restore):
                if was_instance: setattr(manager,name,original)
                else: delattr(manager,name)
            receipt['solver_methods_restored'] = True
            receipt['action_authorized'] = bool(receipt.get('ready') and receipt.get('restored'))
            persist()
