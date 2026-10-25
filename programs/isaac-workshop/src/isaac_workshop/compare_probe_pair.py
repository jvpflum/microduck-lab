"""Compare two actual native initial camera captures; never substitutes fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .probe_controller import depth_stop_evidence


def load(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    with (directory/'navigation.jsonl').open() as stream:row=json.loads(stream.readline())
    relative=row['raw_frame_file']
    if Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative or ':' in relative:
        raise ValueError('Invalid owned frame path')
    path=directory/relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Frame escapes result root')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['raw_frame_sha256']:raise ValueError('Raw frame changed')
    with np.load(path,allow_pickle=False) as data:frame={k:data[k].copy() for k in data.files}
    return summary,row,frame


def compare(clear_directory,block_directory):
    clear,cr,cf=load(clear_directory);block,br,bf=load(block_directory)
    def owned_hashes(summary):
        rows=summary['identity']['owned_files']
        # Both cases execute within the exact same native payload directory.
        return rows if isinstance(rows,dict) and rows else None
    agreements={name:float(np.max(np.abs(cf[name]-bf[name]))) for name in
        ('intrinsics','camera_to_body','root_position_w_m','root_quaternion_xyzw','policy_observation','goal_body_m')}
    ce=depth_stop_evidence(cf['depth_m'],cf['intrinsics'],cf['camera_to_body'])
    be=depth_stop_evidence(bf['depth_m'],bf['intrinsics'],bf['camera_to_body'])
    checks={'both_native_evidence_runs_passed':clear.get('success') is True and block.get('success') is True,
        'expected_actual_cases':clear['identity']['navigation_case']=='clear' and block['identity']['navigation_case']=='left',
        'same_seed':clear['identity']['seed']==block['identity']['seed'],
        'same_executed_source_and_owned_payload':bool(owned_hashes(clear)) and owned_hashes(clear)==owned_hashes(block)
            and clear['identity']['source_files']==block['identity']['source_files'],
        'same_base_and_trial_manifest':clear['identity']['attic']['base']==block['identity']['attic']['base']
            and clear['identity']['attic']['trial_manifest_sha256']==block['identity']['attic']['trial_manifest_sha256'],
        'same_frozen_locomotor':clear['models']['checkpoint_sha256']==block['models']['checkpoint_sha256'] and clear.get('model_state_unchanged') is True and block.get('model_state_unchanged') is True,
        'both_initial_pre_action_frames':cr['control_step']==br['control_step']==0 and cr['decision'] and br['decision'],
        'same_pose_calibration_state_goal':all(v<=2e-5 for v in agreements.values()),
        'actual_depth_changed':not np.array_equal(cf['depth_m'],bf['depth_m']),
        'clear_floor_did_not_trigger':ce['reason']=='depth_corridor_clear' and not ce['stop'],
        'block_triggered_above_floor_pixels':be['reason']=='measured_depth_obstacle' and be['stop'],
        'actual_command_changed_forward_to_stop':cr['command_body_twist'][0]>0 and br['command_body_twist']==[0.,0.,0.],
        'no_missing_camera_fallback':cr['fault'] is None and br['fault'] is None,
        'no_reset_splicing':clear.get('autoresets_performed')==block.get('autoresets_performed')==0}
    return {'schema':'native-camera-obstacle-stop-pair-v1','checks':checks,
        'matched_obstacle_sensing_demonstrated':all(checks.values()),'input_max_abs_differences':agreements,
        'clear_depth_evidence':ce,'blocked_depth_evidence':be,
        'raw_frame_sha256':{'clear':cr['raw_frame_sha256'],'blocked':br['raw_frame_sha256']},
        'source_summary_sha256':{str(p):hashlib.sha256((Path(p)/'summary.json').read_bytes()).hexdigest() for p in (clear_directory,block_directory)},
        'learned_avoidance':False,'robust_navigation_qualified':False,
        'limit':'Matched live sensing and stopping only. Full source/admission/contact/video audits remain separate; no physical safety claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--clear',type=Path,required=True);parser.add_argument('--blocked',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=compare(args.clear,args.blocked)
    with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'passed':value['matched_obstacle_sensing_demonstrated'],'checks':value['checks']}))
    raise SystemExit(0 if value['matched_obstacle_sensing_demonstrated'] else 1)
