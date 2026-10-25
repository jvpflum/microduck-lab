"""Predeclared low-level locomotion scenarios and strict native trace scorer.

This is not camera navigation qualification. It requires actual solved contacts
and complete, unspliced episodes. No renderer, simulator or job is launched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from .contact_metrics import contact_metrics

DT=.02
ROOT_DOMAIN=((-1.,-12.2,-.1),(-.4,-8.8,.6))
# Engineering acceptance tolerances for future matched trials, not calibrated
# hardware safety limits. Preserve zero nonsole/nonfloor contact qualification.
THRESHOLDS={'start_xy_error_m':.001,'start_yaw_error_rad':.001,
    'maximum_tilt_rad':.35,'endpoint_position_error_m':.10,
    'endpoint_yaw_error_rad':.15,'stop_phase_translation_m':.08,
    'last_second_translation_m':.025,'last_second_speed_rms_m_s':.03,
    'final_horizontal_speed_m_s':.03,'last_second_yaw_change_rad':.05,
    'nonsole_nonfloor_force_magnitude_integral_n_s':0.}


def scenario(name,start=(-.5,-11.,math.pi/2),command=(.15,0.,0.),moving=200,stopped=100,held_out=False):
    return {'name':name,'start_xy_yaw':list(start),'segments':[
        {'controls':moving,'body_twist':list(command)},
        {'controls':stopped,'body_twist':[0.,0.,0.]}],
        'held_out_from_training_start_and_command':held_out}


SCENARIOS={s['name']:s for s in [
    scenario('walk_stop'),scenario('left_turn_stop',command=(0.,0.,.3)),
    scenario('right_turn_stop',command=(0.,0.,-.3)),
    scenario('heldout_walk_slow',start=(-.8,-11.6,math.pi/2+.1),command=(.10,0.,0.),held_out=True),
    scenario('heldout_right_turn',start=(-.65,-10.8,math.pi/2-.1),command=(0.,0.,-.2),held_out=True),
    # Prior recorded video is a schema/regression reference, never a held-out trial.
    scenario('legacy_video_500',moving=400)]}


def command_at(case,index):
    if type(index)is not int or index<0:raise ValueError('Invalid control index')
    for segment in case['segments']:
        if index<segment['controls']:return segment['body_twist']
        index-=segment['controls']
    raise ValueError('Control outside declared episode')


def wrapped_angle(value):return math.atan2(math.sin(value),math.cos(value))


def vector(value,n,label):
    if not isinstance(value,list) or len(value)!=n or any(type(x)not in (int,float) or not math.isfinite(x) for x in value):
        raise ValueError('Invalid finite '+label)
    return value


def expected_endpoint(case):
    x,y,yaw=case['start_xy_yaw']
    for segment in case['segments']:
        vx,vy,w=segment['body_twist'];duration=segment['controls']*DT
        if abs(w)<1e-12:
            x+=(vx*math.cos(yaw)-vy*math.sin(yaw))*duration
            y+=(vx*math.sin(yaw)+vy*math.cos(yaw))*duration
        else:
            end=yaw+w*duration
            x+=(vx*(math.sin(end)-math.sin(yaw))+vy*(math.cos(end)-math.cos(yaw)))/w
            y+=(vx*(math.cos(yaw)-math.cos(end))+vy*(math.sin(end)-math.sin(yaw)))/w
        yaw+=w*duration
    return [x,y,yaw]


def validate_episode(case,trace,substeps):
    total=sum(s['controls'] for s in case['segments'])
    initial=[r for r in trace if r.get('boundary')=='initial_after_reset']
    states=[r for r in trace if r.get('boundary')=='after_physics_before_autoreset']
    returned=[r for r in trace if r.get('boundary')=='control_returned']
    if len(initial)!=1 or initial[0].get('step')!=0:raise ValueError('One initial state required')
    if len(trace)!=1+2*total or len(states)!=total or len(returned)!=total:
        raise ValueError('Complete state/action trace required; no terminal truncation accepted')
    if trace[0]!=initial[0]:raise ValueError('Initial state must be first')
    for i in range(total):
        if trace[1+2*i]!=states[i] or trace[2+2*i]!=returned[i] or states[i].get('step')!=i+1 or returned[i].get('step')!=i+1:
            raise ValueError('Noncontiguous or reordered episode trace')
        actions=returned[i].get('actions');
        if not isinstance(actions,list) or len(actions)!=1:raise ValueError('One actual action vector required')
        vector(actions[0],14,'actions')
        command=vector(states[i].get('command_body_twist'),3,'command')
        if any(abs(a-b)>1e-7 for a,b in zip(command,command_at(case,i))):raise ValueError('Actual command differs from predeclared schedule')
    all_states=initial+states
    for row in all_states:
        vector(row.get('root_position_world_m'),3,'position');vector(row.get('root_velocity_body_m_s'),3,'velocity')
        q=vector(row.get('root_quaternion_xyzw'),4,'quaternion')
        if abs(sum(x*x for x in q)-1)>1e-4:raise ValueError('Nonunit root quaternion')
        for key in ('yaw_rad','tilt_rad'):
            if type(row.get(key))not in (int,float) or not math.isfinite(row[key]):raise ValueError('Invalid '+key)
        if not 0<=row['tilt_rad']<=math.pi:raise ValueError('Invalid tilt range')
        x,y,z,w=q
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        tilt=math.acos(max(-1.,min(1.,1-2*(x*x+y*y))))
        if abs(wrapped_angle(row['yaw_rad']-yaw))>1e-4 or abs(row['tilt_rad']-tilt)>1e-4:raise ValueError('Quaternion/angle evidence disagrees')
        if row.get('autoreset')is not False or row.get('terminations')!=[]:raise ValueError('Reset or native termination present')
        xyz=row['root_position_world_m']
        if not all(ROOT_DOMAIN[0][i]<=xyz[i]<=ROOT_DOMAIN[1][i] for i in range(3)):
            raise ValueError('Observed state leaves admitted root domain')
    if substeps and substeps[0].get('boundary')=='initial_after_reset_unsolved_buffer':substeps=substeps[1:]
    if len(substeps)!=total*4:raise ValueError('Every solved substep required')
    metrics=[]
    for i in range(total):
        group=substeps[i*4:i*4+4]
        for j,row in enumerate(group):
            if row.get('control_step')!=i+1 or row.get('physics_step')!=i*4+j+1 or row.get('body_state_all_finite')is not True:
                raise ValueError('Solved-step identity/body coverage differs')
        metrics.append(contact_metrics(group))
    return all_states,metrics


def score_episode(case,trace,substeps):
    base={'schema':'attic-low-level-locomotion-score-v1','scenario':case,
          'thresholds':THRESHOLDS,'camera_navigation_qualified':False,
          'scope':'Privileged-command native locomotion only; not autonomous or camera-conditioned navigation.'}
    try:states,contacts=validate_episode(case,trace,substeps)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(base,valid=False,qualified=False,error=str(exc))
    first,last=states[0],states[-1];expected=expected_endpoint(case)
    stop_start=sum(s['controls'] for s in case['segments'][:-1]);tail=states[-51:]
    xy=lambda row:row['root_position_world_m'][:2]
    speed=lambda row:math.hypot(*row['root_velocity_body_m_s'][:2])
    values={'start_xy_error_m':math.dist(xy(first),case['start_xy_yaw'][:2]),
        'start_yaw_error_rad':abs(wrapped_angle(first['yaw_rad']-case['start_xy_yaw'][2])),
        'maximum_tilt_rad':max(row['tilt_rad'] for row in states),
        'endpoint_position_error_m':math.dist(xy(last),expected[:2]),
        'endpoint_yaw_error_rad':abs(wrapped_angle(last['yaw_rad']-expected[2])),
        'stop_phase_translation_m':math.dist(xy(states[stop_start]),xy(last)),
        'last_second_translation_m':math.dist(xy(tail[0]),xy(last)),
        'last_second_speed_rms_m_s':math.sqrt(sum(speed(row)**2 for row in tail[1:])/50),
        'final_horizontal_speed_m_s':speed(last),
        'last_second_yaw_change_rad':sum(abs(wrapped_angle(b['yaw_rad']-a['yaw_rad'])) for a,b in zip(tail,tail[1:])),
        'nonsole_nonfloor_force_magnitude_integral_n_s':sum(row['nonsole_or_nonfloor_magnitude_integral_n_s'] for row in contacts)}
    checks={key:value<=THRESHOLDS[key] for key,value in values.items()}
    return dict(base,valid=True,qualified=all(checks.values()),checks=checks,measurements=values,
        expected_endpoint_xy_yaw=expected,controls=len(states)-1,solved_substeps=(len(states)-1)*4,
        maximum_illegal_contact_force_n=max(row['maximum_nonsole_or_nonfloor_force_n'] for row in contacts),
        contact_definition='Actual native rows counted once, no threshold or self-contact exemption; force-magnitude time integral is not net impulse or damage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=SCENARIOS,required=True)
    parser.add_argument('--trace',type=Path,required=True);parser.add_argument('--contacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows=lambda p:[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    report=score_episode(SCENARIOS[args.scenario],rows(args.trace),rows(args.contacts))
    report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.trace,args.contacts)}
    report['scorer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'valid':report['valid'],'qualified':report['qualified'],'output':str(args.output)}))


if __name__=='__main__':main()
