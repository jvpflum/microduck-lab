"""Explicit synthetic traces exercise scorer failures; no native evidence claims."""
import math
import unittest
from isaac_workshop import score_locomotion as s


def fixture(case):
    x,y,yaw=case['start_xy_yaw'];trace=[];sub=[]
    def state(step,command):
        return dict(step=step,boundary='initial_after_reset' if not step else 'after_physics_before_autoreset',
            root_position_world_m=[x,y,.15],root_quaternion_xyzw=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)],
            root_velocity_body_m_s=list(command[:2])+[0.],command_body_twist=list(command),
            yaw_rad=yaw,tilt_rad=0.,autoreset=False,terminations=[])
    trace.append(state(0,[0.,0.,0.]))
    for i in range(sum(v['controls'] for v in case['segments'])):
        command=s.command_at(case,i);vx,vy,w=command
        x+=(vx*math.cos(yaw)-vy*math.sin(yaw))*s.DT;y+=(vx*math.sin(yaw)+vy*math.cos(yaw))*s.DT;yaw+=w*s.DT
        trace.extend([state(i+1,command),dict(boundary='control_returned',step=i+1,actions=[[0.]*14])])
        for j in range(4):
            sub.append(dict(control_step=i+1,substep=j+1,physics_step=i*4+j+1,physics_dt_s=.005,
                boundary='after_native_solver_before_control_autoreset',body_state_all_finite=True,
                contacts=dict(all_finite=True,all_worlds_observed=True,contact_substeps_covered=1,native_contact_count=0,raw_rows=[])))
    return trace,sub


class ScorerTests(unittest.TestCase):
    def score(self,name='walk_stop',change=None):
        c=s.SCENARIOS[name];t,p=fixture(c)
        if change:change(t,p)
        return s.score_episode(c,t,p)
    def test_ideal_walk_turn_stop_and_heldout_fixtures(self):
        for name in s.SCENARIOS:
            with self.subTest(name=name):self.assertTrue(self.score(name)['qualified'])
    def test_stationary_actor_cannot_pass_nonzero_goal(self):
        def change(t,p):
            for row in t:
                if 'root_position_world_m'in row:row['root_position_world_m']=t[0]['root_position_world_m'].copy();row['root_velocity_body_m_s']=[0.]*3
        r=self.score(change=change);self.assertTrue(r['valid']);self.assertFalse(r['checks']['endpoint_position_error_m'])
    def test_wrong_turn_and_inconsistent_pose_rejected(self):
        def change(t,p):
            for row in t:
                if 'yaw_rad'in row:row['yaw_rad']=math.pi/2;row['root_quaternion_xyzw']=[0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)]
        self.assertFalse(self.score('left_turn_stop',change)['checks']['endpoint_yaw_error_rad'])
        self.assertFalse(self.score(change=lambda t,p:t[299].update(yaw_rad=0.))['valid'])
    def test_transient_self_contact_counted_once_not_two_sides(self):
        t,p=fixture(s.SCENARIOS['walk_stop']);p[0]['contacts'].update(native_contact_count=1,raw_rows=[dict(row=0,world=0,force_on_shape0_world_n=[3.,4.,0.],shape_paths=['leg','battery'],robot_contacts=[dict(robot_part='leg',object_id='np_f970'),dict(robot_part='np_f970',object_id='leg')])])
        r=s.score_episode(s.SCENARIOS['walk_stop'],t,p)
        self.assertAlmostEqual(r['measurements']['nonsole_nonfloor_force_magnitude_integral_n_s'],.025)
        self.assertFalse(r['qualified'])
    def test_missing_substep_and_false_contact_coverage_fail(self):
        self.assertFalse(self.score(change=lambda t,p:p.pop())['valid'])
        self.assertFalse(self.score(change=lambda t,p:p[0]['contacts'].update(all_worlds_observed=False))['valid'])
        self.assertFalse(self.score(change=lambda t,p:p[2].update(physics_step=2))['valid'])
    def test_reset_splice_nan_or_truncated_trace_fail(self):
        for change in (lambda t,p:t[1].update(autoreset=True),lambda t,p:t[1].update(terminations=['fall']),lambda t,p:t[1].update(tilt_rad=float('nan')),lambda t,p:t.pop(),lambda t,p:t[2].update(actions=[[0.]*13])):
            self.assertFalse(self.score(change=change)['valid'])
    def test_stop_motion_and_stop_speed_fail(self):
        def change(t,p):
            for row in t:
                if 'root_position_world_m'in row and row['step']>200:
                    row['root_position_world_m'][1]+=.1*(row['step']-200)/100;row['root_velocity_body_m_s']=[.05,0.,0.]
        r=self.score(change=change);self.assertFalse(r['checks']['stop_phase_translation_m']);self.assertFalse(r['checks']['last_second_speed_rms_m_s'])
    def test_start_and_command_are_bound(self):
        self.assertFalse(self.score(change=lambda t,p:t[0]['root_position_world_m'].__setitem__(0,-.6))['checks']['start_xy_error_m'])
        self.assertFalse(self.score(change=lambda t,p:t[1].update(command_body_twist=[.14,0.,0.]))['valid'])
    def test_no_camera_navigation_promotion(self):self.assertFalse(self.score()['camera_navigation_qualified'])

if __name__=='__main__':unittest.main()

class TimeAndYawTests(unittest.TestCase):
    def test_stop_rms_uses_exact_50_poststep_samples_not_boundary(self):
        case=s.SCENARIOS['walk_stop'];t,p=fixture(case)
        t[2*250-1]['root_velocity_body_m_s']=[99.,0.,0.]
        r=s.score_episode(case,t,p)
        self.assertEqual(r['measurements']['last_second_speed_rms_m_s'],0.)
        t[2*251-1]['root_velocity_body_m_s']=[.1,0.,0.]
        r=s.score_episode(case,t,p)
        self.assertAlmostEqual(r['measurements']['last_second_speed_rms_m_s'],.1/math.sqrt(50))
    def test_last_second_yaw_unwraps_pi_boundary(self):
        case=s.scenario('cross_pi',start=(-.5,-11.,math.pi-.01),command=(0.,0.,0.))
        t,p=fixture(case)
        for row in t:
            if 'yaw_rad'in row and row['step']>=250:
                angle=math.pi-.01+(row['step']-250)*.0004
                row['yaw_rad']=s.wrapped_angle(angle);row['root_quaternion_xyzw']=[0.,0.,math.sin(angle/2),math.cos(angle/2)]
        r=s.score_episode(case,t,p)
        self.assertTrue(r['valid']);self.assertAlmostEqual(r['measurements']['last_second_yaw_change_rad'],.02)
