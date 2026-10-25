import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from isaac_workshop import curriculum as c
from isaac_workshop import evaluation_contract as e
from isaac_workshop import native_curriculum as n
from isaac_workshop.prepare_trainer import build_proposal


def domain():
    return {'schema':'validated-locomotion-domain-v1','checks':{k:True for k in ('source_geometry','collision_geometry','floor_support','start_clearance','trajectory_clearance','reset_profiles')},
        'root_domain':[[-2.,-2.,-.1],[2.,2.,.6]],'train_starts':[[0.,0.,0.]],'heldout_starts':[[.1,.1,.1]],
        'source_sha256':{'fixture':'1'*64},'reset_profiles':{'nominal':{},'small_perturbation':{}},'terrain_kind':'flat_source'}


def admission():
    return {'schema':'native-foundation-training-admission-v1','checks':{k:True for k in ('strict_model_reload','normalizer_parity','named_action_mapping','ordered_observation_semantics','source_robot_match','runtime_match','unmasked_velocity_commands','native_action_target_evidence','safe_tensor_deserialization')},
        'actor_dimension':61,'critic_dimension':76,'action_dimension':14,'checkpoint_sha256':'1'*64,'normalizer_sha256':'2'*64,
        'robot_sha256':'3'*64,'benchmark_sha256':'4'*64,'command_masked_slots':[],'morphology':'barefoot_backlash','normalizer_mode':'preserve_loaded_state'}


class CurriculumTests(unittest.TestCase):
    def test_parallel_reservations_cover_each_family(self):
        ledger=c.ExposureLedger('flat_control')
        self.assertEqual(len({ledger.begin(i).name for i in range(6)}),6)
        self.assertEqual(ledger.report()['solved_substeps'],0)

    def test_early_fall_retains_actual_denominator_and_rebalances(self):
        ledger=c.ExposureLedger('flat_control')
        for world in range(6):ledger.begin(world)
        for world in range(6):
            count=5 if world==0 else 200
            for index in range(count):ledger.record_solved_control(world,index,ledger.expected(world),substeps=4,terminal=index==count-1)
        self.assertEqual(ledger.begin(0).name,'stand')
        self.assertEqual(ledger.rows['stand']['active'],5)
        self.assertEqual(ledger.rows['stand']['completed'],0)

    def test_complete_stop_accounting(self):
        ledger=c.ExposureLedger('flat_control');ledger.begin(0)
        for index in range(300):ledger.record_solved_control(0,index,ledger.expected(0),substeps=4)
        self.assertEqual(ledger.rows['stand'],dict(active=200,stop=100,episodes=1,completed=1,terminated=0))
        self.assertEqual(ledger.report()['solved_substeps'],1200)

    def test_missing_duplicate_substeps_and_wrong_command_rejected(self):
        for index,substeps,cmd in [(1,4,[0,0,0]),(0,3,[0,0,0]),(0,4,[.1,0,0])]:
            ledger=c.ExposureLedger('flat_control');ledger.begin(0)
            with self.assertRaises(ValueError):ledger.record_solved_control(0,index,cmd,substeps=substeps)
        with self.assertRaises(ValueError):ledger.begin(0)

    def test_native_command_observation_no_mutation(self):
        obs={'policy':[[0.]*61],'critic':[[0.]*76]};obs['policy'][0][50]=.3;obs['critic'][0][53]=.3
        old=copy.deepcopy(obs);c.verify_command_observations(obs,[[0,0,.3]],[[0,0,.3]])
        self.assertEqual(obs,old)
        obs['policy'][0][50]=0
        with self.assertRaises(ValueError):c.verify_command_observations(obs,[[0,0,.3]],[[0,0,.3]])

    def test_target_excess_measured_without_clamping(self):
        values=[6.074,-3.568]+[0.]*12;original=list(values)
        result=c.target_metrics(values,[-.5]*14,[.5]*14)
        self.assertEqual(result['outside_count'],2);self.assertEqual(values,original)
        with self.assertRaises(ValueError):c.target_metrics([float('nan')]*14,[-1]*14,[1]*14)

    def test_legacy_masked_or_mismatched_admission_rejected(self):
        n.admit_foundation(admission())
        for field,value in [('command_masked_slots',[49,50]),('morphology','rollers'),('normalizer_mode','reset'),('actor_dimension',61+1)]:
            a=admission();a[field]=value
            with self.assertRaises(ValueError):n.admit_foundation(a)
        a=admission();a['checks']['normalizer_parity']=False
        with self.assertRaises(ValueError):n.admit_foundation(a)

    def test_domain_hash_failed_review_and_shared_starts_reject(self):
        d=domain();e.validate_domain(d,e.digest(d))
        with self.assertRaises(ValueError):e.validate_domain(d,'0'*64)
        d['checks']['floor_support']=False
        with self.assertRaises(ValueError):e.validate_domain(d,e.digest(d))
        d=domain();d['heldout_starts']=d['train_starts']
        with self.assertRaises(ValueError):e.validate_domain(d,e.digest(d))

    def test_heldout_command_and_start_bank(self):
        d=domain();suite=e.suite(d,e.digest(d));train={s.twist for s in c.skills('mixed_control')}
        for case in suite['cases']:
            if case['held_out_from_training_start_and_command']:
                self.assertNotIn(tuple(case['segments'][0]['body_twist']),train)
                self.assertNotIn(case['start_xy_yaw'],d['train_starts'])
        self.assertTrue(any(x.get('held_out_reset_and_start') for x in suite['cases']))

    def test_unreviewed_case_cannot_be_scored(self):
        d=domain();case=e.suite(d,e.digest(d))['cases'][0]
        with self.assertRaisesRegex(ValueError,'trajectory'):e.score(case,[],[],d,e.digest(d))

    def test_promotion_needs_all_cases_and_cannot_replace_baseline(self):
        d=domain();suite=e.suite(d,e.digest(d));reports={case['name']:{'valid':True,'qualified':True,'scenario':case,'domain_sha256':e.digest(d),'immutable_scorer_sha256':e.SCORER_SHA}for case in suite['cases']}
        result=e.promotion('a'*64,suite,reports,'b'*64)
        self.assertTrue(result['rounded_locomotion_qualified']);self.assertFalse(result['replace_baseline'])
        reports[next(iter(reports))]['qualified']=False
        self.assertFalse(e.promotion('a'*64,suite,reports,'b'*64)['rounded_locomotion_qualified'])
        reports.pop(next(iter(reports)))
        with self.assertRaises(ValueError):e.promotion('a'*64,suite,reports,'b'*64)

    def test_readback_cannot_be_missing_or_outside_declared_range(self):
        good={k:[sum(v)/2] for k,v in c.RANDOMIZATION['mixed_control'].items()}
        self.assertTrue(n.validate_randomization_readbacks('mixed_control',good))
        good['mass_scale']=[1.5]
        with self.assertRaises(ValueError):n.validate_randomization_readbacks('mixed_control',good)

    def test_proposal_cannot_claim_native_readiness(self):
        p=build_proposal();self.assertFalse(p['launch_ready']);self.assertFalse(p['training_performed'])
        self.assertIsNone(p['selected_foundation']);self.assertEqual(p['ppo']['first_smoke_worlds'],64)

    def test_native_prepare_preserves_rewards_and_motor_model(self):
        rewards=NS(**{k:NS(weight=v)for k,v in {'track_lin_vel':2.,'track_ang_vel':2.,'upright':2.,'pose':1.,'dof_pos_limits':-1.,'self_collisions':-1.,'action_rate_l2':-.1}.items()})
        events=NS(**{k:NS(params={})for k in ('foot_friction','mass_inertia','encoder_bias','randomize_com','randomize_head_com')},push_robot=object())
        cfg=NS(sim=NS(dt=.005),decimation=4,actions=NS(joint_pos=NS(scale=1.,use_default_offset=True,clip=None)),
            commands=NS(base_velocity=NS(class_type=object,ranges=NS(lin_vel_x=(-.4,.4),lin_vel_y=(-.3,.3),ang_vel_z=(-1.,1.)))),
            curriculum=NS(old_stage=object()),events=events,rewards=rewards,scene=NS(robot=NS(actuators={'servo':{'max_current':1.75,'delay':[3,6]}})))
        original=copy.deepcopy(cfg.scene.robot.actuators);weights={k:v.weight for k,v in vars(rewards).items()};d=domain()
        ledger,receipt=n.prepare(cfg,admission(),'flat_control',d,e.digest(d),lambda *args:True,NS(tensor=lambda x,**kw:np.array(x)))
        self.assertEqual(cfg.scene.robot.actuators,original);self.assertEqual(receipt['reward_weights'],weights)
        self.assertEqual(events.foot_friction.params['static_friction_range'],(.7,1.3))
        self.assertFalse(receipt['launch_ready']);self.assertTrue(receipt['solved_control_recorder_integration_pending'])
        with self.assertRaises(ValueError):n.prepare(cfg,admission(),'terrain_control',d,e.digest(d),lambda *args:True)

    def test_scorer_preserves_incomplete_trace_failure_and_original_module(self):
        import hashlib
        pin=hashlib.sha256(e.SCORER.read_bytes()).hexdigest();d=domain();case=e.suite(d,e.digest(d))['cases'][0]
        d['admitted_case_sha256']=[e.digest(case)]
        report=e.score(case,[],[],d,e.digest(d))
        self.assertFalse(report['valid']);self.assertFalse(report['qualified'])
        self.assertEqual(hashlib.sha256(e.SCORER.read_bytes()).hexdigest(),pin)

    def test_native_reset_zero_phase_and_profile_rejection(self):
        class Native:
            num_envs=1;device='cpu'
            def __init__(self):
                self.vel_command_b=np.zeros((1,3));self.is_standing_env=np.zeros(1,dtype=bool);self.is_heading_env=np.ones(1,dtype=bool)
        tensor=NS(tensor=lambda value,**kw:np.array(value))
        ledger=c.ExposureLedger('flat_control');cls=n.command_type(Native,ledger,lambda world,profile:True,tensor);obj=cls()
        obj._resample_command([0]);self.assertTrue(obj.is_standing_env[0]);self.assertFalse(obj.is_heading_env[0])
        ledger.record_solved_control(0,0,(0.,0.,0.),substeps=4,terminal=True);obj._resample_command([0]);self.assertEqual(ledger.live[0]['controls'],0)
        bad=n.command_type(Native,c.ExposureLedger('flat_control'),lambda *args:False,tensor)()
        with self.assertRaisesRegex(RuntimeError,'reset'):bad._resample_command([0])


if __name__=='__main__':unittest.main()
