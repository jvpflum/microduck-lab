"""Task-owned integration hooks for a compatible native trainer; no launch API.

Prepare before create_env. Record each solved control BEFORE autoreset. Verify
native command observations immediately before every PPO.act/policy call.
"""
from __future__ import annotations
import copy
import math
from .curriculum import ExposureLedger, RANDOMIZATION, skills, command
from .evaluation_contract import validate_domain


def admit_foundation(receipt):
    required=('strict_model_reload','normalizer_parity','named_action_mapping','ordered_observation_semantics',
              'source_robot_match','runtime_match','unmasked_velocity_commands','native_action_target_evidence',
              'safe_tensor_deserialization')
    if receipt.get('schema')!='native-foundation-training-admission-v1' or any(receipt.get('checks',{}).get(k)is not True for k in required):
        raise ValueError('Foundation compatibility unresolved; no weight conversion or shape-only admission')
    if receipt.get('actor_dimension')!=61 or receipt.get('critic_dimension')!=76 or receipt.get('action_dimension')!=14:
        raise ValueError('Native observation/action dimensions differ')
    for name in ('checkpoint_sha256','normalizer_sha256','robot_sha256','benchmark_sha256'):
        value=receipt.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing exact foundation evidence: '+name)
    if receipt.get('command_masked_slots')!=[] or receipt.get('morphology')!='barefoot_backlash':
        raise ValueError('Roller or masked sprint policy is not a native walking foundation')
    if receipt.get('normalizer_mode')!='preserve_loaded_state':
        raise ValueError('Loaded foundation normalization must be preserved in first matched pilot')
    return receipt


def command_type(native_type, ledger, reset_readback, tensor_module=None):
    if tensor_module is None:
        import torch as tensor_module
    class RoundedCommand(native_type):
        def _assign(self, world, index):
            entry=ledger.live[world]
            value=command(entry['skill'], index)
            self.vel_command_b[world]=tensor_module.tensor(value,device=self.device,dtype=self.vel_command_b.dtype)
            self.is_standing_env[world]=value==(0.,0.,0.)
            if hasattr(self,'is_heading_env'):self.is_heading_env[world]=False

        def _resample_command(self, env_ids):
            ids=list(range(self.num_envs))[env_ids] if isinstance(env_ids,slice) else list(env_ids.tolist() if hasattr(env_ids,'tolist') else env_ids)
            for world in ids:
                world=int(world)
                # Native reset calls this before zeroing old episode length.
                # A source-domain reset adapter can reserve the next skill
                # before applying its physical reset events. Otherwise the
                # readback must reject a recovery profile not actually applied.
                selected=ledger.live[world]['skill'] if world in ledger.live else ledger.begin(world)
                if ledger.live[world]['controls']!=0 or ledger.live[world].get('resampled'):
                    raise RuntimeError('Unexpected mid-episode resampling')
                if reset_readback(world, selected.reset_profile) is not True:
                    raise RuntimeError('Actual source-seated reset does not match selected recovery/nominal profile')
                ledger.live[world]['resampled']=True
                self._assign(world,0)

        def _update_command(self):
            for world in range(self.num_envs):
                if world not in ledger.live:raise RuntimeError('Missing native reset or solved-control accounting')
                self._assign(world, ledger.live[world]['controls'])
    return RoundedCommand


def prepare(cfg, foundation, stage, domain, domain_sha256, reset_readback, tensor_module=None):
    admit_foundation(foundation);validate_domain(domain,domain_sha256);selected=skills(stage)
    if stage=='terrain_control' and domain.get('terrain_kind')!='validated_varied_source':
        raise ValueError('Later terrain stage requires separately validated varied geometry')
    if not math.isclose(cfg.sim.dt,.005) or cfg.decimation!=4:
        raise ValueError('Preserve native200Hz/50Hz cadence')
    action=cfg.actions.joint_pos
    if action.scale!=1. or action.use_default_offset is not True or action.clip is not None:
        raise ValueError('Action affine/default/clip semantics changed')
    ledger=ExposureLedger(stage)
    velocity=cfg.commands.base_velocity
    original_ranges={n:tuple(getattr(velocity.ranges,n))for n in ('lin_vel_x','lin_vel_y','ang_vel_z')}
    if any(not original_ranges[n][0]<=s.twist[i]<=original_ranges[n][1] for s in selected for i,n in enumerate(original_ranges)):
        raise ValueError('Curriculum exceeds native authored command range')
    if not callable(reset_readback):raise ValueError('Native reset-profile readback adapter required')
    velocity.class_type=command_type(velocity.class_type,ledger,reset_readback,tensor_module)
    velocity.heading_command=False;velocity.resampling_time_range=(7.,7.)
    cfg.episode_length_s=6.
    # Avoid inherited automatic schedules overriding this explicit recipe.
    if cfg.curriculum is not None:
        for name in vars(cfg.curriculum):
            if not name.startswith('_'):setattr(cfg.curriculum,name,None)
    cfg.events.push_robot=None
    ranges=RANDOMIZATION[stage]
    for name,field,key in [('foot_friction','static_friction_range','foot_friction'),
                           ('foot_friction','dynamic_friction_range','foot_friction'),
                           ('mass_inertia','mass_distribution_params','mass_scale'),
                           ('encoder_bias','bias_range','encoder_bias')]:
        getattr(cfg.events,name).params[field]=ranges[key]
    for name in ('randomize_com','randomize_head_com'):
        getattr(cfg.events,name).params['com_range']={axis:ranges['com_xyz'] for axis in 'xyz'}
    # Preserve BAM's physical voltage/delay/friction/armature model; record exact
    # readbacks separately. Recovery perturbations are source-domain reset specs,
    # never per-step pose edits or a scripted upright recovery.
    for name in ('track_lin_vel','track_ang_vel','upright','pose','dof_pos_limits','self_collisions','action_rate_l2'):
        term=getattr(cfg.rewards,name)
        if not math.isfinite(term.weight):raise ValueError('Invalid native reward '+name)
    if cfg.rewards.track_lin_vel.weight<=0 or cfg.rewards.track_ang_vel.weight<=0 or cfg.rewards.self_collisions.weight>=0:
        raise ValueError('Tracking/contact objectives must remain active')
    return ledger, {'stage':stage,'foundation_sha256':foundation['checkpoint_sha256'],'domain_sha256':domain_sha256,
        'native_command_ranges':original_ranges,'randomization':copy.deepcopy(ranges),
        'reward_weights':{n:getattr(cfg.rewards,n).weight for n in vars(cfg.rewards) if not n.startswith('_') and getattr(cfg.rewards,n) is not None},
        'reward_policy':'Native command-conditioned velocity/yaw, upright, pose, slip and contact terms retained; no race finish/progress reward.',
        'actor_normalization':'Preserve admitted loaded normalizer; no Arena command masking.',
        'required_runtime_checks':['every pre-inference actor/critic command slice','pre-reset exposure ledger','all4 solved contact samples','domain/reset profile and randomization readbacks','named target excess and BAM saturation diagnostics','fresh checkpoint/normalizer reload'],
        'reset_profile_application_pending':True,'solved_control_recorder_integration_pending':True,
        'native_domain_evidence_pending':True,'launch_ready':False,'training_launched':False}


def validate_randomization_readbacks(stage, samples):
    for key,limits in RANDOMIZATION[stage].items():
        values=samples.get(key)
        if not values or not all(math.isfinite(x) and limits[0]-1e-7<=x<=limits[1]+1e-7 for x in values):
            raise ValueError('Missing or out-of-envelope native sampled values: '+key)
    return True
