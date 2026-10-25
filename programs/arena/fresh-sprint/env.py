"""Isolated Arena-aligned MJLab/BAM environment; upstream packages remain unchanged.

BAM subset computation below follows bam.mjlab at pinned62bd8ce; Copyright2025
Marc Duclusaud and Gregoire Passault, Apache2.0. It changes batching only.
The actor uses61 actual sensor/action/command values. GPU FP32 and Torch RNG
are surrogate physics; the unchanged CPU Arena judge remains authoritative.
"""
from __future__ import annotations
import dataclasses
import importlib.util
import math
from pathlib import Path
import numpy as np

JOINT_NAMES=('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
 'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll',
 'right_hip_pitch','right_knee','right_ankle')
DEFAULT_POSE=np.array([0.,-.08726646259971647,-.457924,-.004940,.452984,
 .3490658503988659,.3490658503988659,0.,0.,0.,.08726646259971647,.457924,.004940,-.452984])
PENALTY_TERMS=('time_cost','heading_cost','lateral_cost','tilt_cost','stall_cost','action_rate_cost','fall_cost')
PHYSICS_DT=.005
CONTROL_DT=.020
SETTLE_STEPS=25


def control_schedule():
    return ('drive','physics','drive','physics','drive','physics','drive','physics','drive','forward')


def compose_actor_numpy(gyro,gravity,qrel,lagged_vel,previous_actual,command_vx):
    value=np.concatenate([gyro,gravity,qrel,lagged_vel,previous_actual,
        np.asarray(command_vx).reshape(-1,1),np.zeros((len(gyro),12))],axis=-1)
    if value.shape[1]!=61 or not np.isfinite(value).all():raise ValueError('FiniteNx61 required')
    return value.astype(np.float32)


def _tensor(x):
    import torch
    return x if isinstance(x,torch.Tensor) else x[...]


def actor_observation(env):
    import torch
    if not getattr(env,'_arena_ready',False):return torch.zeros(env.num_envs,61,device=env.device)
    data=env.sim.data
    gravity=-_tensor(data.xmat)[:,env._trunk,2,:]
    return torch.cat([_tensor(data.sensordata)[:,env._gyro:env._gyro+3],gravity,
        _tensor(data.qpos)[:,env._qpos]-env._default,env._lagged_vel,env._previous_actual,
        env._command_vx[:,None],torch.zeros(env.num_envs,12,device=env.device)],-1)


def critic_observation(env):
    import torch
    obs=actor_observation(env)
    if not getattr(env,'_arena_ready',False):return torch.cat([obs,torch.zeros(env.num_envs,8,device=env.device)],-1)
    qpos=_tensor(env.sim.data.qpos);qvel=_tensor(env.sim.data.qvel)
    w,x,y,z=qpos[:,3:7].unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    privileged=torch.stack([qvel[:,0],qvel[:,1],yaw,qpos[:,1],qpos[:,0]-env._start_x,
        qpos[:,2],env._timed_steps.float()*CONTROL_DT,env._settle_remaining.float()/SETTLE_STEPS],-1)
    return torch.cat([obs,privileged],-1)


def _component(env,name):return env._race_components[name]
def progress_reward(env):return _component(env,'progress')
def finish_reward(env):return _component(env,'finish')
def time_cost(env):return _component(env,'time_cost')
def heading_cost(env):return _component(env,'heading_cost')
def lateral_cost(env):return _component(env,'lateral_cost')
def tilt_cost(env):return _component(env,'tilt_cost')
def stall_cost(env):return _component(env,'stall_cost')
def action_rate_cost(env):return _component(env,'action_rate_cost')
def fall_cost(env):return _component(env,'fall_cost')
def race_terminated(env):return env._race_terminated
def race_timeout(env):return env._race_timeout


# CPU guard tests import this module without GPU packages. Construction never falls back.
if importlib.util.find_spec('mjlab'):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from bam.mjlab import BamActuator,BamActuatorCfg
else:
    ManagerBasedRlEnv=object
    BamActuator=object
    BamActuatorCfg=object


class ArenaBamActuator(BamActuator):
    """Canonical BAM with per-world ring cursors and explicit selective reset drive."""
    def edit_spec(self,spec,target_names):
        super().edit_spec(spec,target_names)
        from arena.actuator import FORCE_LIMIT
        for act in self._mjs_actuators:
            act.forcerange=(-FORCE_LIMIT,FORCE_LIMIT)
            act.ctrllimited=False

    def initialize(self,*args,**kwargs):
        super().initialize(*args,**kwargs)
        self._ring=torch.zeros(self._num_envs,7,len(self._target_ids_list),device=self._device)
        self._heads=torch.full((self._num_envs,),-1,device=self._device,dtype=torch.long)
        self._filled=torch.zeros(self._num_envs,device=self._device,dtype=torch.bool)
        self.advance_counts=torch.zeros(self._num_envs,device=self._device,dtype=torch.long)
        self.last_lags=torch.zeros(self._num_envs,device=self._device,dtype=torch.long)
        self.fixed_lag=None

    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self._ring[ids]=0;self._heads[ids]=-1;self._filled[ids]=False
        self.advance_counts[ids]=0;self.last_lags[ids]=0
        # Official Servos construction clears friction before its first forward.
        model=self._mjwarp_model
        if model is not None:
            ids=torch.arange(self._num_envs,device=self._device) if env_ids is None else env_ids
            model.dof_frictionloss[ids[:,None],self._dof_ids[None,:]]=0.
            model.dof_damping[ids[:,None],self._dof_ids[None,:]]=self._bam_model.friction_viscous.value

    def delayed(self,cmd,ids=None):
        if ids is None:ids=torch.arange(self._num_envs,device=self._device)
        fresh=ids[~self._filled[ids]]
        if len(fresh):self._ring[fresh]=cmd.position_target[fresh,None,:]
        self._heads[ids]=(self._heads[ids]+1)%7
        self._ring[ids,self._heads[ids]]=cmd.position_target[ids]
        self._filled[ids]=True
        lag=torch.randint(3,7,(len(ids),),device=self._device) if self.fixed_lag is None else torch.full((len(ids),),self.fixed_lag,device=self._device,dtype=torch.long)
        self.last_lags[ids]=lag;self.advance_counts[ids]+=1
        delayed=cmd.position_target.clone()
        delayed[ids]=self._ring[ids,(self._heads[ids]-lag)%7]
        return dataclasses.replace(cmd,position_target=delayed)

    def apply_delay(self,cmd):return self.delayed(cmd)

    def compute_subset(self,cmd,ids):
        """Same canonical BAM equations, writing only reset worlds' state/controls."""
        bam=self._bam_model;act=bam.actuator;dofs=self._dof_ids
        vin=self.vin_tensor[ids]-self.vin_drop_gain[ids]*self._prev_motor_torque[ids].abs().sum(-1,keepdim=True)
        vin=vin.clamp_min(self.cfg.vin_min)
        act.vin=vin;act.kp=self._base_kp*self.kp_scale[ids]
        vel=cmd.vel[ids];scaled=vel*self.kd_scale[ids]
        control=act.compute_control(cmd.position_target[ids],cmd.pos[ids],scaled,self._dt)
        torque=act.compute_torque(control,True,cmd.pos[ids],scaled)
        qfrc_fric=self._dof_friction_force(self._as_tensor(self._data.qfrc_bias).shape[-1])
        ext=(-self._as_tensor(self._data.qfrc_bias)[ids][:,dofs]
             +self._as_tensor(self._data.qfrc_constraint)[ids][:,dofs]-qfrc_fric[ids][:,dofs])
        mot=self._as_tensor(self._data.qfrc_actuator)[ids][:,dofs]
        stribeck=torch.exp(-torch.pow(vel.abs()/bam.dtheta_stribeck.value,bam.alpha.value))
        friction=self._compute_friction_budget(mot,ext,stribeck)
        self._mjwarp_model.dof_frictionloss[ids[:,None],dofs[None,:]]=friction
        self._mjwarp_model.dof_damping[ids[:,None],dofs[None,:]]=bam.friction_viscous.value
        self._prev_motor_torque[ids]=torque.detach()
        return torque


if BamActuatorCfg is not object:
    @dataclasses.dataclass(kw_only=True)
    class ArenaBamActuatorCfg(BamActuatorCfg):
        def build(self,entity,target_ids,target_names):return ArenaBamActuator(self,entity,target_ids,target_names)


def arena_spec():
    import mujoco
    from arena.course import build_course_xml
    from arena.manifest import load_event
    xml,assets=build_course_xml(load_event('roller-sprint-2m'))
    return mujoco.MjSpec.from_string(xml,assets=assets)


def make_fresh_env_cfg(num_envs=64,seed=15001):
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg
    from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
    from mjlab.sim import SimulationCfg,MujocoCfg
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers import ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg
    from arena.course import compile_course
    from arena.manifest import load_event
    reference_model=compile_course(load_event('roller-sprint-2m'))
    reference_opt=reference_model.opt
    bam=ArenaBamActuatorCfg(target_names_expr=JOINT_NAMES,motor_name='xl330',model='m6',
        kp_fw=200.,vin=7.35,vin_drop_gain_range=(.1,.1),vin_min=6.,delay_min_lag=0,delay_max_lag=0)
    entity=EntityCfg(spec_fn=arena_spec,init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.12),
        joint_pos=dict(zip(JOINT_NAMES,DEFAULT_POSE.tolist())),joint_vel={'.*':0.}),
        articulation=EntityArticulationInfoCfg(actuators=(bam,),soft_joint_pos_limit_factor=.9))
    cfg=ManagerBasedRlEnvCfg(decimation=4,seed=seed,episode_length_s=40.5,
        scene=SceneCfg(num_envs=num_envs,terrain=None,entities={'robot':entity}),
        sim=SimulationCfg(nconmax=256,njmax=2048,mujoco=MujocoCfg(timestep=.005,
            integrator='implicitfast',iterations=10,ls_iterations=20,
            impratio=float(reference_opt.impratio),cone=('pyramidal','elliptic')[int(reference_opt.cone)],
            solver=('pgs','cg','newton')[int(reference_opt.solver)],
            tolerance=float(reference_opt.tolerance),ls_tolerance=float(reference_opt.ls_tolerance),
            jacobian=('dense','sparse','auto')[int(reference_opt.jacobian)])),
        observations={'actor':ObservationGroupCfg(terms={'arena61':ObservationTermCfg(func=actor_observation)},nan_policy='error'),
            'critic':ObservationGroupCfg(terms={'arena69':ObservationTermCfg(func=critic_observation)},nan_policy='error')},
        actions={'joint_pos':JointPositionActionCfg(entity_name='robot',actuator_names=JOINT_NAMES,
            scale=1.,preserve_order=True,use_default_offset=True)},events={},commands={},curriculum={},
        rewards={'progress':RewardTermCfg(func=progress_reward,weight=10.),
                 'finish':RewardTermCfg(func=finish_reward,weight=5.),
                 'time_cost':RewardTermCfg(func=time_cost,weight=-1.),
                 'heading_cost':RewardTermCfg(func=heading_cost,weight=-.5),
                 'lateral_cost':RewardTermCfg(func=lateral_cost,weight=-2.),
                 'tilt_cost':RewardTermCfg(func=tilt_cost,weight=-.5),
                 'stall_cost':RewardTermCfg(func=stall_cost,weight=-.1),
                 'action_rate_cost':RewardTermCfg(func=action_rate_cost,weight=-.005),
                 'fall_cost':RewardTermCfg(func=fall_cost,weight=-5.)},
        terminations={'race_done':TerminationTermCfg(func=race_terminated),
                      'race_timeout':TerminationTermCfg(func=race_timeout,time_out=True)},
        auto_reset=True,scale_rewards_by_dt=True)
    return cfg


class FreshSprintEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,device='cuda:0',render_mode=None):
        if ManagerBasedRlEnv is object:raise ImportError('Pinned MJLab CUDA runtime required; no CPU fallback')
        if not device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        self._arena_ready=False
        super().__init__(cfg,device=device,render_mode=render_mode)
        import mujoco
        model=self.sim.mj_model
        if (model.nq,model.nv,model.nu)!=(25,24,14):raise ValueError('Arena roller dimensions differ')
        self._qpos=torch.tensor([model.joint('robot/'+n).qposadr[0] for n in JOINT_NAMES],device=device,dtype=torch.long)
        self._qvel=torch.tensor([model.joint('robot/'+n).dofadr[0] for n in JOINT_NAMES],device=device,dtype=torch.long)
        self._trunk=model.body('robot/trunk_base').id
        self._gyro=int(model.sensor('robot/imu_ang_vel').adr[0])
        self._default=torch.tensor(DEFAULT_POSE,device=device,dtype=torch.float32)
        from arena.course import compile_course
        from arena.manifest import load_event
        source_model=compile_course(load_event('roller-sprint-2m'))
        source_stand=source_model.key('STAND').qpos
        stand=np.zeros(model.nq,dtype=np.float64);stand[:7]=source_stand[:7]
        for joint in range(source_model.njnt):
            name=source_model.joint(joint).name
            if name and source_model.joint(joint).type[0]!=mujoco.mjtJoint.mjJNT_FREE:
                destination=int(model.joint('robot/'+name).qposadr[0])
                stand[destination]=source_stand[int(source_model.jnt_qposadr[joint])]
        self._stand_qpos=torch.tensor(stand,device=device,dtype=torch.float32)
        if not torch.allclose(self._stand_qpos[self._qpos],self._default,atol=1e-7,rtol=0):raise ValueError('ArenaSTAND/defaultmetadata mismatch')
        self.sim.expand_model_fields(('dof_frictionloss','dof_damping'))
        self._act=self.scene['robot'].actuators[0]
        if not isinstance(self._act,ArenaBamActuator):raise TypeError('Canonical local Arena BAM required')
        self._previous_actual=torch.zeros(self.num_envs,14,device=device)
        self._lagged_vel=torch.zeros_like(self._previous_actual)
        self._command_vx=torch.zeros(self.num_envs,device=device)
        self._settle_remaining=torch.full((self.num_envs,),25,device=device,dtype=torch.long)
        self._timed_steps=torch.zeros_like(self._settle_remaining)
        self._fall_steps=torch.zeros_like(self._settle_remaining)
        self._start_x=torch.zeros(self.num_envs,device=device)
        self._race_terminated=torch.zeros(self.num_envs,device=device,dtype=torch.bool)
        self._race_timeout=self._race_terminated.clone()
        self._race_components={key:torch.zeros(self.num_envs,device=device) for key in ('progress','finish',*PENALTY_TERMS)}
        self._arena_ready=True

    def _reset_idx(self,env_ids=None):
        if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
        super()._reset_idx(env_ids)
        qpos=_tensor(self.sim.data.qpos);qvel=_tensor(self.sim.data.qvel)
        qpos[env_ids]=self._stand_qpos;qvel[env_ids]=0.
        self._last_reset_force_clear={}
        for field in ('ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','qfrc_actuator','qfrc_constraint'):
            if hasattr(self.sim.data,field):
                values=_tensor(getattr(self.sim.data,field));values[env_ids]=0.
                self._last_reset_force_clear[field]=values[env_ids].clone()
        yaw=torch.empty(len(env_ids),device=self.device).uniform_(-math.pi/90,math.pi/90)
        qpos[env_ids,3]=torch.cos(yaw/2);qpos[env_ids,6]=torch.sin(yaw/2)
        noise=torch.empty(len(env_ids),14,device=self.device).uniform_(-.01,.01)
        qpos[env_ids[:,None],self._qpos[None,:]]=self._default+noise
        self._previous_actual[env_ids]=0.;self._lagged_vel[env_ids]=0.
        self._command_vx[env_ids]=torch.empty(len(env_ids),device=self.device).uniform_(.3,2.)
        self._settle_remaining[env_ids]=25;self._timed_steps[env_ids]=0;self._fall_steps[env_ids]=0
        self._start_x[env_ids]=0.;self._race_terminated[env_ids]=False;self._race_timeout[env_ids]=False
        # Fresh reset gets one zero-target drive, never an extra non-reset drive.
        self.sim.forward()
        cmd=self._act.get_command(self.scene['robot'].data)
        cmd=self._act.delayed(cmd,env_ids)
        torque=self._act.compute_subset(cmd,env_ids)
        _tensor(self.sim.data.ctrl)[env_ids[:,None],self._act.global_ctrl_ids[None,:]]=torque
        self.sim.forward()

    def reset(self,*,seed=None,env_ids=None,options=None):
        if seed is not None:self.seed(seed)
        if env_ids is None:env_ids=torch.arange(self.num_envs,device=self.device)
        self._reset_idx(env_ids);self.sim.sense()
        self.obs_buf=self.observation_manager.compute(update_history=True)
        return self.obs_buf,self.extras

    def step(self,action):
        action=action.to(self.device)
        if action.shape!=(self.num_envs,14) or not torch.isfinite(action).all():raise ValueError('Finite actualNx14 action required')
        before=_tensor(self.sim.data.qpos)[:,0].clone()
        old_action=self._previous_actual.clone()
        next_lagged=_tensor(self.sim.data.qvel)[:,self._qvel].clone()
        settling=self._settle_remaining>0
        self.action_manager.process_action(action)
        for _ in range(4):
            self._sim_step_counter+=1;self.action_manager.apply_action()
            self.scene.write_data_to_sim();self.sim.step();self.scene.update(dt=self.physics_dt)
        self.scene.write_data_to_sim()  # fifth BAM write, with the same current targets
        self.sim.forward();self.sim.sense()
        self._previous_actual.copy_(action);self._lagged_vel.copy_(next_lagged)
        self.episode_length_buf+=1;self.common_step_counter+=1
        self._settle_remaining.clamp_min_(0)
        end_settle=self._settle_remaining==1
        self._settle_remaining-=settling.long()
        qpos=_tensor(self.sim.data.qpos)
        self._start_x[end_settle]=qpos[end_settle,0]
        timed=~settling;self._timed_steps+=timed.long()
        obs=actor_observation(self);up=-obs[:,5]
        self._fall_steps=torch.where(timed&(up<.5),self._fall_steps+1,torch.zeros_like(self._fall_steps))
        finite=torch.isfinite(qpos).all(-1)&torch.isfinite(_tensor(self.sim.data.qvel)).all(-1)
        fell=(self._fall_steps>=15)|~finite
        progress=qpos[:,0]-self._start_x
        finished=timed&(progress>=2.)&~fell
        self._race_terminated=timed&(fell|finished)
        self._race_timeout=timed&(self._timed_steps>=2000)&~self._race_terminated
        w,x,y,z=qpos[:,3:7].unbind(-1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        dx=qpos[:,0]-before
        gate=timed&(up>=.5)&(qpos[:,1].abs()<.35)
        self._race_components={'progress':torch.where(gate,dx,torch.minimum(dx,torch.zeros_like(dx)))*timed/CONTROL_DT,
             'finish':finished.float()/CONTROL_DT,'time_cost':timed.float(),'heading_cost':(1-torch.cos(yaw))*timed,
             'lateral_cost':qpos[:,1].square()*timed,'tilt_cost':(1-up).clamp_min(0),
             'stall_cost':((dx/CONTROL_DT)<.05).float()*timed,
             'action_rate_cost':(action-old_action).square().sum(-1)*timed,
             'fall_cost':(fell&timed).float()/CONTROL_DT}
        self.reset_buf=self.termination_manager.compute()
        self.reset_terminated=self.termination_manager.terminated.clone()
        self.reset_time_outs=self.termination_manager.time_outs.clone()
        self.reward_buf=self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()
        ids=self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.cfg.auto_reset and len(ids):self._reset_idx(ids);self.sim.sense()
        self.obs_buf=self.observation_manager.compute(update_history=True)
        self.extras['race']={'finishes':int(finished.sum()),'falls':int((fell&timed).sum()),
                           'settling_worlds':int(settling.sum()),'nonfinite_states':int((~finite).sum())}
        return self.obs_buf,self.reward_buf,self.reset_terminated,self.reset_time_outs,self.extras

    def domain_receipt(self):return self.contract_report()

    def contract_report(self):
        return {'actor_observations':61,'critic_observations':69,'actions':14,
          'physics_dt_s':.005,'control_dt_s':.02,'settle_steps':25,
          'drive_schedule':list(control_schedule()),'BAM_drive_calls_per_control':5,
          'per_world_selective_reset_drive':True,'joint_noise_rad':[-.01,.01],
          'yaw_noise_deg':[-2.,2.],'command_vx_range':[.3,2.],
          'unused_command_slots_zero':12,'previous_action':'actual submitted output',
          'joint_velocity_observation':'one control step old','reset_pose_source':'exactArenacompiledSTANDkeymappedbyjointname',
          'reset_force_state_clear_fields':['ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','qfrc_actuator','qfrc_constraint'],
          'termination_priority':'fall before finish on samecontrolstep',
          'geometry':'official Arena XML collisionrobot and exact physical lane walls, entityname prefixonly',
          'nominal_BAM':{'vin':7.35,'drop_gain':.1,'vin_min':6.,'kp_fw':200.,'delay_lags':[3,6]},
          'model_dimensions':[self.sim.mj_model.nq,self.sim.mj_model.nv,self.sim.mj_model.nu],
          'GPU_surrogate_differences':['FP32physics/friction','Torch delay RNG distribution matching, not NumPy seed-bit parity'],
          'upstream_packages_modified':False}
