#!/usr/bin/env python3
"""Fresh CUDA PPO policy for the MicroDuck Arena Roller Sprint.

No previous actor, V89 weights, teacher, history adapter or GPU policy graph is
loaded.  A fresh normalized 61D actor, fresh critic and new Adam optimizer train
inside the source-reviewed nominal Arena environment in adjacent env.py.
The official CPU Arena judge evaluates every checkpoint; only learning uses
CUDA.  This script never publishes or submits an entry.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from checkpoint_state import encode_numpy_rng, decode_numpy_rng

EXPERIMENT = 'duckwing-arena-fresh-sprint-r1'
SOURCE_REVISION = 'a3e0c0fbe6c4574fa8b6fb203cd81f363c75e9be'
HORIZON = 24
INITIAL_STD_RAD = 0.35
LEARNING_RATE = 3e-4
EXPORT_ATOL = 2e-5
PHASE_BOUNDS = {'smoke':(64,5,900,5), 'capacity':(4096,2,900,2),
                'train':(4096,3000,7200,250)}
EXPECTED_VERSIONS = {'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1',
                     'warp-lang':'1.12.0','rsl-rl-lib':'5.0.1','torch':'2.9.1'}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_digest(module) -> str:
    result = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def finite(tensor, description: str) -> None:
    import torch
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"Nonfinite {description}")


def cuda(tensor, description: str) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError(f"CPU fallback for {description}: {tensor.device}")


def safe_actor_observations(obs):
    """Match Arena's actor command exactly; never expose relative heading."""
    if obs.ndim != 2 or obs.shape[1] != 61:
        raise ValueError(f"Actor observation shape differs: {tuple(obs.shape)}")
    result = obs.clone()
    # Twist vx lives at48; all other twist/head/body commands are zero in Arena.
    result[:, 49:] = 0.0
    return result


def validate_runtime() -> dict:
    import torch
    import warp as wp
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        if observed[name].split("+")[0] != expected:
            raise RuntimeError(f"Pinned runtime mismatch: {name}={observed[name]}, expected{expected}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Official challenge requires Python3.12")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA unavailable; CPU learning fallback is forbidden")
    # Avoid TF32 perturbing the frozen graph parity check.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    wp.init()
    a = torch.randn(32, 32, device="cuda", requires_grad=True)
    (a @ a.T).square().mean().backward()
    finite(a.grad, "CUDA tensor-smoke gradient")
    cuda(a.grad, "CUDA tensor-smoke gradient")
    torch.cuda.synchronize()
    return {
        "versions": observed, "python": sys.version, "architecture": platform.machine(),
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_tensor_backward_passed": True, "learning_cpu_fallback_allowed": False,
    }


def phase_configuration(phase, num_envs=None, iterations=None, budget_s=None,
                        checkpoint_interval=None):
    defaults=PHASE_BOUNDS[phase]
    values=[num_envs,iterations,budget_s,checkpoint_interval]
    values=[default if value is None else value for value,default in zip(values,defaults)]
    n,i,b,c=values
    if not(1<=n<=defaults[0] and 1<=i<=defaults[1] and 1<=b<=defaults[2] and 1<=c<=i):
        raise ValueError('Requested training exceeds bounded phase contract')
    if phase=='smoke' and(n,i)!=(64,5):
        raise ValueError('Required smoke is64worlds/5updates')
    if phase=='capacity' and n not in(1024,2048,4096):
        raise ValueError('Capacity probe world count must be1024,2048,4096')
    return tuple(values)


def create_models(obs, device):
    import torch
    from rsl_rl.models import MLPModel
    class MaskedExport(torch.nn.Module):
        def __init__(self, exported):
            super().__init__()
            self.exported=exported
        def forward(self,x):
            return self.exported(torch.cat((x[:,:49],torch.zeros_like(x[:,49:])),dim=1))
        def get_dummy_inputs(self): return self.exported.get_dummy_inputs()
        @property
        def input_names(self): return self.exported.input_names
        @property
        def output_names(self): return self.exported.output_names
    class FreshArenaActor(MLPModel):
        def as_onnx(self, verbose=False):
            return MaskedExport(super().as_onnx(verbose))
    groups={'actor':['actor'],'critic':['critic']}
    actor=FreshArenaActor(obs,groups,'actor',14,hidden_dims=(512,256,128),
        activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':INITIAL_STD_RAD,
                          'std_type':'log'}).to(device)
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=(512,256,128),activation='elu',
                    obs_normalization=True).to(device)
    return actor,critic


def mask_obs(obs):
    obs=obs.clone()
    obs['actor']=safe_actor_observations(obs['actor'])
    for name,value in obs.items():
        cuda(value,f'{name} observation')
        finite(value,f'{name} observation')
    if (obs['actor'][:,49:]!=0).any().item():
        raise RuntimeError('Unused/privileged actor command leaked')
    return obs


def interface_report(env):
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    joint=env.action_manager.get_term('joint_pos')
    names=list(joint.target_names)
    if len(names)!=14 or any('passive_'in name for name in names):
        raise RuntimeError('Action selection is not14actuatedjoints')
    if joint.cfg.clip is not None:
        raise RuntimeError('Train-only action clipping is forbidden')
    defaults=joint.offset[0].detach().cpu().numpy()
    scales=(joint.scale[0].detach().cpu().numpy() if isinstance(joint.scale,torch.Tensor)
            else np.full(14,float(joint.scale)))
    if len(defaults)!=14 or not np.isfinite(defaults).all():
        raise RuntimeError('Invalid actual default joint positions')
    if not np.allclose(scales,1.,atol=1e-6,rtol=0):
        raise RuntimeError('Arena action scaling must equal1')
    if abs(env.step_dt-.02)>1e-9 or abs(env.physics_dt-.005)>1e-9:
        raise RuntimeError('Control/physics cadence differs fromArena50/200Hz')
    m=env.sim.mj_model
    wheels=[]
    for j in range(m.njnt):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
        if name and 'passive_'in name and 'wheel'in name:
            friction=float(m.dof_frictionloss[m.jnt_dofadr[j]])
            if friction!=0.: raise RuntimeError('Wheel friction differs fromArena')
            wheels.append({'name':name,'frictionloss':friction})
    if len(wheels)!=4: raise RuntimeError('Expected four passive wheels')
    classes=[type(a).__module__+'.'+type(a).__name__ for a in env.scene['robot'].actuators]
    if not any('bam'in x.lower() for x in classes): raise RuntimeError('BAMactuator missing')
    qpos=wp.to_torch(env.sim.wp_data.qpos)
    cuda(qpos,'MuJoCo-Warp positions');finite(qpos,'initial physical positions')
    # Use actual actuator metadata, excluding passive joints. The Arena itself
    # uses its canonical robot/motor constants; these fields enable inspection.
    act_ids=[]
    for name in names:
        matches=[i for i in range(m.nu)
                 if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,
                                      int(m.actuator_trnid[i,0])) or '').split('/')[-1]==name]
        if len(matches)!=1: raise RuntimeError(f'Actuator mapping ambiguous:{name}')
        act_ids.append(matches[0])
    return {'joint_names':names,'default_joint_pos':defaults.tolist(),
        'action_scale':scales.tolist(),'actor_observation_dim':61,'action_dim':14,
        'observation_names':list(env.observation_manager.active_terms['actor']),
        'command_names':['twist'],'control_dt_s':env.step_dt,'physics_dt_s':env.physics_dt,
        'wheel_joints':wheels,'actuator_classes':classes,
        'joint_stiffness':m.actuator_gainprm[act_ids,0].tolist(),
        'joint_damping':(-m.actuator_biasprm[act_ids,2]).tolist(),
        'physical_tensor_device':str(qpos.device),'actor_heading_position_privilege':False,
        'pretrained_actor_loaded':False,'train_only_action_clipping':False,
        'environment_source_sha256':sha256(Path(inspect.getfile(type(env))))}


def actor_mean(actor,samples):
    from tensordict import TensorDict
    return actor(TensorDict({'actor':samples},batch_size=[len(samples)]))


def export_policy(actor,destination,interface,samples):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    if destination.exists(): raise FileExistsError('Prior export must bepreserved')
    # Official RSL exporter contains BOTH current empirical normalization and
    # actual network weights. Never export actor.mlp alone.
    with torch.inference_mode():
        exported=actor.as_onnx(verbose=False).to('cpu').eval()
    with torch.no_grad():
        torch.onnx.export(exported,exported.get_dummy_inputs(),str(destination),
            export_params=True,opset_version=18,dynamo=False,
            input_names=exported.input_names,output_names=exported.output_names)
    model=onnx.load(str(destination))
    metadata={key:(','.join(map(str,interface[key])) if isinstance(interface[key],list)
                   else str(interface[key])) for key in
              ['joint_names','default_joint_pos','action_scale','joint_stiffness','joint_damping',
               'command_names','observation_names']}
    metadata.update({'run_path':'None','duckwing_training_initialization':'fresh_random',
        'duckwing_actor_observations':'61','duckwing_control_frequency_hz':'50',
        'duckwing_observation_normalizer':'baked_current_rsl_empirical_normalization',
        'duckwing_experiment':EXPERIMENT,'duckwing_source_revision':SOURCE_REVISION,
        'duckwing_pretrained_actor_loaded':'false','duckwing_release_promoted':'false'})
    onnx.helper.set_model_props(model,metadata)
    onnx.checker.check_model(model);onnx.save(model,str(destination))
    if destination.stat().st_size>64*1024*1024: raise RuntimeError('Export exceedsArena64MiB')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(destination),sess_options=options,
                                 providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape!=[1,61] or session.get_outputs()[0].shape!=[1,14]:
        raise RuntimeError('Export61D/14D interface changed')
    maximum=0.
    with torch.no_grad():
        for x in samples[:32].split(1):
            expected=actor_mean(actor,x).cpu().numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x.cpu().numpy()})[0]
            if not np.isfinite(actual).all(): raise RuntimeError('Nonfiniteexportedaction')
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
    if maximum>EXPORT_ATOL: raise RuntimeError(f'NormalizedCUDA/ONNXparityfailed:{maximum}')
    return {'path':destination.name,'bytes':destination.stat().st_size,'sha256':sha256(destination),
        'normalized_cuda_actor_vs_onnx_max_abs':maximum,'atol':EXPORT_ATOL,'passed':True,
        'normalizer_baked':True,'pretrained_actor_loaded':False}


def judge_checkpoint(args,output,iteration,policy):
    if args.judge_contract is None: return None
    root=args.judge_contract.resolve().parent;policy=policy.resolve()
    if not policy.is_relative_to(root): raise ValueError('Policy escapesjudgecontractroot')
    contract=json.loads(args.judge_contract.read_text())
    candidate=f'fresh-{args.phase}-{iteration}'
    contract['candidates']={candidate:{'path':policy.relative_to(root).as_posix(),
        'sha256':sha256(policy),'provenance':'Fresh61Dcompetitionactor; no pretrained weights'}}
    path=root/f'judge-fresh-{args.phase}-{iteration}-contract.json'
    if path.exists(): raise FileExistsError('Refuse priorjudgecontractoverwrite')
    write_json(path,contract)
    dest=output/f'judge-{iteration}'
    if not args.judge_python.is_file() or not args.judge_script.is_file():
        raise FileNotFoundError('OfficialCPUjudgepython/scriptmissing')
    started=time.monotonic()
    print(json.dumps({'phase':'checkpoint_cpu_judge_start','iteration':iteration,
                      'cpu_qualification_exception':True}),flush=True)
    env=os.environ.copy();env.update({k:'1'for k in
        ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    with(output/f'judge-{iteration}.log').open('x')as log:
        result=subprocess.run([str(args.judge_python),str(args.judge_script),'--contract',str(path),
            '--candidate',candidate,'--output',str(dest)],env=env,stdout=log,
            stderr=subprocess.STDOUT,timeout=int(contract.get('budget_s',600))+120)
    summary=json.loads((dest/'summary.json').read_text())
    report={'iteration':iteration,'exit_code':result.returncode,'elapsed_s':time.monotonic()-started,
        'evidence_valid':summary.get('evidence_valid',False),'score_s':summary.get('score_s'),
        'seeds_finished':summary.get('seeds_finished'),'command_vx':summary.get('command_vx'),
        'summary':str(dest.relative_to(output)/'summary.json'),'official_space_score':False,
        'cpu_qualification_exception':'OfficialCPUjudge; alllearningCUDA','submitted':False}
    print(json.dumps({'phase':'checkpoint_cpu_judge_complete',**report}),flush=True)
    if result.returncode!=0 or not report['evidence_valid']:
        raise RuntimeError(f'Invalidofficialevaluation:{report}')
    # Random initial policy is allowed to fall. This is evidence qualification,
    # not a requirement to inherit the previous teacher's score.
    return report


def checkpoint(alg,env,args,output,iteration,interface,samples):
    import numpy as np
    import torch
    path=output/f'model_{iteration}.pt'
    if path.exists(): raise FileExistsError('Refusecheckpointoverwrite')
    data=alg.save();data.update({'iter':iteration,'common_step_counter':env.common_step_counter,
        'seed':args.seed,'initialization':'fresh_random','pretrained_actor_loaded':False,
        'ppo_learning_rate':alg.learning_rate,'rng_python':random.getstate(),
        'rng_numpy':encode_numpy_rng(np.random.get_state()),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all()})
    torch.save(data,path)
    with torch.no_grad():before=actor_mean(alg.actor,samples).clone()
    optimizer_before=copy.deepcopy(alg.optimizer.state_dict())
    loaded=torch.load(path,map_location='cuda:0',weights_only=True)
    if loaded['initialization']!='fresh_random' or loaded['pretrained_actor_loaded']:
        raise RuntimeError('Unexpectedpretrainedcheckpointprovenance')
    with torch.inference_mode():alg.load(loaded,load_cfg=None,strict=True)
    alg.learning_rate=loaded['ppo_learning_rate']
    with torch.no_grad():error=float((actor_mean(alg.actor,samples)-before).abs().max().item())
    if error!=0.:raise RuntimeError('Strictcheckpointnormalizedactorparityfailed')
    for key,state in optimizer_before['state'].items():
        for name,value in state.items():
            restored=alg.optimizer.state_dict()['state'][key][name]
            if isinstance(value,torch.Tensor)and not torch.equal(value.to(restored.device),restored):
                raise RuntimeError(f'Optimizerrestoremismatch:{name}')
    random.setstate(loaded['rng_python']);np.random.set_state(decode_numpy_rng(loaded['rng_numpy']))
    torch.set_rng_state(loaded['rng_torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu()for x in loaded['rng_cuda']])
    export=export_policy(alg.actor,output/f'policy_{iteration}.onnx',interface,samples)
    judge=judge_checkpoint(args,output,iteration,output/export['path'])
    return {'iteration':iteration,'checkpoint':path.name,'sha256':sha256(path),
        'restore_normalized_actor_max_abs':error,'optimizer_state_restored':True,
        'rng_restored':True,'physical_world_state_resume_claimed':False,'export':export,'judge':judge}


def save_trace(output,trace):
    import numpy as np
    path=output/'sampled-training-trace.npz'
    if trace and not path.exists():
        np.savez_compressed(path,**{key:np.stack([row[key]for row in trace])for key in trace[0]})


def run(args,output):
    import numpy as np
    import torch
    import warp as wp
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    from mjlab.rl import RslRlVecEnvWrapper
    # Source-owned competition environment; installed upstream stays untouched.
    import env as competition_env
    runtime=validate_runtime()
    runtime.update({'experiment':EXPERIMENT,'pretrained_actor_loaded':False,
        'actor_architecture':[61,512,256,128,14],'actor_normalization':True,
        'initial_log_gaussian_std_rad':INITIAL_STD_RAD,'learning_rate':LEARNING_RATE,
        'learning_rate_schedule':'fixed','ppo_implementation':'rsl_rl.algorithms.PPO5.0.1'})
    free,total=torch.cuda.mem_get_info();runtime['gpu_memory_before_env']={'free':free,'total':total}
    write_json(output/'runtime.json',runtime)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    cfg=competition_env.make_fresh_env_cfg(num_envs=args.num_envs,seed=args.seed)
    env=competition_env.FreshSprintEnv(cfg=cfg,device='cuda:0',render_mode=None)
    wrapped=None;trace=[]
    try:
        wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
        obs=mask_obs(wrapped.get_observations())
        interface=interface_report(env)
        write_json(output/'interface.json',interface)
        domain=env.contract_report()
        write_json(output/'domain-receipt.json',domain)
        actor,critic=create_models(obs,'cuda:0')
        initial_weights={k:v.detach().clone()for k,v in actor.mlp.state_dict().items()}
        initial_digest=state_digest(actor)
        write_json(output/'initialization.json',{'kind':'fresh_random','seed':args.seed,
            'actor_state_sha256':initial_digest,'pretrained_actor_loaded':False,
            'teacher_loaded':False,'normalizer_fresh':True,'critic_fresh':True,'optimizer_fresh':True})
        storage=RolloutStorage('rl',args.num_envs,HORIZON,obs,[14],'cuda:0')
        alg=PPO(actor,critic,storage,device='cuda:0',num_learning_epochs=5,num_mini_batches=4,
            learning_rate=LEARNING_RATE,entropy_coef=.01,clip_param=.2,gamma=.99,lam=.95,
            desired_kl=.01,schedule='fixed',max_grad_norm=1.,use_clipped_value_loss=True,
            value_loss_coef=1.,symmetry_cfg=None)
        gradients={'hook_calls':0,'max_abs':0.,'finite':True,'all_cuda':True}
        def hook(gradient):
            cuda(gradient,'freshactorgradient');finite(gradient,'freshactorgradient')
            gradients['hook_calls']+=1
            gradients['max_abs']=max(gradients['max_abs'],float(gradient.abs().max().item()))
            return gradient
        for p in actor.mlp.parameters():p.register_hook(hook)
        for name,p in list(actor.named_parameters())+list(critic.named_parameters()):
            cuda(p,f'actor/criticparameter{name}')
        runtime['source_sha256']={'train.py':sha256(Path(__file__)),
            'env.py':sha256(Path(inspect.getfile(competition_env))),
            'rsl_ppo.py':sha256(Path(inspect.getfile(PPO)))}
        free,total=torch.cuda.mem_get_info()
        runtime['gpu_memory_after_env_models']={'free':free,'total':total}
        runtime['physics_cuda']=True;runtime['policy_cuda']=True
        write_json(output/'runtime.json',runtime)
        reward_contract={name:{'weight':float(term.weight),
            'function':term.func.__module__+'.'+getattr(term.func,'__name__',type(term.func).__name__)}
            for name,term in cfg.rewards.items()}
        write_json(output/'reward-contract.json',{'terms':reward_contract,
            'source_reviewed_intentional_bundle':'Freshpolicy plus nominalArena startup/drive chronology plus raceprogressobjective',
            'no_pretrained_actor':True,'train_forward_range':domain['command_vx_range']})
        samples=obs['actor'][:32].detach().clone()
        checkpoints=[checkpoint(alg,env,args,output,0,interface,samples)]
        updates=0;terminals=0;budget_hit=False;judgement_s=0.;started=time.monotonic()
        penalties=set(getattr(competition_env,'PENALTY_TERMS',()))
        penalty_max={name:-math.inf for name in penalties if name in env.reward_manager.active_terms}
        if not penalty_max:raise RuntimeError('No explicit penalty signs provided bycompetitionenv')
        alg.train_mode()
        with(output/'iterations.jsonl').open('x')as log:
            for iteration in range(1,args.iterations+1):
                if time.monotonic()-started-judgement_s>=args.budget_s:
                    budget_hit=True;break
                rollout_reward=0.
                with torch.inference_mode():
                    for step in range(HORIZON):
                        actions=alg.act(obs);cuda(actions,'freshactoractions');finite(actions,'actions')
                        finite(alg.transition.actions_log_prob,'Gaussianlogprob')
                        for p in alg.transition.distribution_params:
                            cuda(p,'distributionparameters');finite(p,'distributionparameters')
                        if(alg.transition.distribution_params[1]<=0).any().item():
                            raise RuntimeError('Nonpositiveexplorationstd')
                        prior=obs['actor'][:4].clone()
                        obs,reward,done,extras=wrapped.step(actions)
                        if extras.get('race',{}).get('nonfinite_states',0):
                            raise RuntimeError('Nonfinite physical state detected before autoreset')
                        obs=mask_obs(obs)
                        cuda(reward,'racingreward');finite(reward,'racingreward')
                        qpos=wp.to_torch(env.sim.wp_data.qpos);qvel=wp.to_torch(env.sim.wp_data.qvel)
                        cuda(qpos,'physicalpositions');cuda(qvel,'physicalvelocities')
                        finite(qpos,'physicalpositions');finite(qvel,'physicalvelocities')
                        if 'nan_state'in env.termination_manager.active_terms:
                            if env.termination_manager.get_term('nan_state').any().item():
                                raise RuntimeError('NaNstatewasautoreset')
                        weighted=env.reward_manager._step_reward;finite(weighted,'weightedrewards')
                        for name in penalty_max:
                            index=env.reward_manager.active_terms.index(name)
                            maximum=float(weighted[:,index].max().item())
                            penalty_max[name]=max(penalty_max[name],maximum)
                            if maximum>1e-6:raise RuntimeError(f'Double-negatedpenalty{name}={maximum}')
                        terminals+=int(done.sum().item());rollout_reward+=float(reward.mean().item())
                        alg.process_env_step(obs,reward,done,extras)
                        if len(trace)<120:
                            trace.append({'actor_obs':prior.cpu().numpy(),
                                'actions':actions[:4].cpu().numpy(),'reward':reward[:4].cpu().numpy(),
                                'dones':done[:4].cpu().numpy(),'qpos':qpos[:4].cpu().numpy().copy(),
                                'qvel':qvel[:4].cpu().numpy().copy()})
                        samples=obs['actor'][:32].clone()
                    alg.compute_returns(obs)
                losses=alg.update()
                if any(not math.isfinite(float(x))for x in losses.values()):
                    raise RuntimeError('NonfinitePPOloss')
                for name,p in list(actor.named_parameters())+list(critic.named_parameters()):
                    finite(p,f'updatedparameter{name}')
                for state in alg.optimizer.state.values():
                    for name,value in state.items():
                        if isinstance(value,torch.Tensor):
                            finite(value,f'Adam{name}')
                            if name!='step':cuda(value,f'Adam{name}')
                updates=iteration
                row={'iteration':iteration,'transitions':iteration*args.num_envs*HORIZON,
                    'elapsed_learning_s':time.monotonic()-started-judgement_s,
                    'mean_rollout_reward':rollout_reward/HORIZON,'losses':losses,
                    'learning_rate':alg.learning_rate,'actor_gradient_max_abs':gradients['max_abs'],
                    'terminals_total':terminals,'common_step_counter':env.common_step_counter}
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if iteration<=5 or iteration%25==0:print(json.dumps(row),flush=True)
                if iteration%args.checkpoint_interval==0 or iteration==args.iterations:
                    checkpoints.append(checkpoint(alg,env,args,output,iteration,interface,samples))
                    if checkpoints[-1]['judge']:judgement_s+=checkpoints[-1]['judge']['elapsed_s']
            if updates and checkpoints[-1]['iteration']!=updates:
                checkpoints.append(checkpoint(alg,env,args,output,updates,interface,samples))
                if checkpoints[-1]['judge']:judgement_s+=checkpoints[-1]['judge']['elapsed_s']
        changed=any(not torch.equal(initial_weights[k],v)for k,v in actor.mlp.state_dict().items())
        gates={'actual_cuda_physics_and_learning':True,'finite_all':True,
            'nonzero_actor_gradient':gradients['max_abs']>0,'fresh_actor_weights_changed':changed,
            'penalty_signs_nonpositive':all(v<=1e-6 for v in penalty_max.values()),
            'checkpoint_normalizer_and_onnx_parity':all(c['export']['passed']for c in checkpoints),
            'required_smoke_updates':updates==5 if args.phase=='smoke'else updates>0}
        free,total=torch.cuda.mem_get_info()
        runtime['gpu_memory_end']={'free':free,'total':total}
        runtime['torch_peak_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
        runtime['torch_peak_memory_reserved_bytes']=torch.cuda.max_memory_reserved()
        runtime['memory_measurement_limit']='CUDAfree includesallallocations; torchpeakexcludesWarpallocations.'
        write_json(output/'runtime.json',runtime)
        report={'schema_version':1,'experiment':EXPERIMENT,'phase':args.phase,'seed':args.seed,
            'num_envs':args.num_envs,'horizon':HORIZON,'planned_updates':args.iterations,
            'completed_updates':updates,'transitions':updates*args.num_envs*HORIZON,
            'training_budget_s':args.budget_s,'budget_hit':budget_hit,
            'learning_elapsed_s':time.monotonic()-started-judgement_s,
            'cpu_judge_elapsed_s':judgement_s,'gradients':gradients,'penalty_max':penalty_max,
            'gates':gates,'pipeline_passed':all(gates.values()),'checkpoints':checkpoints,
            'pretrained_actor_loaded':False,'teacher_loaded':False,'release_promoted':False,
            'entry_submitted':False,'competition_win_claimed':False,
            'local_goal_time_s':args.target_time_s,'official_score_after_submission_required':True,
            'initial_actor_sha256':initial_digest,'final_actor_sha256':state_digest(actor)}
        valid=[c for c in checkpoints if c['judge']and c['judge']['evidence_valid']]
        if valid:
            best=min(valid,key=lambda c:(-c['judge']['seeds_finished'],c['judge']['score_s']))
            report['best_local_checkpoint']=best['iteration'];report['best_local_judge']=best['judge']
            report['local_time_target_met']=(best['judge']['seeds_finished']==5
                                            and best['judge']['score_s']<args.target_time_s)
        write_json(output/'summary.json',report)
        if not report['pipeline_passed']:raise RuntimeError(f'Freshtraininggatesfailed:{gates}')
        return report
    finally:
        save_trace(output,trace)
        if wrapped is not None:wrapped.close()
        else:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=tuple(PHASE_BOUNDS),default='smoke')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=14001)
    parser.add_argument('--num-envs',type=int);parser.add_argument('--iterations',type=int)
    parser.add_argument('--budget-s',type=int);parser.add_argument('--checkpoint-interval',type=int)
    parser.add_argument('--judge-contract',type=Path);parser.add_argument('--judge-python',type=Path)
    parser.add_argument('--judge-script',type=Path,default=Path('evaluate_entry.py'))
    parser.add_argument('--target-time-s',type=float,default=1.66)
    args=parser.parse_args()
    if(args.judge_contract is None)!=(args.judge_python is None):
        parser.error('judgecontract/pythonmustbesuppliedtogether')
    try:
        args.num_envs,args.iterations,args.budget_s,args.checkpoint_interval=phase_configuration(
            args.phase,args.num_envs,args.iterations,args.budget_s,args.checkpoint_interval)
    except ValueError as error:parser.error(str(error))
    if not math.isfinite(args.target_time_s)or args.target_time_s<=0:
        parser.error('Targettimemustbefiniteandpositive')
    if args.output.exists():parser.error('Preserveexistingoutput; chooseanewimmutablepath')
    args.output.mkdir(parents=True,mode=0o700)
    write_json(args.output/'invocation.json',{'experiment':EXPERIMENT,'phase':args.phase,
        'seed':args.seed,'num_envs':args.num_envs,'iterations':args.iterations,'budget_s':args.budget_s,
        'checkpoint_interval':args.checkpoint_interval,'horizon':HORIZON,
        'actor_architecture':[61,512,256,128,14],'actor_normalizer':'fresh_empirical_baked_on_export',
        'pretrained_actor_loaded':False,'teacher_loaded':False,'initial_log_gaussian_std_rad':INITIAL_STD_RAD,
        'learning_rate':LEARNING_RATE,'learning_rate_schedule':'fixed','target_time_s':args.target_time_s})
    try:
        result=run(args,args.output)
        print(json.dumps({'phase':args.phase,'pipeline_passed':result['pipeline_passed'],
            'completed_updates':result['completed_updates'],'entry_submitted':False}),flush=True)
        return 0
    except BaseException as error:
        write_json(args.output/'execution-failure.json',{'error_type':type(error).__name__,
            'message':str(error),'entry_submitted':False,'release_promoted':False})
        raise

if __name__=='__main__':raise SystemExit(main())
