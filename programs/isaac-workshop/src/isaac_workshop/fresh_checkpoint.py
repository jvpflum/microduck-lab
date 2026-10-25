"""Fresh CPU MLP/Adam checkpoint validation; never copy a live model cache."""
from __future__ import annotations
import hashlib
import inspect
import math
from pathlib import Path

PPO_MODULE_SHA='01a82cea45fa9d8ccb161bd75e07d8623d25450ff92982e9c9d59ca8a9b78393'


def assert_tree_equal(actual, expected, path='state'):
    import torch
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError('Tensor schema differs: '+path)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not torch.equal(actual.detach().cpu(),expected.detach().cpu()):
            raise ValueError('Tensor value/finite check differs: '+path)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError('Keys differ: '+path)
        for key in expected:assert_tree_equal(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):raise ValueError('Sequence differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):assert_tree_equal(a,b,path+'/'+str(i))
    elif type(actual)!=type(expected) or actual!=expected or (isinstance(expected,float) and not math.isfinite(expected)):
        raise ValueError('Scalar differs: '+path)


def finite_tree(value):
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu' or not torch.isfinite(value).all():raise ValueError('Nonfinite/non-CPU saved tensor')
        return 1
    if isinstance(value,dict):return sum(finite_tree(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(finite_tree(v) for v in value)
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite scalar')
    return 0


def construct_models(observations):
    from rsl_rl.models import MLPModel
    if set(observations.keys())!={'policy','critic'}:raise ValueError('Unexpected observation groups')
    batch=observations['policy'].shape[0]
    if batch<1 or tuple(observations['policy'].shape)!=(batch,61) or tuple(observations['critic'].shape)!=(batch,76):
        raise ValueError('Expected native 61/76 observation contract')
    finite_tree(dict(observations.items()))
    groups={'actor':['policy'],'critic':['critic']}
    return {name:MLPModel(observations,groups,name,outdim,hidden_dims=[512,256,128],activation='elu',
            obs_normalization=True,distribution_cfg=({'class_name':'GaussianDistribution','init_std':1.,'std_type':'scalar'} if name=='actor' else None)).cpu()
            for name,outdim in [('actor',14),('critic',1)]}


def validate_fresh_state(saved, observations, *, expected_models=None, expected_optimizer=None, teacher_saved=None):
    """Load exact state into newly constructed objects, without optimizer steps.

    observations must be CPU native 61/76 tensors. Optional live comparisons
    copy only state tensors to CPU, never Python models/distribution caches.
    """
    import importlib.metadata
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    ppo_file=Path(inspect.getfile(PPO))
    if importlib.metadata.version('rsl-rl-lib')!='5.5.1' or hashlib.sha256(ppo_file.read_bytes()).hexdigest()!=PPO_MODULE_SHA:
        raise ValueError('Unreviewed native PPO/Adam parameter ordering')
    finite_count=finite_tree(saved)
    models=construct_models(observations)
    for name,model in models.items():
        state=saved[name+'_state_dict'];target=model.state_dict()
        if target.keys()!=state.keys():raise ValueError('Model keys differ: '+name)
        for key in target:
            if target[key].shape!=state[key].shape or target[key].dtype!=state[key].dtype:raise ValueError('Model tensor schema differs: '+name+'/'+key)
        model.load_state_dict(state,strict=True)
        assert_tree_equal(model.state_dict(),state,name)
        if expected_models is not None:assert_tree_equal(expected_models[name].state_dict(),state,'live/'+name)
        model.eval()
    params=list(models['actor'].parameters())+list(models['critic'].parameters())
    opt_saved=saved['optimizer_state_dict']
    if len(opt_saved['param_groups'])!=1 or opt_saved['param_groups'][0]['lr']!=3e-5:raise ValueError('Expected one fixed-rate Adam group')
    ids=opt_saved['param_groups'][0]['params']
    if len(ids)!=len(params) or len(set(ids))!=len(ids) or set(opt_saved['state'])!=set(ids):raise ValueError('Adam parameter coverage differs')
    steps=[]
    for ident,param in zip(ids,params):
        slot=opt_saved['state'][ident]
        if set(slot)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected Adam state keys')
        for key in ('exp_avg','exp_avg_sq'):
            if slot[key].shape!=param.shape or slot[key].dtype!=param.dtype:raise ValueError('Adam moment shape/dtype differs')
        step=float(slot['step'])
        if not math.isfinite(step) or step<=0 or not step.is_integer():raise ValueError('Missing trained Adam steps')
        steps.append(int(step))
    optimizer=torch.optim.Adam(params,lr=3e-5)
    optimizer.load_state_dict(opt_saved)
    assert_tree_equal(optimizer.state_dict(),opt_saved,'fresh_adam')
    if expected_optimizer is not None:assert_tree_equal(expected_optimizer.state_dict(),opt_saved,'live_adam')
    outputs={}
    with torch.inference_mode():
        for name,model in models.items():
            first=model(observations,stochastic_output=False);second=model(observations,stochastic_output=False)
            if not torch.isfinite(first).all() or not torch.equal(first,second):raise ValueError('Nondeterministic/nonfinite fresh inference')
            outputs[name]={'shape':list(first.shape),'max_abs':float(first.abs().max()),'repeat_exact':True}
    # Inference must not mutate loaded normalizers, weights, or Adam state.
    for name,model in models.items():assert_tree_equal(model.state_dict(),saved[name+'_state_dict'],'post_inference/'+name)
    assert_tree_equal(optimizer.state_dict(),opt_saved,'post_inference_adam')
    finite_tree(optimizer.state_dict())
    for model in models.values():finite_tree(model.state_dict())
    teacher_checks=None
    if teacher_saved is not None:
        teacher_checks={}
        for name in models:
            current=saved[name+'_state_dict'];old=teacher_saved[name+'_state_dict']
            normalizers=[k for k in current if k.startswith('obs_normalizer.')]
            if not normalizers:raise ValueError('Missing native normalizer state')
            for key in normalizers:assert_tree_equal(current[key],old[key],name+'/'+key)
            changed=any(not torch.equal(value,old[key]) for key,value in current.items() if not key.startswith('obs_normalizer.'))
            teacher_checks[name]={'normalizer_exact_teacher':True,'non_normalizer_state_changed':changed}
        if not teacher_checks['actor']['non_normalizer_state_changed']:raise ValueError('Student actor did not change')
    return {'strict_fresh_model_reload_verified':True,'strict_fresh_optimizer_reload_verified':True,
        'finite_tensor_count':finite_count,'observation_samples':int(observations['policy'].shape[0]),
        'inference':outputs,'optimizer_parameter_count':len(params),'adam_step_range':[min(steps),max(steps)],
        'optimizer_step_called':False,'model_or_optimizer_mutated_by_inference':False,
        'teacher_comparison':teacher_checks,'ppo_module_sha256':PPO_MODULE_SHA,
        'mlp_module_sha256':hashlib.sha256(Path(inspect.getfile(MLPModel)).read_bytes()).hexdigest(),
        'device':'cpu'}
