"""Native command refresh with exact noncommand observation preservation.

Requires the pinned native environment; does not start a simulator or submit work.
"""
import math


def apply_command_and_refresh(base,wrapped,old_observations,command):
    """Use the real command term, then recompute without advancing history.

    Refuse a native mismatch instead of patching arbitrary actor input indices.
    The dynamic slice is only an audit of the command term, never a tensor edit.
    """
    import torch
    manager=base.observation_manager
    names=manager.active_terms['policy'];dims=manager.group_obs_term_dim['policy']
    if names.count('velocity_commands')!=1 or len(names)!=len(dims):raise RuntimeError('Velocity observation term differs')
    offset=0;selected=None
    for name,shape in zip(names,dims):
        size=math.prod(shape)
        if name=='velocity_commands':selected=(offset,offset+size)
        offset+=size
    if offset!=61 or selected[1]-selected[0]!=3:raise RuntimeError('61D actor command contract differs')
    before=old_observations['policy'].clone()
    base._navigation_command=tuple(command)
    base.command_manager.get_term('base_velocity')._update_command()
    base.obs_buf=manager.compute(update_history=False)
    fresh=wrapped.get_observations()
    a,b=selected
    expected=torch.tensor([command],device=base.device,dtype=fresh['policy'].dtype)
    if not torch.allclose(fresh['policy'][:,a:b],expected,rtol=0,atol=1e-7):raise RuntimeError('Actual actor command differs')
    if not torch.equal(before[:,:a],fresh['policy'][:,:a]) or not torch.equal(before[:,b:],fresh['policy'][:,b:]):
        raise RuntimeError('Command refresh changed a noncommand actor observation')
    return fresh
