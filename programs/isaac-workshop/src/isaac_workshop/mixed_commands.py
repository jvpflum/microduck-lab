"""Explicit bounded exposure and raw native command-observation evidence.

MicroDuck's forward/turn bucket probabilities only select resampling values;
there are no persistent forward/turn mode flags. The actual command tensor,
standing mask and heading mask are the native interface used here.
"""
import math

CASES = ('walk', 'left_turn', 'right_turn')
TWISTS = ((.15, 0., 0.), (0., 0., .3), (0., 0., -.3))
ACTIVE_CONTROLS = 200
EPISODE_CONTROLS = 300


def case_at(generation):
    if type(generation) is not int or generation < 0:
        raise ValueError('An actual initialized episode generation is required')
    return CASES[generation % len(CASES)]


def command_at(generation, completed_episode_controls):
    case_at(generation)
    if type(completed_episode_controls) is not int or not 0 <= completed_episode_controls <= EPISODE_CONTROLS:
        raise ValueError('Invalid per-episode control counter')
    return TWISTS[generation % len(CASES)] if completed_episode_controls < ACTIVE_CONTROLS else (0., 0., 0.)


def recipe():
    return {'cases': list(CASES), 'body_twists': [list(v) for v in TWISTS],
        'active_controls': ACTIVE_CONTROLS, 'stop_controls': EPISODE_CONTROLS-ACTIVE_CONTROLS,
        'control_dt_s': .02, 'episode_seconds': 6., 'resampling_seconds': 7.,
        'case_selection': 'Actual native reset generation modulo three; early terminal resets advance case',
        'initial_sampling': 'Write complete first command in _resample_command, before reset clears episode_length_buf',
        'standing_mask': 'True only during stop; heading mask always false',
        'command_observation_slices': {'policy': [48,51], 'critic': [51,54]},
        'balanced_exposure_claimed': False,
        'limitation': 'Requested episode count is not actual exposure; early falls, resets and partial episodes are retained'}


def command_type(native_type, tensor_module=None):
    """Subclass the installed native command without changing installed sources."""
    if tensor_module is None:
        import torch as tensor_module

    class MixedPreservationCommand(native_type):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            if self.num_envs != 1:
                raise RuntimeError('This bounded mixed-exposure recipe requires one world')
            self.episode_generation = -1

        def _assign(self, value):
            self.vel_command_b[:] = tensor_module.tensor([value], device=self.device, dtype=self.vel_command_b.dtype)
            self.is_standing_env[:] = value == (0., 0., 0.)
            if hasattr(self, 'is_heading_env'):
                self.is_heading_env[:] = False

        def _resample_command(self, env_ids):
            ids = list(range(self.num_envs))[env_ids] if isinstance(env_ids, slice) else (
                env_ids.tolist() if hasattr(env_ids, 'tolist') else list(env_ids))
            if not ids:
                return
            if ids != [0]:
                raise RuntimeError('Unexpected reset world IDs')
            # The pinned RLEnv calls command_manager.reset before it clears the
            # episode-length buffer. Do not read the old terminal count here.
            self.episode_generation += 1
            self._assign(command_at(self.episode_generation, 0))

        def _update_command(self):
            self._assign(command_at(self.episode_generation, int(self._env.episode_length_buf[0])))

    return MixedPreservationCommand


def verify_observations(observations, command, expected):
    """Check copied raw values; no normalized/action input mutation is possible."""
    if set(observations) != {'policy', 'critic'}:
        raise RuntimeError('Unexpected native observation groups')
    if len(command) != 1 or len(command[0]) != 3 or len(expected) != 3:
        raise RuntimeError('Expected one native body-twist row')
    if any(not math.isfinite(float(x)) for x in (*command[0], *expected)):
        raise RuntimeError('Nonfinite command')
    if any(abs(float(a)-float(b)) > 1e-7 for a,b in zip(command[0],expected)):
        raise RuntimeError('Native command differs from declared exposure')
    result = {'native': list(command[0]), 'expected': list(expected)}
    for role, width, start in (('policy',61,48), ('critic',76,51)):
        rows=observations[role]
        if len(rows)!=1 or len(rows[0])!=width or any(not math.isfinite(float(x)) for x in rows[0]):
            raise RuntimeError('Invalid raw native '+role+' observations')
        actual=list(rows[0][start:start+3])
        if any(abs(float(a)-float(b)) > 1e-7 for a,b in zip(actual,command[0])):
            raise RuntimeError(role+' command observation differs before inference')
        result[role]=actual
    return result


def empty_exposure():
    return {case:{'active_controls':0,'stop_controls':0,'episode_generations':[],
        'completed_timeouts':0,'early_terminal_episodes':0,'terminations':{},
        'nonsole_or_nonfloor_magnitude_integral_n_s':0.,'maximum_nonsole_or_nonfloor_force_n':0.}
        for case in CASES}


def record_exposure(totals, exposure, done, contacts, terminations):
    generation=exposure['generation'];case=case_at(generation)
    control=exposure['episode_control_before_step'];command_at(generation,control)
    phase='active' if control<ACTIVE_CONTROLS else 'stop'
    if exposure['case']!=case or exposure['phase']!=phase:
        raise RuntimeError('Recorded exposure differs from actual case/phase')
    row=totals[case];row[phase+'_controls']+=1
    if generation not in row['episode_generations']:row['episode_generations'].append(generation)
    for name in ('nonsole_or_nonfloor_magnitude_integral_n_s','maximum_nonsole_or_nonfloor_force_n'):
        value=contacts[name]
        if not math.isfinite(value) or value<0:raise RuntimeError('Invalid contact exposure metric')
        if name.startswith('maximum'):row[name]=max(row[name],value)
        else:row[name]+=value
    for name,flag in terminations.items():
        if flag:row['terminations'][name]=row['terminations'].get(name,0)+1
    if done:
        if control+1>=EPISODE_CONTROLS and terminations.get('time_out'):
            row['completed_timeouts']+=1
        else:row['early_terminal_episodes']+=1


def begin_behavior(episodes, generation, position, yaw):
    """Start from a measured pre-action state, including the first episode."""
    key=str(generation)
    if key not in episodes:
        if len(position)!=3 or not all(math.isfinite(x) for x in (*position,yaw)):
            raise RuntimeError('Invalid measured episode start')
        episodes[key]={'case':case_at(generation),'start_position_m':list(position),
            'last_position_m':list(position),'last_yaw_rad':yaw,'yaw_change_rad':0.,
            'controls':0,'active_controls':0,'stop_controls':0,'max_tilt_rad':0.,
            'active_linear_error_squared_sum':0.,'active_yaw_error_squared_sum':0.,
            'stop_speed_squared_sum':0.,'stop_start_position_m':None,
            'ended':False,'terminations':{},'behavior_qualified':False}


def record_behavior(episodes, exposure, position, yaw, tilt, velocity, angular_velocity, terminations):
    row=episodes[str(exposure['generation'])]
    if row['ended']:raise RuntimeError('Cannot append to an ended episode')
    numbers=(*position,yaw,tilt,*velocity,*angular_velocity)
    if not all(math.isfinite(x) for x in numbers):raise RuntimeError('Nonfinite behavior state')
    command=command_at(exposure['generation'],exposure['episode_control_before_step'])
    phase=exposure['phase'];row['controls']+=1;row[phase+'_controls']+=1
    # Unwrap only the measured incremental angle, never desired command yaw.
    row['yaw_change_rad']+=math.atan2(math.sin(yaw-row['last_yaw_rad']),math.cos(yaw-row['last_yaw_rad']))
    row['last_yaw_rad']=yaw;row['max_tilt_rad']=max(row['max_tilt_rad'],tilt)
    if phase=='active':
        row['active_linear_error_squared_sum']+=sum((velocity[i]-command[i])**2 for i in (0,1))
        row['active_yaw_error_squared_sum']+=(angular_velocity[2]-command[2])**2
    else:
        if row['stop_start_position_m'] is None:row['stop_start_position_m']=list(row['last_position_m'])
        row['stop_speed_squared_sum']+=sum(velocity[i]**2 for i in (0,1))
    row['last_position_m']=list(position)
    row['displacement_xy_m']=math.dist(position[:2],row['start_position_m'][:2])
    row['active_linear_tracking_rms_m_s']=(row['active_linear_error_squared_sum']/row['active_controls'])**.5 if row['active_controls'] else None
    row['active_yaw_tracking_rms_rad_s']=(row['active_yaw_error_squared_sum']/row['active_controls'])**.5 if row['active_controls'] else None
    row['stop_speed_rms_m_s']=(row['stop_speed_squared_sum']/row['stop_controls'])**.5 if row['stop_controls'] else None
    row['stop_displacement_m']=math.dist(position[:2],row['stop_start_position_m'][:2]) if row['stop_start_position_m'] else None
    row['terminations']={name:bool(flag) for name,flag in terminations.items()}
    row['ended']=any(terminations.values())
