"""CPU-testable camera-to-command boundary; no motor policy or optimizer.

One packet is supplied before each 50 Hz locomotor action. Navigation runs at
10 Hz, but freshness is checked at every control interval. A bad packet latches
a zero command until an explicit new episode. This is integration code, not a
trained navigator or a claim of native obstacle avoidance.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import numpy as np


def digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def finite_vector(value, size, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError('Invalid ' + name)
    result = value.copy(); result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Contract:
    height: int = 48
    width: int = 64
    control_dt_s: float = .02
    navigation_stride: int = 5
    near_m: float = .03
    far_m: float = 5.
    minimum_valid_fraction: float = .01
    minimum_frontal_valid_fraction: float = .8
    max_forward_m_s: float = .15
    max_yaw_rad_s: float = .4

    def __post_init__(self):
        if (type(self.height) is not int or type(self.width) is not int
                or min(self.height, self.width) < 2 or self.control_dt_s != .02
                or type(self.navigation_stride) is not int or self.navigation_stride != 5
                or not 0 < self.near_m < self.far_m
                or not 0 < self.minimum_valid_fraction <= 1
                or not 0 < self.minimum_frontal_valid_fraction <= 1
                or not 0 < self.max_forward_m_s <= .15 or not 0 < self.max_yaw_rad_s <= .4
                or not all(math.isfinite(x) for x in (self.near_m, self.far_m,
                    self.minimum_valid_fraction, self.minimum_frontal_valid_fraction,
                    self.max_forward_m_s, self.max_yaw_rad_s))):
            raise ValueError('Unsupported bounded pilot contract')


@dataclass(frozen=True)
class CameraPacket:
    episode_id: int
    control_step: int
    sensor_frame: int
    sensor_timestamp_s: float
    simulation_time_s: float
    depth_m: np.ndarray


def image_channels(packet, contract):
    raw = np.asarray(packet.depth_m)
    if raw.shape != (contract.height, contract.width) or raw.dtype.kind != 'f':
        raise ValueError('Invalid optical depth shape/type')
    if np.isnan(raw).any() or np.isneginf(raw).any() or (raw < 0).any():
        raise ValueError('Invalid optical depth values')
    # The native renderer uses unclipped depth. A finite sub-near hit is not
    # empty/far space. Stop the bounded probe instead of hiding it in the mask.
    if (np.isfinite(raw) & (raw < contract.near_m)).any():
        raise ValueError('Near-clipped/too-close depth requires stop')
    valid = np.isfinite(raw) & (raw >= contract.near_m) & (raw < contract.far_m)
    if float(valid.mean()) < contract.minimum_valid_fraction:
        raise ValueError('Insufficient valid depth')
    safe = np.where(valid, raw, contract.far_m).astype(np.float32)
    image = np.stack(((safe-contract.near_m)/(contract.far_m-contract.near_m), valid), axis=0).astype(np.float32)
    image.flags.writeable = False
    return image


def bounded_command(output, contract):
    """Explicit [forward fraction, signed yaw fraction], no lateral motion."""
    output = finite_vector(output, 2, 'navigator output')
    if not 0 <= output[0] <= 1 or not -1 <= output[1] <= 1:
        raise ValueError('Navigator output outside declared bounds')
    return (float(output[0])*contract.max_forward_m_s, 0., float(output[1])*contract.max_yaw_rad_s)


class NavigationBridge:
    def __init__(self, contract=None):
        self.contract = contract or Contract()
        self.episode = None
        self.command = (0., 0., 0.)
        self.fault = None
        self.previous = None
        self.decision_step = None

    def begin_episode(self, episode_id):
        if type(episode_id) is not int or episode_id < 0 or self.episode is not None and episode_id <= self.episode:
            raise ValueError('Explicit increasing episode identity required')
        self.episode = episode_id; self.previous = None; self.fault = None
        self.command = (0., 0., 0.); self.decision_step = None

    def advance(self, packet, locomotor_observation, goal_body_m, predict, *, goal_provenance):
        """Return the held/new command and a JSON-safe per-control evidence row.

        `predict` receives only normalized depth/mask, existing 61-value policy
        observation and destination vector. No map, object poses, waypoints or
        collision labels are exposed. It returns two bounded fractions.
        """
        row = {'episode_id': self.episode, 'control_step': getattr(packet, 'control_step', None),
               'decision': False, 'command_body_twist': [0., 0., 0.],
               'goal_provenance': goal_provenance, 'depth_used_by_locomotor': False}
        if self.fault is not None:
            row.update(fault=self.fault, latched_stop=True)
            return self.command, row
        try:
            if self.episode is None or packet.episode_id != self.episode:
                raise ValueError('Packet crosses episode boundary')
            if any(type(x) is not int for x in (packet.control_step, packet.sensor_frame)) or packet.control_step < 0 or packet.sensor_frame < 1:
                raise ValueError('Invalid control/frame index')
            times = (packet.sensor_timestamp_s, packet.simulation_time_s)
            if not all(math.isfinite(x) and x >= 0 for x in times):
                raise ValueError('Invalid sensor/simulation clock')
            age = packet.simulation_time_s-packet.sensor_timestamp_s
            if not -1e-6 <= age <= self.contract.control_dt_s+1e-6:
                raise ValueError('Stale/future depth timestamp')
            if self.previous is not None:
                step, frame, stamp, simtime = self.previous
                if packet.control_step != step+1 or packet.sensor_frame <= frame or packet.sensor_timestamp_s <= stamp+1e-7:
                    raise ValueError('Skipped control or repeated sensor capture')
                if not math.isclose(packet.simulation_time_s-simtime, self.contract.control_dt_s, abs_tol=1e-6):
                    raise ValueError('Control clock changed')
            if goal_provenance not in ('simulator_pose_scaffold', 'measured_localization'):
                raise ValueError('Destination localization provenance required')
            image = image_channels(packet, self.contract)
            # This is a conservative probe interlock, not generic obstacle
            # safety: unknown frontal pixels never authorize forward motion.
            y0,y1=int(self.contract.height*.25),math.ceil(self.contract.height*.8)
            x0,x1=int(self.contract.width*.2),math.ceil(self.contract.width*.8)
            frontal_valid=float(image[1,y0:y1,x0:x1].mean())
            if frontal_valid < self.contract.minimum_frontal_valid_fraction:
                raise ValueError('Unknown frontal depth requires stop')
            state = finite_vector(locomotor_observation, 61, 'unchanged locomotor observation')
            goal = finite_vector(goal_body_m, 2, 'body-relative destination')
            row.update(sensor_frame=packet.sensor_frame, sensor_timestamp_s=packet.sensor_timestamp_s,
                       simulation_time_s=packet.simulation_time_s, frame_age_s=age,
                       raw_depth_sha256=digest(packet.depth_m), image_sha256=digest(image),
                       state_sha256=digest(state), goal_body_m=goal.tolist(),
                       valid_fraction=float(image[1].mean()), frontal_valid_fraction=frontal_valid)
            if self.decision_step is None or packet.control_step-self.decision_step == self.contract.navigation_stride:
                hashes = tuple(map(digest, (image, state, goal)))
                output = predict({'head_depth': image, 'policy': state, 'goal_body_m': goal})
                if hashes != tuple(map(digest, (image, state, goal))):
                    raise ValueError('Navigator modified its observation')
                self.command = bounded_command(output, self.contract)
                self.decision_step = packet.control_step; row['decision'] = True
            self.previous = (packet.control_step, packet.sensor_frame, packet.sensor_timestamp_s, packet.simulation_time_s)
            row.update(command_body_twist=list(self.command), decision_control_step=self.decision_step,
                       fault=None, latched_stop=False)
        except (ValueError, TypeError, AttributeError) as error:
            self.fault = str(error); self.command = (0., 0., 0.)
            row.update(fault=self.fault, latched_stop=True)
        return self.command, row


def goal_in_body(root_xy, yaw_rad, goal_xy):
    """Destination localization only. Caller must label simulator pose as privileged."""
    root, goal = finite_vector(root_xy, 2, 'root XY'), finite_vector(goal_xy, 2, 'goal XY')
    if not math.isfinite(yaw_rad): raise ValueError('Invalid yaw')
    dx, dy = goal-root; c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([c*dx+s*dy, -s*dx+c*dy], dtype=np.float32)


def camera_counterfactual(predict, actual, alternate_depth):
    """Paired feed-forward diagnostic, not a rollout or competence certificate.

    Use an independent eval-mode navigator with fixed state. Recurrent policies
    need two equal clones of the same hidden state and must not use this helper.
    """
    baseline = {k: np.array(v, copy=True) for k, v in actual.items()}
    changed = {k: np.array(v, copy=True) for k, v in actual.items()}
    changed['head_depth'] = np.array(alternate_depth, copy=True)
    if set(baseline) != {'head_depth', 'policy', 'goal_body_m'} or changed['head_depth'].shape != baseline['head_depth'].shape:
        raise ValueError('Counterfactual input contract differs')
    if not all(np.isfinite(v).all() for x in (baseline, changed) for v in x.values()):
        raise ValueError('Nonfinite counterfactual')
    before = [{k:digest(v) for k,v in x.items()} for x in (baseline,changed)]
    outputs = [finite_vector(predict(x), 2, 'paired output') for x in (baseline,changed)]
    if before != [{k:digest(v) for k,v in x.items()} for x in (baseline,changed)]:
        raise ValueError('Counterfactual mutated input')
    return {'actual_output':outputs[0].tolist(), 'alternate_output':outputs[1].tolist(),
            'maximum_command_fraction_difference':float(np.max(np.abs(outputs[0]-outputs[1]))),
            'same_state_and_goal':all(before[0][k]==before[1][k] for k in ('policy','goal_body_m')),
            'different_depth':before[0]['head_depth']!=before[1]['head_depth'],
            'depth_hashes':[x['head_depth'] for x in before], 'task_success_proven':False}
