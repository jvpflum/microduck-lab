"""Balanced command curriculum; pure Python, no simulator or submission side effects.

Reservations distribute concurrent worlds. Actual solved controls, including
terminal controls, determine exposure. Early falls never count as full episodes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

DT = .02
ACTIVE = 200
STOP = 100


@dataclass(frozen=True)
class Skill:
    name: str
    twist: tuple[float, float, float]
    reset_profile: str = 'nominal'


STAGES = {
    'flat_control': (
        Skill('stand', (0., 0., 0.)), Skill('forward', (.15, 0., 0.)),
        Skill('backward', (-.10, 0., 0.)), Skill('turn_left', (0., 0., .3)),
        Skill('turn_right', (0., 0., -.3)), Skill('recover_stand', (0., 0., 0.), 'small_perturbation')),
    'mixed_control': (
        Skill('stand', (0., 0., 0.)), Skill('forward', (.15, 0., 0.)),
        Skill('backward', (-.10, 0., 0.)), Skill('turn_left', (0., 0., .3)),
        Skill('turn_right', (0., 0., -.3)), Skill('recover_stand', (0., 0., 0.), 'small_perturbation'),
        Skill('side_left', (0., .06, 0.)), Skill('side_right', (0., -.06, 0.)),
        Skill('arc_left', (.10, 0., .3)), Skill('arc_right', (.10, 0., -.3))),
}
STAGES['terrain_control'] = STAGES['mixed_control']

# Preserve the source recipe's initial randomization across this first staged
# skill/terrain curriculum. Do not invent a nominal-only training ablation.
# Automatic later source COM expansions require separate native readback and
# geometry admission before enabling; these are not hardware tolerances.
RANDOMIZATION = {stage: {'foot_friction': (.7, 1.3), 'mass_scale': (.95, 1.05),
                        'encoder_bias': (-.015, .015), 'com_xyz': (-.003, .003)}
                 for stage in STAGES}


def skills(stage):
    if stage not in STAGES:
        raise ValueError('Unknown explicit stage')
    return STAGES[stage]


def command(skill, index):
    if type(index) is not int or not 0 <= index < ACTIVE + STOP:
        raise ValueError('Control outside complete episode')
    return skill.twist if index < ACTIVE else (0., 0., 0.)


class ExposureLedger:
    def __init__(self, stage):
        self.skills = skills(stage)
        self.stage = stage
        self.rows = {s.name: dict(active=0, stop=0, episodes=0, completed=0, terminated=0) for s in self.skills}
        self.live = {}
        self.generation = 0

    def begin(self, world):
        if type(world) is not int or world < 0 or world in self.live:
            raise ValueError('Invalid or still-active world')
        # Count active reservations to prevent all parallel resets choosing the
        # same least-exposed skill. Release on actual completion/termination.
        def priority(skill):
            reserved = sum(ACTIVE - min(v['controls'], ACTIVE) for v in self.live.values() if v['skill'] == skill)
            row = self.rows[skill.name]
            return (row['active'] + reserved, row['episodes'], self.skills.index(skill))
        skill = min(self.skills, key=priority)
        self.live[world] = {'skill': skill, 'controls': 0, 'generation': self.generation}
        self.generation += 1
        self.rows[skill.name]['episodes'] += 1
        return skill

    def expected(self, world):
        entry = self.live[world]
        return command(entry['skill'], entry['controls'])

    def record_solved_control(self, world, index, actual_command, *, substeps, terminal=False):
        entry = self.live[world]
        if type(index) is not int or index != entry['controls'] or substeps != 4:
            raise ValueError('Missing, repeated or out-of-order solved control/substeps')
        if len(actual_command) != 3 or any(not math.isfinite(float(x)) for x in actual_command):
            raise ValueError('Nonfinite or invalid command')
        if any(abs(a-b) > 1e-7 for a,b in zip(actual_command, self.expected(world))):
            raise ValueError('Actual command differs from assigned curriculum')
        row = self.rows[entry['skill'].name]
        row['active' if index < ACTIVE else 'stop'] += 1
        entry['controls'] += 1
        if terminal or entry['controls'] == ACTIVE + STOP:
            row['terminated' if terminal else 'completed'] += 1
            del self.live[world]

    def report(self):
        counts = [r['active'] for r in self.rows.values()]
        return {'stage': self.stage, 'actual_exposure': {k: dict(v) for k,v in self.rows.items()},
                'active_control_range': [min(counts), max(counts)],
                'solved_substeps': 4*sum(r['active']+r['stop'] for r in self.rows.values()),
                'unfinished_worlds': sorted(self.live), 'balanced_exposure_claimed': False,
                'reset_is_not_recovery': True}


def verify_command_observations(observations, native_commands, expected):
    """Raw pre-normalizer copies only. Does not rewrite any observation slot."""
    if set(observations) != {'policy', 'critic'} or len(native_commands) != len(expected):
        raise ValueError('Unexpected observation groups/world count')
    for name,width,offset in [('policy',61,48), ('critic',76,51)]:
        rows = observations[name]
        if len(rows) != len(expected):
            raise ValueError('Observation batch differs')
        for row,native,wanted in zip(rows,native_commands,expected):
            if len(row) != width or len(native) != 3 or len(wanted) != 3 or not all(math.isfinite(float(x)) for x in row):
                raise ValueError('Invalid finite native observation')
            if any(abs(a-b)>1e-7 for a,b in zip(native,wanted)) or any(abs(a-b)>1e-7 for a,b in zip(row[offset:offset+3],wanted)):
                raise ValueError('Stale or masked native command before inference')
    return {'worlds': len(expected), 'policy_slice': [48,51], 'critic_slice': [51,54], 'mutated_observations': False}


def target_metrics(requested, lower, upper):
    """Measure out-of-range targets. Never clamp or reinterpret the actor."""
    if not len(requested) == len(lower) == len(upper) == 14:
        raise ValueError('14 named servo target/limit values required')
    if not all(math.isfinite(float(x)) for x in (*requested,*lower,*upper)) or any(lo>=hi for lo,hi in zip(lower,upper)):
        raise ValueError('Invalid finite target/limits')
    excess = [max(lo-q, 0., q-hi) for q,lo,hi in zip(requested,lower,upper)]
    return {'outside_count': sum(x>0 for x in excess), 'excess_rad': excess,
            'squared_excess_rad2': sum(x*x for x in excess), 'targets_changed': False}
