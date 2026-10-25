"""General-domain adapter around the immutable strict native trace scorer.

Only the explicitly validated world bounds vary. Contact semantics, thresholds,
complete sequences and exact final-one-second stop metrics remain unchanged.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SCORER = Path(__file__).resolve().with_name('score_locomotion.py')
SCORER_SHA = '38aa56a5a2b6d2ab805f536ce3f2215d4d8813a2ea3d565e7ebfbba559d8ac4e'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_domain(receipt, expected_sha256):
    if digest(receipt) != expected_sha256 or receipt.get('schema') != 'validated-locomotion-domain-v1':
        raise ValueError('Domain receipt or schema differs')
    required = ('source_geometry', 'collision_geometry', 'floor_support', 'start_clearance', 'trajectory_clearance', 'reset_profiles')
    if any(receipt.get('checks',{}).get(k) is not True for k in required):
        raise ValueError('Domain source/support/collision review incomplete')
    bounds = receipt['root_domain']
    if len(bounds)!=2 or any(len(x)!=3 for x in bounds) or not all(math.isfinite(v) for row in bounds for v in row) or any(a>=b for a,b in zip(*bounds)):
        raise ValueError('Invalid measured root domain')
    starts=receipt['train_starts']+receipt['heldout_starts']
    if not receipt['train_starts'] or not receipt['heldout_starts'] or len({tuple(s) for s in starts})!=len(starts):
        raise ValueError('Training and held-out starts must be nonempty and disjoint')
    for start in starts:
        if len(start)!=3 or not all(math.isfinite(v) for v in start) or not all(bounds[0][i]<=start[i]<=bounds[1][i] for i in (0,1)):
            raise ValueError('Invalid start bank')
    if set(receipt['reset_profiles']) != {'nominal', 'small_perturbation'}:
        raise ValueError('Recovery start profile must have measured support and clearance')
    if any(not isinstance(v,str) or len(v)!=64 for v in receipt['source_sha256'].values()) or not receipt['source_sha256']:
        raise ValueError('Exact geometry source pins required')
    return receipt


def suite(domain, domain_sha256):
    validate_domain(domain, domain_sha256)
    # Held-out command values are absent from the training command grid.
    commands = {'stand':(0.,0.,0.), 'walk':(.15,0.,0.), 'reverse':(-.10,0.,0.),
                'left':(0.,0.,.3), 'right':(0.,0.,-.3),
                'heldout_walk':(.12,0.,0.), 'heldout_left':(0.,0.,.22), 'heldout_right':(0.,0.,-.22),
                'heldout_arc_left':(.08,0.,.22), 'heldout_arc_right':(.08,0.,-.22)}
    cases=[]
    for name,twist in commands.items():
        held=name.startswith('heldout')
        for index,start in enumerate(domain['heldout_starts'] if held else domain['train_starts']):
            cases.append({'name':f'{name}-{index}', 'start_xy_yaw':start,
                'segments':[{'controls':200,'body_twist':list(twist)}, {'controls':100,'body_twist':[0.,0.,0.]}],
                'held_out_from_training_start_and_command':held, 'reset_profile':'nominal'})
    # Recovery means a physically measured small perturbation at start; reset
    # or teleporting upright after a fall is still rejected by the scorer.
    for index,start in enumerate(domain['heldout_starts']):
        cases.append({'name':f'heldout_recover-{index}', 'start_xy_yaw':start,
            'segments':[{'controls':200,'body_twist':[0.,0.,0.]}, {'controls':100,'body_twist':[0.,0.,0.]}],
            'held_out_from_training_start_and_command':False, 'held_out_reset_and_start':True,
            'reset_profile':'small_perturbation'})
    return {'schema':'rounded-locomotion-suite-v1', 'domain_sha256':domain_sha256,
            'scorer_sha256':SCORER_SHA, 'cases':cases, 'camera_navigation_qualified':False}


def score(case, trace, substeps, domain, domain_sha256):
    validate_domain(domain, domain_sha256)
    if digest(case) not in domain.get('admitted_case_sha256', []):
        raise ValueError('This exact start/command/reset trajectory lacks source-domain clearance')
    if hashlib.sha256(SCORER.read_bytes()).hexdigest()!=SCORER_SHA:
        raise ValueError('Immutable scorer changed')
    # Isolated module instance: never mutate the frozen scorer or another run.
    spec=importlib.util.spec_from_file_location('isaac_workshop._rounded_locomotion_scorer', SCORER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT_DOMAIN=tuple(tuple(row) for row in domain['root_domain'])
    result=module.score_episode(case,trace,substeps)
    result.update(domain_sha256=domain_sha256, declared_root_domain=domain['root_domain'],
                  immutable_scorer_sha256=SCORER_SHA, domain_override_explicit=True)
    return result


def promotion(candidate_sha256, suite_contract, reports, protected_baseline_sha256):
    for pin in (candidate_sha256, protected_baseline_sha256):
        if not isinstance(pin,str) or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):
            raise ValueError('Exact model hashes required')
    names={x['name'] for x in suite_contract['cases']}
    if len(names)!=len(suite_contract['cases']) or set(reports)!=names:
        raise ValueError('Every predeclared evaluation case required, including failures')
    strict=all(r.get('valid') is True and r.get('qualified') is True and
               r.get('domain_sha256')==suite_contract['domain_sha256'] and
               r.get('immutable_scorer_sha256')==SCORER_SHA and r.get('scenario')==case
               for case in suite_contract['cases'] for r in [reports[case['name']]])
    return {'candidate_checkpoint_sha256':candidate_sha256, 'protected_baseline_sha256':protected_baseline_sha256,
            'suite_sha256':digest(suite_contract), 'rounded_locomotion_qualified':strict,
            'replace_baseline':False, 'camera_navigation_qualified':False,
            'all_cases':{k:{'valid':v.get('valid'), 'qualified':v.get('qualified')} for k,v in reports.items()}}
