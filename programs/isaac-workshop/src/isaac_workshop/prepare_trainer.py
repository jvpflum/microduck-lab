"""Write a reviewable native trainer proposal. Never creates a job or trains.

The native command/reward adapter is executable, but source-seated reset/profile
application and foundation admission must come from independently verified native
contracts before integration. This program cannot turn a draft into approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .curriculum import STAGES, RANDOMIZATION
from .native_curriculum import admit_foundation
from .evaluation_contract import validate_domain, suite


def build_proposal(foundation=None, domain=None, domain_sha256=None):
    if foundation is not None:admit_foundation(foundation)
    if domain is not None:validate_domain(domain,domain_sha256)
    return {'schema':'rounded-locomotion-trainer-proposal-v1', 'launch_ready':False, 'training_performed':False,
        'selected_foundation':foundation, 'evaluation_suite':suite(domain,domain_sha256) if domain else None,
        'stage_order':list(STAGES), 'stages':{n:{'skills':[{'name':s.name,'body_twist':s.twist,'reset_profile':s.reset_profile}for s in values],
            'active_controls':200,'stop_controls':100,'randomization':RANDOMIZATION[n]} for n,values in STAGES.items()},
        'ppo':{'source':'pinned native RSL-RL5.5.1; preserve its reviewed source contract',
            'horizon':24,'first_smoke_worlds':64,'first_smoke_updates':5,
            'initialization':'Exact selected compatible checkpoint, fresh Adam; protected parent remains immutable.',
            'normalizers':'Preserve loaded states in first pilot; fresh-model normalization is a separate explicit recipe.',
            'teacher_anchor':'No global anchor to the failed-turn teacher. Optional preservation loss requires a qualified skill-specific teacher and a separately reviewed coefficient.',
            'main_run_size':'Choose after native capacity smoke; no fabricated throughput or duration claim.'},
        'integration_order':[
            'Validate exact benchmark admission, checkpoint/normalizer and named observation/action contract.',
            'Build a source-derived collision domain and exact train/held-out start+trajectory clearance receipt.',
            'Install domain reset-profile adapter before create_env; native_curriculum.prepare configures command exposure and retains native tracking rewards.',
            'After create_env verify source geometry, full robot/collision/motor contract and actual randomized parameter readbacks.',
            'Before every PPO.act validate all raw native actor48:51/critic51:54 command slices without masking.',
            'After every solved control and before autoreset record all4 actual contacts plus ExposureLedger.record_solved_control, including falls/partial episodes.',
            'Save raw traces, per-family exposure/behavior/contact/target-limit/current-cap metrics and fresh reload-verified model/normalizer/Adam lineage.',
            'Run the frozen full held-out evaluation suite; use strict scorer thresholds and zero illegal-contact qualification unchanged.',
            'Retain every failure, stage checkpoint and protected baseline. Reward cannot promote a checkpoint or unlock a geometry stage.'],
        'remaining_native_work':['Foundation candidate benchmark/admission pending unless supplied.',
            'Source-seated perturbed reset adapter and exact episode trajectory/domain admission must be implemented and independently checked.',
            'Native integration smoke must verify recorder/reset lifecycle and sampled randomization values.',
            'No walking, recovery, terrain robustness or autonomous navigation success established by this local proposal.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--foundation',type=Path);p.add_argument('--domain',type=Path)
    p.add_argument('--domain-sha256');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    load=lambda p:json.loads(p.read_text()) if p else None
    value=build_proposal(load(args.foundation),load(args.domain),args.domain_sha256)
    value['source_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    with args.output.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'proposal':str(args.output.resolve()),'launch_ready':False,'training_performed':False}))


if __name__=='__main__':main()
