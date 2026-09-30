"""Submit one smoke first; fixed two-seed training requires its passing receipt."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'compute'))
from home_ray import Client

parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['smoke', 'training'])
parser.add_argument('--anchor', type=Path, default=ROOT / 'anchor.pt')
args = parser.parse_args()
c = Client()  # HOME_RAY_URL and HOME_RAY_TOKEN_FILE are operator-managed.
PRIVATE = ROOT / 'private-results'; PRIVATE.mkdir(exist_ok=True)
assert hashlib.sha256(args.anchor.read_bytes()).hexdigest() == '2d1a028358803e845e58c10d22bcb80b4c9ae343f679e2b4c4d56846b2cfe3cd'
manifest = json.loads((ROOT / 'campaign.json').read_text())
files = {name: (ROOT / name).read_text() for name in (
    'run.py', 'launch.py', 'race_env.py', 'v89_gpu_env.py', 'cuda_policy.py',
    'reward_spec.py', 'qualification.py', 'course.json', 'campaign.json')}
files['anchor.b64'] = base64.b64encode(args.anchor.read_bytes()).decode()
files['source-hashes.json'] = json.dumps({name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}, indent=2)
receipt = PRIVATE / (args.stage + '-jobs.json')
if receipt.exists(): raise SystemExit('Submission receipt already exists; inspect job states before any new submission.')
if args.stage == 'training':
    smoke_jobs = json.loads((PRIVATE / 'smoke-jobs.json').read_text())
    smoke = c.get_job_results(smoke_jobs[0]['id'], output_dir=PRIVATE / 'smoke-results')
    assert smoke['job']['state'] == 'SUCCEEDED'
    import base64 as b64
    result = json.loads(b64.b64decode(smoke['artifacts_base64']['summary.json']))
    assert result['status'] == 'smoke-passed'
    current_hashes = json.loads(files['source-hashes.json'])
    saved_hashes = json.loads((PRIVATE / 'smoke-source-hashes.json').read_text())
    assert current_hashes == saved_hashes, 'Source changed since passing smoke'
seeds = [99099] if args.stage == 'smoke' else manifest['training_seeds']
artifacts = ['runtime.json', 'summary.json', 'progress.json', 'progress.jsonl',
             'source-hashes.json', 'selection.json']
if args.stage == 'training':
    artifacts += [f'driver-{u}.pt' for u in manifest['checkpoint_updates']]
    artifacts += [f'selection-anchor-{i}.json' for i in range(2)]
    artifacts += [f'selection-{u}-{i}.json' for u in manifest['checkpoint_updates'] for i in range(2)]
    artifacts += [f'final-{name}-{i}.json' for name in ('anchor', 'candidate') for i in range(3)]
else:
    artifacts += ['smoke-reference.json']
specifications = [dict(name=f'duckwing-mjwarp-push-r1-{args.stage}-{seed}',
                      command=['bash', '-c', 'exec "$HOME/microduck-lab/upstream/microduck_rl/.venv/bin/python" launch.py'],
                      files=files, env={'TRAINING_SEED': str(seed), 'TRAINING_SMOKE': '1' if args.stage == 'smoke' else '0',
                                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'},
                      artifacts=artifacts, cpus=4, gpus=1, memory_gb=12,
                      capabilities=manifest['resources']['capabilities'],
                      timeout_s=1200 if args.stage == 'smoke' else manifest['timeout_s_per_job'],
                      idempotent=False, system_retries=0) for seed in seeds]
jobs = c.submit_batch(specifications)
receipt.write_text(json.dumps(jobs, indent=2) + '\n')
if args.stage == 'smoke': (PRIVATE / 'smoke-source-hashes.json').write_text(files['source-hashes.json'])
print(json.dumps([{k: job.get(k) for k in ('id', 'name', 'state')} for job in jobs], indent=2))
