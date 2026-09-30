"""Verified source staging and an exclusive per-seed trainer lock."""
import base64
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

for name, digest in json.loads(Path('source-hashes.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
Path('anchor.pt').write_bytes(base64.b64decode(Path('anchor.b64').read_text()))
assert hashlib.sha256(Path('anchor.pt').read_bytes()).hexdigest() == '2d1a028358803e845e58c10d22bcb80b4c9ae343f679e2b4c4d56846b2cfe3cd'
seed = int(os.environ['TRAINING_SEED'])
lock_root = Path.home() / 'home-ray-locks'; lock_root.mkdir(exist_ok=True)
with (lock_root / f'duckwing-mjwarp-push-r1-{seed}.lock').open('a') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, 'run.py', '--seed', str(seed)]
    if os.environ.get('TRAINING_SMOKE') == '1': command.append('--smoke')
    raise SystemExit(subprocess.run(command).returncode)
