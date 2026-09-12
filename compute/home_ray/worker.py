"""One Ray task owns one subprocess tree and streams output to durable head storage."""
import base64
import json
import os
import platform
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def execute(spec, job_id, attempt, endpoint, token):
    import ray
    def call(action, data):
        req = Request(f'{endpoint}/internal/{job_id}/{attempt}/{action}', data=json.dumps(data).encode(),
                      headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
        with urlopen(req, timeout=10) as r:
            return json.load(r)

    node = {'hostname': socket.gethostname(), 'architecture': platform.machine(),
            'node_id': ray.get_runtime_context().get_node_id(),
            'gpu_ids': ray.get_runtime_context().get_accelerator_ids(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    start = call('start', {'node': node, 'time': time.time()})
    if start.get('cancel'):
        return {'state':'CANCELLED', 'node':node}
    root = Path(os.environ.get('HOME_RAY_WORK', str(Path.home()/'home-ray-work')))
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'{job_id}-{attempt}-', dir=root))
    for name, contents in spec['files'].items():
        p = work/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    env = os.environ.copy()
    # Child workloads don't need the cluster's bearer credentials.
    for key in ('RAY_AUTH_TOKEN', 'RAY_AUTH_TOKEN_PATH', 'HOME_RAY_API_TOKEN'):
        env.pop(key, None)
    env.update(spec['env'])
    env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '') if spec['gpus'] else ''
    env['OMP_NUM_THREADS'] = str(max(1, int(spec['cpus'])))
    env['HOME_RAY_JOB_ID'], env['HOME_RAY_ATTEMPT'] = job_id, str(attempt)
    env['HOME_RAY_OUTPUT_DIR'] = str(work)
    started = time.time()
    offsets = {'stdout':0, 'stderr':0}
    proc = None
    state = 'FAILED'
    reason = None
    def kill_tree():
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    try:
        with (work/'stdout.log').open('wb') as out, (work/'stderr.log').open('wb') as err:
            proc = subprocess.Popen(spec['command'], cwd=work, env=env, stdout=out, stderr=err, start_new_session=True)
            while True:
                for stream in offsets:
                    with (work/(stream+'.log')).open('rb') as f:
                        f.seek(offsets[stream])
                        chunk = f.read(256*1024)
                    if chunk:
                        call('log', {'stream':stream, 'offset':offsets[stream], 'data':base64.b64encode(chunk).decode()})
                        offsets[stream] += len(chunk)
                control = call('heartbeat', {'time':time.time()})
                if control.get('cancel'):
                    state = 'CANCELLED'
                    kill_tree()
                    break
                if time.time() - started > spec['timeout_s']:
                    reason = 'timeout'
                    kill_tree()
                    break
                if proc.poll() is not None:
                    state = 'SUCCEEDED' if proc.returncode == 0 else 'FAILED'
                    # Drain all bytes, including output emitted immediately before exit.
                    if all(offsets[s] == (work/(s+'.log')).stat().st_size for s in offsets):
                        break
                time.sleep(.25)
    finally:
        # A completed command may have left children; release the GPU only after cleanup.
        kill_tree()
    artifacts = {}
    total = 0
    for name in spec['artifacts']:
        p = work/name
        if p.exists():
            if not p.is_file() or not p.resolve().is_relative_to(work.resolve()):
                raise ValueError('Artifact must be an in-directory regular file')
            total += p.stat().st_size
            if total > 32*1024*1024:
                raise ValueError('Inline artifact limit is 32 MiB; use external storage for large outputs')
            artifacts[name] = base64.b64encode(p.read_bytes()).decode()
    return {'state':state, 'node':node, 'returncode':proc.returncode, 'reason':reason,
            'started':started, 'finished':time.time(), 'workdir':str(work), 'artifacts':artifacts}
