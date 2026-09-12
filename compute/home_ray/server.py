"""Durable submission ledger; Ray alone places and queues executable tasks."""
import base64
import hmac
import json
import os
import sqlite3
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .spec import validate

TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED'}

class Broker:
    def __init__(self, root, ray, endpoint, token):
        from .worker import execute
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ray, self.endpoint, self.token = ray, endpoint, token
        self.task = ray.remote(max_retries=0)(execute)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root/'jobs.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        self.refs = {}
        # Never silently duplicate unknown work after control-plane loss.
        for job in self.jobs():
            if job['state'] not in TERMINAL and job['state'] != 'QUEUED':
                job.update(state='INTERRUPTED', error='Control service restarted; verify external effects before retry', finished=time.time())
                self.save(job)

    def jobs(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM jobs ORDER BY rowid')]

    def get(self, jid):
        row = self.db.execute('SELECT data FROM jobs WHERE id=?', (jid,)).fetchone()
        if not row:
            raise KeyError(jid)
        return json.loads(row[0])

    def save(self, job):
        self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', (job['id'], json.dumps(job)))
        self.db.commit()

    def submit(self, specs):
        if not isinstance(specs, list) or not 1 <= len(specs) <= 1000:
            raise ValueError('A batch must contain 1..1000 jobs')
        specs = [validate(s) for s in specs]
        result = []
        # Single transaction makes a batch all-or-nothing.
        with self.db:
            for spec in specs:
                job = dict(id=uuid.uuid4().hex, spec=spec, state='QUEUED', attempt=0, submitted=time.time(), attempts=[])
                self.db.execute('INSERT INTO jobs VALUES (?,?)', (job['id'], json.dumps(job)))
                result.append(job)
        return result

    def cluster(self):
        nodes = [dict(id=n['NodeID'], address=n['NodeManagerAddress'], alive=n['Alive'], resources=n.get('Resources', {})) for n in self.ray.nodes()]
        counts = {}
        for j in self.jobs():
            counts[j['state']] = counts.get(j['state'], 0)+1
        return dict(nodes=nodes, total=self.ray.cluster_resources(), available=self.ray.available_resources(), jobs=counts)

    def tick(self):
        with self.lock:
            # All queued tasks go to Ray. No polling for free hosts and no host selection.
            # Cap submission window only to bound driver memory, not to schedule resources.
            for job in self.jobs():
                if job['state'] != 'QUEUED' or len(self.refs) >= 1000:
                    continue
                spec = job['spec']
                job['attempt'] += 1
                job['state'] = 'PENDING'
                job['attempts'].append({'attempt':job['attempt'], 'submitted':time.time()})
                self.save(job)
                try:
                    ref = self.task.options(num_cpus=spec['cpus'], num_gpus=spec['gpus'],
                        memory=int(spec['memory_gb']*1024**3),
                        resources={c:0.001 for c in spec['capabilities']},
                        name=spec['name']).remote(spec, job['id'], job['attempt'], self.endpoint, self.token)
                    self.refs[ref] = (job['id'], job['attempt'])
                except Exception as exc:
                    job.update(state='FAILED', error=str(exc), finished=time.time())
                    self.save(job)
            if not self.refs:
                return
            ready, _ = self.ray.wait(list(self.refs), num_returns=len(self.refs), timeout=0)
            for ref in ready:
                jid, attempt = self.refs.pop(ref)
                job = self.get(jid)
                try:
                    result = self.ray.get(ref)
                    path = self.root/jid/str(attempt)
                    path.mkdir(parents=True, exist_ok=True)
                    # Preserve immutable per-attempt result and artifacts.
                    for name, data in result.pop('artifacts', {}).items():
                        target = path/'artifacts'/name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(base64.b64decode(data))
                    (path/'result.json').write_text(json.dumps(result, indent=2))
                    job.update(state=result['state'], result=result, finished=time.time())
                    job['attempts'][-1].update(result=result)
                except Exception as exc:
                    error = dict(type=type(exc).__name__, message=str(exc), time=time.time())
                    job['attempts'][-1]['error'] = error
                    retryable = type(exc).__name__ in ('NodeDiedError', 'WorkerCrashedError', 'ObjectLostError')
                    if job['state'] == 'CANCELLING':
                        job.update(state='CANCELLED', finished=time.time())
                    elif retryable and attempt <= job['spec']['system_retries']:
                        job.update(state='QUEUED', error=error)
                    else:
                        job.update(state='FAILED', error=error, finished=time.time())
                self.save(job)

    def internal(self, jid, attempt, action, data):
        job = self.get(jid)
        if attempt != job['attempt'] or job['state'] in TERMINAL:
            return {'cancel':True}
        path = self.root/jid/str(attempt)
        path.mkdir(parents=True, exist_ok=True)
        if action == 'start':
            job['node'] = data['node']
            job['attempts'][-1].update(data)
            if job['state'] != 'CANCELLING':
                job['state'] = 'RUNNING'
        elif action == 'heartbeat':
            job['heartbeat'] = data['time']
        elif action == 'log':
            stream = data['stream']
            if stream not in ('stdout','stderr'):
                raise ValueError('Invalid stream')
            target = path/(stream+'.log')
            chunk = base64.b64decode(data['data'], validate=True)
            offset = data['offset']
            if not isinstance(offset, int) or offset < 0:
                raise ValueError('Invalid offset')
            with target.open('a+b') as f:
                f.seek(0, 2)
                current = f.tell()
                if offset == current:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                elif offset + len(chunk) > current:
                    raise ValueError('Noncontiguous log upload')
        else:
            raise ValueError('Unknown internal action')
        self.save(job)
        return {'cancel':job['state'] == 'CANCELLING'}

    def route(self, method, raw_path, data):
        u = urlparse(raw_path)
        parts = u.path.strip('/').split('/')
        if parts == ['cluster'] and method == 'GET':
            return self.cluster()
        if parts == ['jobs']:
            return {'jobs':self.submit(data['jobs']) if method == 'POST' else self.jobs()}
        if len(parts) == 5 and parts[0] == 'internal' and method == 'POST':
            raise ValueError('Invalid path')
        if len(parts) == 4 and parts[0] == 'internal' and method == 'POST':
            return self.internal(parts[1], int(parts[2]), parts[3], data)
        if len(parts) < 2 or parts[0] != 'jobs':
            raise KeyError(u.path)
        job = self.get(parts[1])
        if len(parts) == 2 and method == 'GET':
            return job
        action = parts[2]
        if method == 'POST' and action == 'cancel':
            if job['state'] == 'QUEUED':
                job.update(state='CANCELLED', finished=time.time())
            elif job['state'] not in TERMINAL:
                pending = job['state'] == 'PENDING'
                job['state'] = 'CANCELLING'
                if pending:
                    for ref, (jid, _) in self.refs.items():
                        if jid == job['id']:
                            self.ray.cancel(ref, force=False)
            self.save(job)
            return job
        if method == 'POST' and action == 'retry':
            if job['state'] not in TERMINAL:
                raise ValueError('Cannot retry nonterminal work')
            spec = dict(job['spec'], retry_of=job['id'])
            return self.submit([spec])[0]
        if method == 'GET' and action == 'results':
            files = {}
            root = self.root/job['id']/str(job['attempt'])/'artifacts'
            if root.exists():
                for f in root.rglob('*'):
                    if f.is_file():
                        files[str(f.relative_to(root))] = base64.b64encode(f.read_bytes()).decode()
            return dict(job=job, artifacts_base64=files)
        if method == 'GET' and action == 'logs':
            q = parse_qs(u.query)
            attempt = int(q.get('attempt',[job['attempt']])[0])
            offset = max(0, int(q.get('offset',[0])[0]))
            stream = q.get('stream',['stdout'])[0]
            if stream not in ('stdout','stderr'):
                raise ValueError('Invalid stream')
            path = self.root/job['id']/str(attempt)/(stream+'.log')
            chunk = b''
            if path.exists():
                with path.open('rb') as f:
                    f.seek(offset)
                    chunk = f.read(256*1024)
            return dict(text=chunk.decode(errors='replace'), next_offset=offset+len(chunk), attempt=attempt)
        raise KeyError(u.path)


def main():
    import ray
    import fcntl
    root = Path(os.environ['HOME_RAY_STATE'])
    root.mkdir(parents=True, exist_ok=True)
    lockfile = (root/'broker.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = Path(os.environ['HOME_RAY_API_TOKEN_FILE']).read_text().strip()
    endpoint = os.environ['HOME_RAY_URL']
    ray.init(address='auto', namespace='home-ray')
    broker = Broker(root, ray, endpoint, token)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.handle_request('GET')
        def do_POST(self):
            self.handle_request('POST')
        def handle_request(self, method):
            status = 200
            try:
                if not hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+token):
                    self.send_error(401)
                    return
                size = int(self.headers.get('Content-Length',0))
                if not 0 <= size <= 64*1024*1024:
                    raise ValueError('Request exceeds 64 MiB')
                data = json.loads(self.rfile.read(size)) if size else {}
                with broker.lock:
                    result = broker.route(method, self.path, data)
            except KeyError as exc:
                status, result = 404, {'error':str(exc)}
            except (ValueError, TypeError, IndexError) as exc:
                status, result = 400, {'error':str(exc)}
            except Exception:
                traceback.print_exc()
                status, result = 500, {'error':'Internal error; inspect service journal'}
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    host = urlparse(endpoint)
    server = ThreadingHTTPServer((host.hostname, host.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    while True:
        broker.tick()
        time.sleep(.2)

if __name__ == '__main__':
    main()
