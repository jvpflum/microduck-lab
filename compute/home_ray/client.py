"""Dependency-free client: no Ray installation needed on an agent's laptop."""
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

class Client:
    def __init__(self, url=None, token_file=None):
        self.url = (url or os.environ.get('HOME_RAY_URL', 'http://127.0.0.1:18787')).rstrip('/')
        self.token = Path(token_file or os.environ.get('HOME_RAY_TOKEN_FILE', '~/.config/home-ray/api-token')).expanduser().read_text().strip()

    def request(self, path, data=None):
        req = Request(self.url + path, data=None if data is None else json.dumps(data).encode(),
                      headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        with urlopen(req, timeout=60) as response:
            return json.load(response)

    def submit_job(self, command, **options):
        return self.submit_batch([dict(command=command, **options)])[0]

    def submit_batch(self, jobs):
        return self.request('/jobs', {'jobs': jobs})['jobs']

    def get_cluster_status(self):
        return self.request('/cluster')

    def get_job_status(self, job_id):
        return self.request('/jobs/' + job_id)

    def get_job_results(self, job_id, output_dir=None):
        result = self.request('/jobs/' + job_id + '/results')
        if output_dir is not None:
            root = Path(output_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            (root/'metadata.json').write_text(json.dumps(result['job'], indent=2))
            for name, data in result['artifacts_base64'].items():
                target = (root/'artifacts'/name).resolve()
                if not target.is_relative_to(root/'artifacts'):
                    raise ValueError('Invalid artifact path returned by server')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(data))
            for stream in ('stdout', 'stderr'):
                offset = 0
                with (root/(stream+'.log')).open('w') as f:
                    while True:
                        page = self.get_logs(job_id, offset=offset, stream=stream)
                        f.write(page['text'])
                        if page['next_offset'] == offset:
                            break
                        offset = page['next_offset']
        return result

    def get_logs(self, job_id, attempt=None, offset=0, stream='stdout'):
        status = self.get_job_status(job_id)
        attempt = attempt or status['attempt']
        return self.request(f'/jobs/{job_id}/logs?attempt={attempt}&offset={offset}&stream={stream}')

    def cancel_job(self, job_id):
        return self.request('/jobs/' + job_id + '/cancel', {})

    def retry_job(self, job_id):
        return self.request('/jobs/' + job_id + '/retry', {})
