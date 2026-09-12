import argparse
import json
from pathlib import Path
from .client import Client
p = argparse.ArgumentParser(prog='python -m home_ray')
s = p.add_subparsers(dest='action', required=True)
s.add_parser('cluster')
s.add_parser('list')
for action in ('submit','batch'):
    s.add_parser(action).add_argument('file')
for action in ('status','results','cancel','retry','logs'):
    q=s.add_parser(action)
    q.add_argument('job_id')
    if action == 'logs':
        q.add_argument('--stream', choices=['stdout','stderr'], default='stdout')
        q.add_argument('--offset', type=int, default=0)
    if action == 'results':
        q.add_argument('--output')
a=p.parse_args()
c=Client()
if a.action == 'cluster': r=c.get_cluster_status()
elif a.action == 'list': r=c.request('/jobs')
elif a.action in ('submit','batch'):
    value=json.loads(Path(a.file).read_text())
    r=c.submit_batch([value] if a.action == 'submit' else value)
elif a.action == 'logs': r=c.get_logs(a.job_id, offset=a.offset, stream=a.stream)
elif a.action == 'results' and a.output:
    r=c.get_job_results(a.job_id, output_dir=a.output)
    r={'id':a.job_id, 'state':r['job']['state'], 'saved_to':a.output}
else:
    fn={'status':c.get_job_status,'results':c.get_job_results,'cancel':c.cancel_job,'retry':c.retry_job}[a.action]
    r=fn(a.job_id)
print(json.dumps(r, indent=2))
