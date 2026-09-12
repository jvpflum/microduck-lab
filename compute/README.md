# Ray compute and optional MinIO artifact storage

This portable package is the execution layer used for V89 and the ongoing V90
campaign. Ray places jobs by CPU, GPU, memory and capability requirements. The
head participates as a worker. Tested deployment: one ARM64 GB10 head and two
x86 RTX 5090 Linux/WSL2 workers, Ray 2.58.0, Python 3.12.3. A 500-job CUDA
validation finished in 124.53 seconds with work distributed across all three
GPUs. These infrastructure checks do not qualify a robotics policy.

Install `pip install -e './compute[node,storage]'` on the head, and the `node`
extra on workers. Keep all nodes on the same Ray/Python versions. Configure
private connectivity and Ray authentication outside Git. Start Ray on the head
with `ray start --head`; join workers with `ray start --address=<private-head>:6379`.
Set architecture capability resources when starting each node, such as
`--resources='{"arch:arm64":1,"cuda":1,"gpu:gb10":1}'`, or
`--resources='{"arch:x86_64":1,"cuda":1,"gpu:rtx5090":1}'`.

For the authenticated durable submission service, set `HOME_RAY_STATE` to a
private persistent directory, `HOME_RAY_API_TOKEN_FILE` to a mode-0600 token
file, and `HOME_RAY_URL` to the private callback URL reachable by every worker.
Run `python -m home_ray.server` on the head. The API binds to the host and port in `HOME_RAY_URL`. Restrict Ray
and API ports to loopback or your private overlay; never expose them publicly.
Run node/API processes under operator-managed service supervision.

Agents need only the dependency-free client and the private URL/token file:

```python
from home_ray import Client
c = Client()  # HOME_RAY_URL and HOME_RAY_TOKEN_FILE; defaults to loopback
job = c.submit_job(['python', 'train.py'], name='training', cpus=8, gpus=1,
                   memory_gb=12, capabilities=['cuda'], timeout_s=7200)
status = c.get_job_status(job['id'])
results = c.get_job_results(job['id'], output_dir='private-results')
```

Use an environment that exists on eligible workers; Ray does not install your
science stack or translate an architecture-specific binary. Require `arch:arm64`
for authoritative ARM64 evaluation and `arch:x86_64` for x86 portability checks. GPU
resources control placement, not VRAM isolation. The API provides atomic
batches, SQLite receipts, logs, small artifacts, cancellation, and bounded
idempotent infrastructure retries. It does not provide preemption, dependency
DAGs, high availability, or application-science validation. WSL workers need a
persistent Linux session and a Windows logon keep-alive; unattended cold boot
is not established.

## MinIO/S3

MinIO is optional private artifact storage, separate from Ray's submission
ledger and small inline result transport. Set `HOME_RAY_S3_ENDPOINT`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally
`AWS_DEFAULT_REGION` through your operator secret mechanism. Create a private
bucket and a least-privilege scoped account outside the repository.

```python
from home_ray.storage import upload, download
ref = upload('checkpoint.onnx', bucket='research-artifacts')
download(ref, 'restored.onnx')
```

Keys include SHA-256; downloads verify the expected hash before replacing the
destination. Attach object references to private run receipts rather than
embedding large checkpoint files in job submissions. Transfers are explicit;
the broker does not automatically archive every job to MinIO. Public releases
contain reviewed inference models and curated metrics only, never credentials,
private endpoints, raw receipts or optimizer state.

Run `PYTHONPATH=compute python3 -m unittest discover -s compute/tests`.
