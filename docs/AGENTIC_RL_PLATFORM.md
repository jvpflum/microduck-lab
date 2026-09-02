# DuckLab as an agentic robotics RL platform

DuckLab's product boundary is intentionally small: a coding agent performs the
open-ended research, while the app makes the work observable, comparable, and
easy to try. The dashboard is not a no-code reward designer and does not require
a new UI workflow for every idea.

## The operating loop

1. A user gives Codex a robotics goal in plain language.
2. Codex inspects the robot and prior evidence, proposes a falsifiable plan,
   writes or adapts the task/reward/evaluator, checks resources, and launches
   the job.
3. The job publishes a generic run receipt containing its goal, status,
   progress, key metrics, immutable artifact hashes, and simulation/report
   links.
4. DuckLab renders that receipt next to live training and mature benchmark
   programs. A person watches progress, opens the correct simulator, reviews
   evidence, and decides what to test next.
5. If an experiment becomes a recurring capability, it can adopt a built-in
   adapter with a deterministic evaluator and promotion gates. Hardening comes
   after discovery, not before it.

This gives novel projects a zero-scaffolding entry path while preserving strong
standards for results that claim to be leaders.

## Generic agent run receipt

Start with [`examples/agent-run-receipt.json`](../examples/agent-run-receipt.json).
The only required concepts are a stable run ID, project/title/goal, status, and
schema version. Metrics, progress, artifacts, hypotheses, and action links are
generic lists.

```bash
cp examples/agent-run-receipt.json /tmp/my-run.json
# Let the coding agent update /tmp/my-run.json with real paths and metrics.
make publish-agent-run RECEIPT=/tmp/my-run.json
make bench-dashboard
```

Publishing the same `run_id` updates the card atomically. Artifact paths must
exist, and DuckLab computes or verifies their SHA-256 values. A receipt cannot
execute shell code; it can only expose reviewed HTTP(S) or dashboard-relative
links. The coding agent remains responsible for launching and supervising the
training, evaluation, and viewer processes.

For the incoming RTX 5090 front-flip result, use project `MicroDuck front flip`,
include the normalizer-aware ONNX and 256-episode evaluation as artifacts, and
link the live or finished simulator. The established `backflip` internal task
token is retained only for compatibility; all user-facing text says Front flip.

## Built-in adapters are the hardened path

[`config/robotics-capabilities.json`](../config/robotics-capabilities.json)
describes stable robot/task adapters already understood by the app. It contains
training launchers, deterministic evaluators, suites, and arena settings for
the current MicroDuck programs. The dashboard and backend read this catalog,
so duplicated task maps no longer need to be changed in several files.

The catalog is not required for exploration. Add an adapter only when a task
needs one-click training, an app-launched simulator, or an official promotion
workflow. New robots can supply their own policy format, evaluator, and viewer;
the generic agent receipt remains unchanged.

## Two distinct notions of progress

- **Research progress** is whatever evidence answers the experiment's stated
  hypothesis. The agent chooses and publishes the relevant metrics.
- **Benchmark progress** uses a frozen evaluator, physical parameters, baseline,
  and promotion rule. A mature capability must never silently redefine these
  after seeing a result.

For skating, DuckWing V67 remains the frozen leader under exact Race5 physics.
For front flip, the current evaluator separately measures 256 unassisted
episodes, takeoff/landing/settling rates, body strikes, clearance, forward and
off-axis rotation, and horizontal drift. Those measurements are not forced into
the skating table.

## Lightweight deployment

DuckLab remains a local Python HTTP server with static generated HTML and the
upstream simulator. It requires no cloud control plane or database. A user can
run it on a GPU workstation and forward the dashboard/viewer ports over SSH.
Raw trainer state stays local; Git carries code, compact release policies,
checksums, schemas, and scrubbed evidence.

The near-term hardening priorities are adapter conformance tests, signed or
content-addressed result bundles, task-specific dashboard summaries, and an
agent heartbeat/watchdog. These should strengthen the evidence boundary without
turning the app into a second orchestration agent.
