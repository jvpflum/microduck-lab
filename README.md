# DuckLab Robotics RL

DuckLab is a lightweight, agent-first workbench for training and evaluating
small-robot policies. A coding agent chooses the research approach, writes the
task or reward when needed, launches and monitors jobs, evaluates artifacts,
and publishes simulator/report links. The dashboard is the human evidence
surface; new research does not require a new dashboard wizard.

The flagship case study is DuckWing, a Pollen MicroDuck roller-skating policy.
Front flip is the first separate program. Other robots and behaviors can enter
through the same generic run-receipt interface without being forced into the
skating benchmark.

## Current leader: DuckWing V67

V67 is the definitive simulation-qualified skating model. It is evaluated in
deterministic CPU MuJoCo with wheel `frictionloss=0.003`, motor current limit
`1.75 A`, and the frozen Race5 line controller. It passes all 15 retained
qualification gates and wins all 9 comparable Pollen dimensions.

| Metric | V67 | Pollen roller | V67 vs Pollen |
| --- | ---: | ---: | ---: |
| Race sustained speed | **2.240 mph** | 1.066 mph | 2.10× |
| Verified top speed (0.5 s) | **3.060 mph** | 1.283 mph | 2.39× |
| 100-ft elapsed time | **25.815 s** | 57.589 s | 55.2% sooner |
| First-second acceleration | **0.471 m/s²** | 0.323 m/s² | 46.0% higher |
| Maximum lateral drift | **0.775 ft** | 1.250 ft | 38.0% less |
| Maximum heading error | **10.22°** | 11.06° | 7.6% less |

The 5 mph target is not reached. V68 and V69 produced faster experimental
variants, but every candidate regressed at least one required control or
stability measurement, so V67 remains the leader. See the [V67 release
notes](releases/v67/README.md), [V68 report](releases/v67/V68-SEARCH.md), and
[V69 report](releases/v67/V69-SEARCH.md).

These are simulation results, not a physical-robot claim or independent
certification.

## Download only the model

If you only need the current skating policy, start with the [V67 release
directory](releases/v67/). It contains the [standalone ONNX
artifact](releases/v67/duckwing-v67-joint-specialist-fusion.onnx), SHA-256
checksum, leader metrics, and evaluation summary; the rest of this platform is
not required to copy or inspect those files.

The recommended next publication is that same small bundle as a separate
model-only GitHub repository (for example, `duckwing-v67-model`). It should
contain only the ONNX model, model card, license/provenance, checksum, and
measured metrics—not the simulator, training checkpoints, or dashboard. Once
published, this paragraph can link directly to that model repository without
making the platform depend on it.

## Ask an agent to run RL

Start with a plain-language goal. Repository instructions in
[`AGENTS.md`](AGENTS.md) tell Codex to inspect prior evidence and resources,
state a falsifiable hypothesis, choose the appropriate technique, prevent
duplicate jobs, launch a bounded run, publish progress, evaluate checkpoints,
and report a promoted, rejected, or exploratory result.

For example:

> Read `AGENTS.md`. Make this robot perform a stable front flip. Choose the
> best RL approach, run it, monitor it, evaluate it, and publish the evidence
> and simulator to DuckLab.

The agent publishes a generic schema-v1 receipt. It can contain any task's
metrics, artifacts, hypotheses, progress, and simulator/report actions:

```bash
make publish-agent-run RECEIPT=/path/to/agent-run-receipt.json
make bench-dashboard
```

Publishing the same `run_id` updates the dashboard card atomically. Artifact
SHA-256 values are computed or verified. Receipts expose reviewed links only;
they cannot execute shell commands. See the [agentic platform design](docs/AGENTIC_RL_PLATFORM.md),
[coding-agent workflow](docs/CODING_AGENT_WORKFLOW.md), and [example receipt](examples/agent-run-receipt.json).

## Quick start

Requirements: Git, Python 3.12, network access for first setup, and either a
DGX Spark/GB10 or a Linux NVIDIA GPU. Windows RTX workers use Ubuntu through
WSL2; see [Windows 5090 setup](docs/WINDOWS_5090.md).

```bash
git clone --recurse-submodules https://github.com/jvpflum/microduck-lab.git
cd microduck-lab
make bootstrap
make preflight
make test
make import-pollen-baselines   # optional; needed for factory comparisons
make bench-dashboard
```

Open `http://localhost:8091`. Forward the port over SSH when the checkout is
on a remote workstation:

```bash
ssh -L 8091:localhost:8091 <ssh-user>@<spark-address>
```

The dashboard can open the colorful Pollen browser arena for finished policies
and a lightweight live-training viewer for active jobs. It also shows agent
receipts, immutable artifacts, evaluation reports, and the appropriate task
simulator. The dashboard is local and requires no cloud control plane.

## Repository topology

The simulator is intentionally a pinned submodule, not a copy hidden inside
this repository:

| Path | Role | Pinned source |
| --- | --- | --- |
| `upstream/microduck_rl` | Training fork and CPU MuJoCo evaluation | [microduck_rl](https://github.com/jvpflum/microduck_rl) |
| `upstream/microduck` | Robot runtime and model | [Pollen microduck](https://github.com/pollen-robotics/microduck) |
| `upstream/microduck-simulator` | Browser arena and simulator | [microduck-simulator](https://github.com/jvpflum/microduck-simulator) |

`git clone --recurse-submodules` checks out the exact commits recorded by
DuckLab. Keeping the browser app separate preserves its own deployable Docker/
Vite lifecycle, Hugging Face Space mirror, LFS assets, and upstream history;
DuckLab can update the pointer only after validating a new simulator revision.
Deleting that GitHub repository would break fresh clones and the submodule URL.
If a single physical repository becomes a hard requirement, migrate it first
with a tested `git subtree`/vendoring plan, preserve its license and history,
update every build path, and archive the old remote only after the new clone and
Hugging Face deployment work.

## Verify the public leader

The V67 release is inference-complete and includes checksums, metrics, and the
exact ONNX artifact. A clean clone can reproduce the public next-step search
without private optimizer checkpoints:

```bash
(cd releases/v67 && sha256sum -c SHA256SUMS)
./scripts/verify-artifact.sh \
  "$(pwd)/releases/v67/duckwing-v67-joint-specialist-fusion.onnx"
make v69-search
```

V69 reconstructs the best rejected V68 challenger, applies a measured body-yaw
state guard, and evaluates 150 candidates under the official `0.003` friction
and `1.75 A` contract. It never changes the leader unless both speed measures
improve and every retained gate is preserved.

## Programs and common commands

The declarative capability catalog is at
[`config/robotics-capabilities.json`](config/robotics-capabilities.json).
Built-in adapters are hardened conveniences, not a requirement for novel
research.

| Program | Purpose | Useful commands |
| --- | --- | --- |
| MicroDuck skating | Official-friction speed, steering, braking, and stability | `make race5-smoke`, `make train-race5`, `make v69-search` |
| MicroDuck front flip | Unassisted takeoff, forward rotation, clean landing, settling | `make backflip-smoke`, `make train-backflip` |
| MicroDuck walking | Upstream locomotion reference | `make train-baseline` |

For a saved run, Policy Bench provides discovery, immutable registration,
evaluation, comparison, metrics, and review:

```bash
make bench-discover
make bench-list
./scripts/policy-bench.sh evaluate <run-id>
./scripts/policy-bench.sh compare <candidate-run-id> <baseline-run-id>
./scripts/policy-bench.sh metrics <run-id>
make bench-dashboard
```

The evaluator for a mature capability owns its physics and metrics. A front
flip is never ranked by skating speed, and a new robot does not need to pretend
it has MicroDuck's 61-observation/14-action contract.

## Training safely on shared hardware

DGX Spark/GB10 memory is unified. Before a heavy run, check available memory,
swap, disk, GPU process attribution, and existing trainers. Do not run full
training concurrently with memory-heavy inference. DuckLab's default resource
mode is `shared` and does not stop unrelated services.

For an explicitly authorized GPU training run, the optional
`training-priority` mode can stop and restore operator-provided services. See
the [resource workflow](docs/CODING_AGENT_WORKFLOW.md#heavy-gpu-runs-on-a-shared-spark).
CPU composition and evaluation searches do not require pausing Hermes or vLLM.

Training outputs and raw optimizer checkpoints stay local/ignored. Git carries
code, compact inference artifacts, checksums, schemas, and reviewed evidence.
Never treat an exported ONNX actor as a lossless PPO continuation checkpoint.

## Promotion rules

Reward curves and one impressive rollout are evidence, not promotion. A skating
candidate must be faster over 100 ft and have higher sustained speed while
preserving top speed, acceleration, drift, heading, tilt, bilateral contact,
idle hold, turns, and braking. Other capabilities define their own frozen
contract. Every result is labeled `promoted`, `rejected`, or `exploratory`.

The complete process and history—from Pollen's roller baseline through V67,
including failed transfers and the V68/V69 negative results—is in the
[DuckWing research paper](docs/DUCKWING_RESEARCH_PAPER.md), with a
[shareable Word version](docs/DUCKWING_RESEARCH_PAPER.docx).

## Repository map

- [`AGENTS.md`](AGENTS.md): instructions for Codex and other coding agents.
- [`docs/AGENTIC_RL_PLATFORM.md`](docs/AGENTIC_RL_PLATFORM.md): generic
  agent-to-dashboard architecture.
- [`docs/CODING_AGENT_WORKFLOW.md`](docs/CODING_AGENT_WORKFLOW.md): reproducible
  setup, exact skating prompt, resource hooks, and definition of done.
- [`docs/POLICY_BENCH.md`](docs/POLICY_BENCH.md): dashboard and review workflow.
- [`docs/RACE5.md`](docs/RACE5.md): exact skating task and private continuation.
- [`releases/v67/`](releases/v67/): current leader, checksums, metrics, and
  follow-up search reports.
- [`upstream/microduck_rl`](upstream/microduck_rl): pinned training/runtime
  source kept as a Git submodule.

## Contributing

Read `AGENTS.md` and the relevant capability/release notes first. Keep changes
small, reproducible, and attributable. Run `make test`, `git diff --check`,
artifact verification, and a machine-path/secret review before committing.
Do not overwrite a release or promote a model without complete evidence.
