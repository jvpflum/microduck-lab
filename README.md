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

## Latest validated control model: DuckWing V89

[V89](releases/v89/README.md) adds learned high-speed braking to the frozen
V80 driving policy. In 16 independent held-out simulation episodes, V89
completed every high-speed brake healthily; V80 had two unhealthy brakes.
The following comparisons use the 14 episodes where both policies braked
healthily, so falling never counts as a successful fast stop.

| High-speed braking metric | V80 | V89 |
| --- | ---: | ---: |
| Mean lateral drift | 0.599 m | **0.325 m** |
| Worst lateral drift | 1.179 m | **0.787 m** |
| Mean stopping time | 2.930 s | **2.371 s** |
| Mean stopping path | 1.812 m | **1.493 m** |
| Healthy brakes, all held-out episodes | 14/16 | **16/16** |

Mean braking drift fell **45.7%**. Race5 and manual-agility phase metrics
matched V80 exactly on all 16 paired seeds. Low-speed braking remained
unchanged. This is a qualified braking-control variant, not a new speed
record. V80 remains the immutable speed benchmark; the 5 mph goal and
physical-robot validation remain open. V90 training is underway through Ray
and has not been qualified or released.

Download the inference-complete [V89 ONNX policy](releases/v89/duckwing-v89-braking-control.onnx)
and verify its [checksums](releases/v89/SHA256SUMS). Its 15,616-value temporal
input requires the history contract in the release notes; it cannot be fed a
single 61-value frame.

## Speed benchmark: DuckWing V80

V80 is the simulation-qualified speed benchmark. It is evaluated in
deterministic CPU MuJoCo with wheel `frictionloss=0.003`, motor current limit
`1.75 A`, and the frozen Race5 line controller. It passes all 15 retained
qualification gates, satisfies the repository's formal advancement rule over
V67, and wins all six direct Race5 comparisons against the Pollen roller.

| Metric | V80 | Pollen roller | V80 vs Pollen |
| --- | ---: | ---: | ---: |
| Race sustained speed | **2.267 mph** | 1.066 mph | 2.13× |
| Verified top speed (0.5 s) | **3.084 mph** | 1.283 mph | 2.40× |
| 100-ft elapsed time | **25.729 s** | 57.589 s | 55.3% sooner |
| First-second acceleration | **0.466 m/s²** | 0.323 m/s² | 44.3% higher |
| Maximum lateral drift | **0.403 ft** | 1.250 ft | 67.8% less |
| Maximum heading error | **9.54°** | 11.06° | 13.7% less |

V80 preserves V67 exactly for commands at or below `0.5 m/s`. Above that
threshold, it adds a small state-dependent residual to the six propulsion
joints. Compared with V67, it finishes 100 ft sooner, improves sustained and
top speed, cuts long-run drift nearly in half, reduces heading error and tilt,
and improves the five-episode randomized-start speed test from 2.438 to 2.497
mph with 100% survival.

The 5 mph target is not reached. The protected speed donor's 5.405 mph result
was a peak measured at zero wheel friction; it is useful research evidence,
not an official-friction record. See the [V80 release notes](releases/v80/README.md),
the frozen [V67 predecessor](releases/v67/README.md), and the historical
[V68](releases/v67/V68-SEARCH.md) and [V69](releases/v67/V69-SEARCH.md) searches.

These are simulation results, not a physical-robot claim or independent
certification.

## Download only the model

For the original 61-input speed benchmark, download the public
[DuckWing V80 model release on Hugging Face](https://huggingface.co/juicenv/duckwing-v80-roller-skating):

```bash
hf download juicenv/duckwing-v80-roller-skating policy.onnx \
  --local-dir policies/duckwing-v80
```

The model repository contains only the inference-complete ONNX policy, model
card, Apache-2.0 license, machine-readable manifest, checksums, and measured
evaluation evidence. The source [V80 release directory](releases/v80/) remains
the canonical in-repository copy and includes the exact construction recipe.

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

The V80 release is inference-complete and includes checksums, metrics, the full
Race5 evaluation, clean and randomized-start screens, and the exact ONNX
artifact. A clean clone can verify the public artifact without private
optimizer checkpoints:

```bash
(cd releases/v80 && sha256sum -c SHA256SUMS)
./scripts/verify-artifact.sh \
  "$(pwd)/releases/v80/duckwing-v80-high-command-residual.onnx"
```

The V80 artifact is rebuilt from the immutable V67 policy by
`tools/build_command_gated_dynamic_residual_policy.py`. The checked-in
evaluation files use the official `0.003` friction and `1.75 A` contract.

## Programs and common commands

The declarative capability catalog is at
[`config/robotics-capabilities.json`](config/robotics-capabilities.json).
Built-in adapters are hardened conveniences, not a requirement for novel
research.

| Program | Purpose | Useful commands |
| --- | --- | --- |
| MicroDuck skating | Official-friction speed, steering, braking, and stability | `make race5-smoke`, `make train-race5`, `./scripts/verify-artifact.sh releases/v80/duckwing-v80-high-command-residual.onnx` |
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

## Ray compute and MinIO artifacts

The portable [compute package](compute/README.md) provides authenticated job
submission, durable receipts, resource-based Ray scheduling, cancellation and
bounded retries. The head also runs jobs when its resources match. ARM64 GB10
and x86 RTX workers can share the cluster; application environments must be
validated independently on each architecture.

Optional MinIO/S3 artifact transfers use content hashes and credentials from
operator environment variables. The dashboard remains local; MinIO is artifact
storage, not a public model distribution endpoint. See the
[privacy and publishing guide](docs/PRIVACY.md).

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
candidate must finish 100 ft faster, preserve the complete qualification gate,
avoid regressions in long-run drift and heading, and retain low-speed agility
and measured controller usage. Sustained, trap, and top speed then rank
qualified racers. Other capabilities define their own frozen contract. Every
result is labeled `promoted`, `rejected`, or `exploratory`.

The detailed history from Pollen's roller baseline through V67, including
failed transfers and the V68/V69 negative results, is in the [DuckWing research
paper](docs/DUCKWING_RESEARCH_PAPER.md), with a [shareable Word
version](docs/DUCKWING_RESEARCH_PAPER.docx). The [V80 release
notes](releases/v80/README.md) continue that record with the current result.

## Repository map

- [`AGENTS.md`](AGENTS.md): instructions for Codex and other coding agents.
- [`docs/AGENTIC_RL_PLATFORM.md`](docs/AGENTIC_RL_PLATFORM.md): generic
  agent-to-dashboard architecture.
- [`docs/CODING_AGENT_WORKFLOW.md`](docs/CODING_AGENT_WORKFLOW.md): reproducible
  setup, exact skating prompt, resource hooks, and definition of done.
- [`docs/POLICY_BENCH.md`](docs/POLICY_BENCH.md): dashboard and review workflow.
- [`docs/RACE5.md`](docs/RACE5.md): exact skating task and private continuation.
- [`releases/v80/`](releases/v80/): current leader, checksums, metrics, and
  complete evaluation evidence.
- [`releases/v67/`](releases/v67/): frozen predecessor and V68/V69 search
  history.
- [`upstream/microduck_rl`](upstream/microduck_rl): pinned training/runtime
  source kept as a Git submodule.

## Contributing

Read `AGENTS.md` and the relevant capability/release notes first. Keep changes
small, reproducible, and attributable. Run `make test`, `git diff --check`,
artifact verification, and a machine-path/secret review before committing.
Do not overwrite a release or promote a model without complete evidence.
