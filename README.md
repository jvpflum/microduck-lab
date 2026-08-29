# MicroDuck Lab

Reproducible reinforcement-learning workspace for training Pollen Robotics'
MicroDuck on an NVIDIA DGX Spark before the physical robot arrives.

The lab pins Pollen's official `microduck_rl` project as a Git submodule and
adds guarded setup, validation, smoke-training, and baseline-training commands.
It does not fork or alter the robot's observation contract, actuator model,
domain randomization, or ONNX export path.

The product goal is a simple robotics teaching loop—sign in, describe a skill,
train, understand the result, promote a policy, and drive the robot. See
[docs/PRODUCT_VISION.md](docs/PRODUCT_VISION.md) for the user-facing contract
and staged platform roadmap.

## Requirements

- Linux ARM64 DGX Spark / GB10 with working NVIDIA drivers
- Python 3.12 and `venv`
- Git
- At least 20 GiB available unified memory for GPU commands
- Network access for the first dependency synchronization

## Start

```bash
git clone --recurse-submodules <repository-url>
cd microduck-lab
make bootstrap
make preflight
make test
make smoke
make verify-artifact
```

`make smoke` follows Pollen's required 64-environment, five-iteration gate.
Weights & Biases is disabled; Policy Bench remains the local system of record.
Training automatically exports the final checkpoint through Pollen's official
normalizer-aware ONNX path. `make verify-artifact` validates its 61-to-14
contract, metadata, finite CPU inference, and a 100-step CPU MuJoCo rehearsal.
Successful skating runs are then automatically verified, registered, scored in
the CPU deployment battery, and given training-curve metrics. They are never
automatically starred or promoted.

## Train a walking baseline

```bash
make train-baseline
```

Defaults are 4,096 parallel environments and 4,000 PPO iterations. Override
them without editing source:

```bash
DUCKLAB_ENVS=2048 DUCKLAB_ITERATIONS=1000 make train-baseline
```

Do not run a full training job concurrently with memory-heavy inference
services. This system uses unified memory; the preflight blocks when less than
20 GiB is available or swap usage exceeds 50%.

The dashboard offers two explicit resource modes for new runs:

- **Shared** keeps vLLM and Hermes inference online. It is the safe daytime
  default, but inference traffic may reduce or vary training throughput.
- **Training priority** temporarily stops the local Docker-backed vLLM service,
  trains with the GPU reserved for RL, and restores vLLM plus the Hermes health
  gate when training exits. Telegram/local model responses are unavailable
  during that window. A hard-crash recovery marker is stored under
  `policy-bench/training-priority.json`; run
  `./scripts/resource-profile.sh restore` if manual recovery is ever required.

## Train a roller-skating policy

Qualify Pollen's official passive-wheel environment before a long run:

```bash
make skate-smoke
make verify-skate-artifact
```

Then train the skating baseline:

```bash
make train-skate
```

For symmetric forward/reverse propulsion with both blades grounded, use the
dedicated swizzle workflow:

```bash
make swizzle-smoke
make train-swizzle
make evaluate-swizzle
```

See [docs/SWIZZLE_EVALUATION.md](docs/SWIZZLE_EVALUATION.md) for the checkpoint
qualification battery and [docs/ROLLER_POLICY_SUITE.md](docs/ROLLER_POLICY_SUITE.md)
for active braking, spin, recovery, and supervisor design.

### Compare and promote policies

MicroDuck Policy Bench provides an entirely open-source, offline workflow for
immutable checkpoint snapshots, evaluation history, candidate comparisons,
human-reviewed promotion, and a local HTML dashboard:

```bash
make bench-discover
make bench-list
./scripts/policy-bench.sh evaluate <run-id>
./scripts/policy-bench.sh compare <candidate-run-id> <baseline-run-id>
./scripts/policy-bench.sh metrics <run-id>
make bench-dashboard
```

The dashboard provides one-click **Drive training arena** buttons that launch
each selected checkpoint in its exact Pollen mjlab task, with an isolated
Viser/controller session. Multiple saved policies
can remain open side by side, and the dashboard always shows which arena and
controller belong together. **Deployment check** runs the exported ONNX policy
through Pollen's CPU MuJoCo runtime and adds a scored forward/reverse/coast/
heading evaluation to the run. Its local DuckLab Assistant
can turn a request such as “train swizzle for 8000 iterations with 4096
environments” into a validated configuration and an explicit confirmation
button. It cannot execute arbitrary shell commands and blocks concurrent full
training.

Promoted policies move sequentially through experimental, evaluated,
sim-qualified, hardware-candidate, and production stages. Hardware stages
require explicit sign-off, and the viewer automatically selects the current
sim-qualified swizzle checkpoint after verifying its hash. See
[docs/POLICY_BENCH.md](docs/POLICY_BENCH.md) for the complete workflow.

### Test a policy with an Xbox controller

The remote Viser launcher includes a browser-based Xbox controller bridge. For
a single direct viewer, connect to the Spark from your local computer with both
ports forwarded:

```bash
ssh -L 8080:localhost:8080 -L 8090:localhost:8090 <ssh-user>@<spark-address>
```

In that SSH session, launch the final skating checkpoint:

```bash
./scripts/view-final-skate.sh
```

Open `http://localhost:8080` for Viser and `http://localhost:8090` for the
controller. Connect the Xbox controller to the local computer, press a button
so the browser detects it, and select **Arm Controller**. The left stick or
triggers control propulsion, the right stick controls heading, and X coasts.
Reset, pause/play, and emergency controls are on-screen to prevent accidental
face-button actions. **Resume controls** clears an emergency latch. Commands
automatically fall back to zero if the browser disconnects or stops updating
for 500 ms. A nonzero drive command also resumes a paused simulation.

Use the **Original roller** preset with the first roller checkpoint; its
negative command means braking and it was not trained for reverse. Use the
**Swizzle** preset with a qualified swizzle checkpoint for symmetric forward
and reverse control. This interface controls simulation only and is not a
physical-robot safety system. See [docs/GAMEPAD.md](docs/GAMEPAD.md) for the
complete workflow and troubleshooting notes. Policy Bench can run multiple
models at once; its full SSH forwarding configuration is in
[docs/POLICY_BENCH.md](docs/POLICY_BENCH.md#dashboard-control-center).

The default is 4,096 environments and 5,000 PPO iterations. Override either
setting with `DUCKLAB_ENVS` and `DUCKLAB_ITERATIONS`, as with walking. The
policy learns the upstream roller command semantics: coast at zero, push with
a positive forward command, brake with a negative command, and track heading.
Artifact verification rehearses the policy contract against the roller model.

## Direct upstream commands

```bash
./scripts/ducklab.sh list-envs
./scripts/ducklab.sh scripts/infer_policy.py --walking policy.onnx
```

Use the official `scripts/export.py` for every deployable ONNX file. It embeds
the observation normalizer required by the 50 Hz robot runtime.

Every immutable Policy Bench candidate records the DuckLab commit and the exact
Pollen `microduck_rl` remote, branch, commit, and dirty state used to create it.
The dashboard shows the current upstream revision and the per-run revision so a
future upstream update cannot silently change the meaning of an older result.

See [docs/V1.md](docs/V1.md) for v1.0 scope, qualification gates, and the
arrival-day deployment plan.

## License

Lab automation and documentation are Apache-2.0. The pinned upstream project
retains its own licenses; its 3D model files are CC BY-SA-NC.
