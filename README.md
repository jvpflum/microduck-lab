# DuckLab — MicroDuck Skate Racing

An open, reproducible reinforcement-learning lab for making Pollen Robotics'
MicroDuck skate faster, straighter, and more controllably in simulation.

DuckLab uses Pollen's official robot runtime, browser arena, physics, and
roller baseline as the reference point. It adds the practical layer around
training: repeatable race measurements, saved ONNX policies, a simple dashboard
to test them, and promotion only when a candidate actually improves.

## Why use DuckLab?

Because “it looked fast” is not a benchmark. DuckLab lets you train a skating
policy, open it in the browser simulator, and see whether it truly beat Pollen
on speed **without giving up straight-line control, braking, turning, or
stability**.

The current public all-around champion is
[`Race5 V11`](releases/v11/README.md). In the same deterministic CPU MuJoCo
race battery, with the same measured line-hold controller applied to both
policies, V11 beats Pollen's roller baseline:

| Verified race metric | Race5 V11 | Pollen roller | Improvement |
| --- | ---: | ---: | ---: |
| Sustained forward speed | 1.42 mph | 1.07 mph | **33.1% faster** |
| Verified top speed (0.5 s) | 1.65 mph | 1.28 mph | **28.5% higher** |
| 100-ft elapsed time | 44.06 s | 57.59 s | **23.5% sooner** |
| First-second acceleration | 1.12 mph/s | 0.72 mph/s | **55.0% higher** |
| Maximum lateral drift | 1.06 ft | 1.25 ft | **14.9% less** |
| Maximum heading error | 7.30° | 11.06° | **34.0% less** |

V11 passed all 14 retained control checks in that evaluation. It is a
simulation-only result—not a hardware-speed claim—and the 5 mph target remains
the next milestone. Faster experimental policies stay unpromoted until they
also preserve the full control battery.

## What you get

- Pollen's colorful browser arena for trying factory and custom roller policies.
- A race scoreboard that shows the current champion, Pollen comparison, speed,
  drift, heading, acceleration, and 100-ft result.
- Repeatable evaluation and ONNX checks so results are shareable and auditable.
- Separate Spark and Windows/RTX worker workflows for larger training runs.
- A public V11 inference export with checksum and scrubbed measurement summary.

## Requirements

- Linux ARM64 DGX Spark / GB10 **or** Linux x86_64 with a working NVIDIA GPU
  (Windows workers use Ubuntu through WSL2)
- Python 3.12 and `venv`
- Git
- At least 20 GiB available unified memory for GPU commands
- Network access for the first dependency synchronization

For a Windows RTX worker, follow [Windows 5090 worker setup](docs/WINDOWS_5090.md).
The Spark and Windows machine should use separate clones and unique run names;
Git is the code handoff, while raw training checkpoints stay private.

## Start

```bash
git clone --recurse-submodules <repository-url>
cd microduck-lab
make bootstrap
make preflight
make test
make import-pollen-baselines
make bench-dashboard
```

Forward port 8091 over SSH, open `http://localhost:8091`, then click
**Open factory playground**. In the arena, hold D-pad up for about one second
to switch between walking and rollers; the left stick drives and turns.

`make smoke` follows Pollen's required 64-environment, five-iteration gate.
Weights & Biases is disabled; Policy Bench remains the local system of record.
Training automatically exports the final checkpoint through Pollen's official
normalizer-aware ONNX path. `make verify-artifact` validates its 61-to-14
contract, metadata, finite CPU inference, and a 100-step CPU MuJoCo rehearsal.
Successful skating runs are then automatically verified, registered, scored in
the CPU deployment battery, and given training-curve metrics. They are never
automatically starred or promoted.

## Train a custom walking baseline

Custom training is an advanced path. Test Pollen's factory policies first and
write down the specific capability or evaluation gate that needs improvement.

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

The dashboard offers two resource modes for new runs:

- **Shared** is the portable default. DuckLab does not start, stop, or assume
  any other service on the host.
- **Training priority** is for a multi-use GPU machine. It runs only the
  optional stop/restore hooks supplied by that machine's operator, then restores
  them when training exits. It is safe to select with no hooks configured; no
  service is changed. See `scripts/resource-profile.sh` for the three optional
  `DUCKLAB_RESOURCE_*_CMD` variables.

## Train a custom roller-skating policy

Qualify Pollen's official passive-wheel environment before a long run:

```bash
make skate-smoke
make verify-skate-artifact
```

Then train the skating baseline:

```bash
make train-skate
```

## Race5: reproducible straight-line skate racing

Race5 is the separate 100-foot, centreline-controlled skate-racing task behind
the public V11 result. Its public integration gate needs no private artifact:

```bash
make race5-smoke
```

For an exact continuation from the V11 champion, copy the raw trainer donor
privately and point the recipe at it; raw `.pt` state is deliberately not
published. A clean clone can still use every source configuration, test, smoke
gate, simulator speed test, and the public V11 ONNX evaluation export.

```bash
DUCKLAB_RACE5_WARMSTART_CHECKPOINT=/absolute/path/to/v11-model_10.pt \
DUCKLAB_ENVS=4096 DUCKLAB_ITERATIONS=4000 \
make train-race5
```

See [docs/RACE5.md](docs/RACE5.md) for the task map, evaluation procedure, and
what is intentionally excluded from Git.

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

For simulation-only maximum-speed discovery, use the separate permissive task:

```bash
DUCKLAB_SPEED_WARMSTART_CHECKPOINT=/absolute/path/to/model_10.pt \
DUCKLAB_ENVS=4096 DUCKLAB_ITERATIONS=4000 \
./scripts/train-speed-discovery.sh
```

It does not change the normal skating or Race5 recipes. See
[docs/SPEED_DISCOVERY.md](docs/SPEED_DISCOVERY.md) for the performance-gated
2.5→6.7 m/s curriculum, 4,096/8,192 profiles, GPU telemetry, checkpoint
selection, and evaluation metrics.

## Compare and promote policies

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

The dashboard's **Open simulator** action opens Pollen's colorful browser arena
with native gamepad support for both factory and custom models. For a custom
saved model, Policy Bench verifies its immutable ONNX snapshot, loads that
exact artifact into the correct policy slot, and selects feet or rollers
automatically. The older white Viser surface is retained only as an advanced
engineering debugger and is never the dashboard's normal Play path.
While a run is active, its separate **Watch training live** action opens a
six-robot gray-tile mjlab view of the newest immutable checkpoint. The trainer
still runs thousands of environments headlessly; this lightweight sample is
refreshed by reopening the button after a newer checkpoint is saved and never
replaces the finished-model arena.

The main colorful arena continuously buffers six seconds of MuJoCo state and
policy actions. After any useful manual maneuver, immediately click
**Replay → Save clip**. The universal button uploads the previous six seconds
to `reports/demonstrations/`; failed attempts can simply be left unsaved. The
observation-only live-training viewer intentionally has no recorder.
**Evaluate** runs the exported ONNX policy
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

### Drive the factory robot with an Xbox controller

On macOS, install the Policy Bench companion once to manage every dashboard,
Viser, and controller forward and automatically open viewers launched by Codex.
See [the Policy Bench Mac setup](docs/POLICY_BENCH.md#dashboard-control-center).

Connect from your laptop with the dashboard and factory arena forwarded:

```bash
ssh -L 8091:localhost:8091 <ssh-user>@<spark-address>
```

Open `http://localhost:8091`, click **Open simulator** on a factory or saved
model, press a button on the controller so the browser detects it, and drive
with the left stick. This is Pollen's controller implementation and is the
default test path. No Viser or controller port forwarding is required. Custom
policy previews start with a clean floor; use the in-arena **Ball on / Ball
off** control when a ball is useful. That choice persists when the robot is
reset.

### Debug an exact custom checkpoint

The Viser launcher remains an advanced compatibility tool. For a single direct
viewer, forward both of its ports:

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
