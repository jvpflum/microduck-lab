# MicroDuck Lab

Reproducible reinforcement-learning workspace for training Pollen Robotics'
MicroDuck on an NVIDIA DGX Spark before the physical robot arrives.

The lab pins Pollen's official `microduck_rl` project as a Git submodule and
adds guarded setup, validation, smoke-training, and baseline-training commands.
It does not fork or alter the robot's observation contract, actuator model,
domain randomization, or ONNX export path.

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
Weights & Biases defaults to offline mode so a cloud account is not required.
Training automatically exports the final checkpoint through Pollen's official
normalizer-aware ONNX path. `make verify-artifact` validates its 61-to-14
contract, metadata, finite CPU inference, and a 100-step CPU MuJoCo rehearsal.

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

See [docs/V1.md](docs/V1.md) for v1.0 scope, qualification gates, and the
arrival-day deployment plan.

## License

Lab automation and documentation are Apache-2.0. The pinned upstream project
retains its own licenses; its 3D model files are CC BY-SA-NC.
