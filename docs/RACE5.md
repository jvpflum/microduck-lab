# Race5 skate-racing reproduction

Race5 is DuckLab's flat, 100-foot skate race. It starts every duck on the
centreline, rewards measured world-forward progress, and penalizes cross-track
velocity and heading departure. The goal is a faster A-to-B result without
silently replacing it with a different task.

## Fresh clone gate

After `make bootstrap` and `make preflight`, run:

```bash
make race5-smoke
```

This is deliberately a from-scratch 128-environment, 30-iteration integration
check. It proves that the public task registration and Race5 dependency closure
work. It is not a useful policy and does not require a checkpoint.

## Continue V11 privately

The published [V11 release](../releases/v11/README.md) is an ONNX inference
artifact. Raw PPO `.pt` donor state stays out of the public repository because
it includes optimizer and local-run metadata. To reproduce a V11 continuation,
copy that donor through a private channel, then run:

```bash
DUCKLAB_RACE5_WARMSTART_CHECKPOINT=/absolute/path/to/v11-model_10.pt \
DUCKLAB_ENVS=4096 DUCKLAB_ITERATIONS=4000 \
make train-race5
```

Use a unique `DUCKLAB_RACE5_RUN_NAME` on each worker. The training recipe saves
frequent raw checkpoints locally; Git intentionally ignores checkpoints, logs,
reports, W&B data, and Policy Bench results.

## Evaluate a candidate

Export the normalizer-aware ONNX artifact through the normal DuckLab training
finalizer, then run the deterministic CPU MuJoCo race evaluation:

```bash
make evaluate-race5 POLICY=/absolute/path/to/policy.onnx
```

The result reports mph and m/s speed, acceleration, elapsed time, lateral drift,
heading, survival, and retained control checks. Compare candidates against the
published V11 summary rather than PPO reward alone.

## Public components

The `upstream/microduck_rl` submodule points to DuckLab's public fork. It
contains the Race5, race, sprint, and speed-discovery task configurations plus
their tests. The `upstream/microduck-simulator` submodule contains the browser
arena's matched speed-test instrumentation. Both are pinned by this repository
and come down with `git clone --recurse-submodules`.
