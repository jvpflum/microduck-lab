# MicroDuck roller speed discovery

`Mjlab-SpeedDiscovery-Flat-MicroDuck-Rollers` is a simulation-only experiment
for discovering the fastest passive-wheel locomotion policy. It does not modify
the normal roller, Sprint, Race5, or Pollen baseline tasks.

## Why this task exists

The previous Race5 continuation optimized 30 simultaneous terms. Its dominant
control constraints included action-rate `-1.2`, heading hold `+16`, lane error
`-22`, world-lateral speed `-13`, heading error `-22`, pose and CoM targets,
torque cost, gait symmetry, lean, glide, and skating-form rewards. Its PPO trust
region was also deliberately tiny. That configuration is appropriate for
qualifying an existing gait, but it suppresses aggressive gait discovery.

The public `HannesVonEssen/microduck-running` manifest demonstrates a useful
discovery pattern: flat terrain, actual forward progress at weight `5.0`, weak
action-rate cost `-0.1`, no symmetry or heading feedback, relaxed pose/motion
regularization, then robustness only after a fast gait exists. This task adapts
only that optimization pattern. Physics, contacts, actions, warm start, and the
resulting policy remain the MicroDuck passive-roller skating setup.

## Discovery configuration

- actual body-forward chassis velocity: weight `+5.0`, matching Hannes's
  published discovery/evaluation metric;
- signed velocity squared: weight `+0.75`;
- reward keeps increasing beyond the current command up to a simulator impulse
  guard at `7.5 m/s` (the goal is `6.7 m/s`);
- explicit terminal fall charge: `-500`;
- only weak anti-exploit costs remain: action rate `-0.1`, self-collision
  `-0.1`, and command-past-joint-limit `-0.05`;
- no pose, posture, gait-form, wheel-spin, glide, lean, energy, torque, slip,
  lane, lateral, yaw, or steering rewards;
- flat plane, fixed launch pose, no pushes, no CoM/mass/friction/armature/IMU/
  encoder randomization, no observation noise, and no observation delay;
- actor observation remains the deployable 61D contract and output remains 14
  joint-position targets at 50 Hz.
- PPO continuation uses a conservative `3e-5` learning rate, `0.10` clip,
  `0.001` entropy coefficient, and `0.15` initial action standard deviation.
  Spark pilots found that `0.35` exploration and high entropy erased the
  deterministic skate gait within a few updates.

The curriculum stages are `2.5`, `3.5`, `4.5`, `5.5`, and `6.7 m/s`. The actor
continues to receive the known deployable `0.8` full-effort token; literal
multi-m/s commands made the V11 roller actor crouch/stop because they were far
outside its training distribution. The curriculum instead raises the measured
speed milestone in a dense progress term, while raw velocity rewards keep
increasing beyond it. A stage
advances only after two independent windows meet both sustained-speed and
survival thresholds. Iteration count never advances it.

## PPO scale

The rollout is 24 steps per environment.

| Profile | Environments | Batch/update | Minibatches | Samples/minibatch |
|---|---:|---:|---:|---:|
| Recommended first run | 4,096 | 98,304 | 4 | 24,576 |
| 5090 scale comparison | 8,192 | 196,608 | 8 | 24,576 |

Keeping samples per minibatch constant makes the wall-clock comparison useful;
8,192 environments do not silently double each optimizer minibatch.

## Launch

The default warm start is the V11 skating champion. On another machine, point
the variable at the copied `model_10.pt` checkpoint.

```bash
DUCKLAB_SPEED_WARMSTART_CHECKPOINT=/absolute/path/to/model_10.pt \
DUCKLAB_ENVS=4096 \
DUCKLAB_ITERATIONS=4000 \
./scripts/train-speed-discovery.sh
```

The 8,192-environment comparison is:

```bash
DUCKLAB_SPEED_WARMSTART_CHECKPOINT=/absolute/path/to/model_10.pt \
DUCKLAB_ENVS=8192 \
DUCKLAB_ITERATIONS=4000 \
./scripts/train-speed-discovery.sh
```

Extend a profile to 6,000 iterations by changing `DUCKLAB_ITERATIONS=6000`.

## Outputs

- Checkpoints every 25 PPO iterations.
- Trainer log: simulation steps/s, collection time, learning time, episode
  length, speed metrics, curriculum state, falls, and all reward components.
- GPU JSONL: utilization, VRAM, power, and temperature from `nvidia-smi`.
- Run summary: policy collection FPS, optimizer throughput, GPU statistics,
  and best checkpoint metrics for 4,096-vs-8,192 comparison.
- `best_speed_discovery.pt` and `.onnx`: selected lexicographically by
  horizon-normalized sustained forward speed, survival, then peak speed.
- At least five deterministic episodes are required per saved checkpoint.
- Per-checkpoint reports include m/s and mph, average and maximum velocity,
  best one-second speed, distance, time alive, falls, and estimated gait
  frequency. Net world-X displacement is retained as a diagnostic; straight
  A-to-B performance remains a separate Race5 qualification problem.

Evaluate any exported policy directly:

```bash
.tools/uv/bin/uv run python tools/evaluate_speed_discovery.py POLICY.onnx \
  --episodes 5 --duration 20 --command-mps 0.8 --output result.json
```

## Later robustness stage

Do not deploy a discovery checkpoint. The later robustness task should derive
from this module and warm-start from `best_speed_discovery.pt`, then progressively
restore actuator/observation delays, foot or wheel friction variation, encoder
and IMU error, CoM/mass variation, pushes, action/torque limits, steering, and
backlash. Each addition should be performance-gated and evaluated against the
frozen nominal speed checkpoint so robustness cannot silently erase the gait.
