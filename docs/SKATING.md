# Roller-skating track

MicroDuck Lab uses Pollen Robotics' official
`Mjlab-Velocity-Flat-MicroDuck-Rollers` task and passive-wheel MuJoCo model.
The lab does not change its observation layout, actuator model, rewards, domain
randomization, or export path.

## Progression

1. Run `make skate-smoke` at 64 environments for five PPO iterations.
2. Run `make verify-skate-artifact` to check the 61D-to-14D ONNX contract and
   rehearse 100 CPU MuJoCo steps with the roller model.
3. Stop or reduce other unified-memory GPU workloads before `make train-skate`.
4. Train the baseline, retaining checkpoints and offline W&B metrics.
5. Evaluate balance, push/glide, heading response, braking, falls, impacts,
   joint-limit behavior, and action smoothness before hardware deployment.

The five-iteration smoke policy proves integration only; it is not trained
enough to skate and must never be deployed to the physical robot.
