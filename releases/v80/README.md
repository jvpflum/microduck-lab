# DuckWing V80 high-command residual

V80 is a simulation-qualified successor candidate to the frozen V67 release.
It keeps V67 exactly unchanged for commands at or below `0.5 m/s`, then adds a
small state-dependent residual to the six propulsion joints for high-speed
commands:

- joint-position gain: `-0.075`
- joint-velocity gain: `-0.010`
- previous-action gain: `-0.070`
- route: `command_x > 0.5`

The route was verified numerically: low-command actions match V67 with maximum
absolute error `0.0`; high-command actions match the selected residual policy
with maximum absolute error `0.0`.

## Official-friction result

All measurements use CPU MuJoCo, `1.75 A`, wheel friction loss `0.003`, and the
same measured line-hold controller as V67 (`yaw_kp=0.70`, `lateral_kp=0.22`,
`yaw_kd=0.07`, `max_wz=0.15`).

| Metric | V67 | V80 |
|---|---:|---:|
| 100-ft elapsed time | 25.815 s | **25.729 s** |
| Long-run mean world speed | 2.643 mph | **2.652 mph** |
| Peak world speed | 3.272 mph | **3.313 mph** |
| Verified 0.5 s top speed | 3.060 mph | **3.084 mph** |
| Long-run heading error | 10.22 deg | **9.54 deg** |
| Long-run drift | 0.775 ft | **0.403 ft** |
| Long-run maximum tilt | 17.28 deg | **15.77 deg** |
| Randomized-start 5x20 sustained | 2.438 mph | **2.497 mph** |
| Clean 20 s screen | 2.570 mph | **2.574 mph** |

V80 passes all 15 Race5 qualification gates, satisfies the repository's formal
`race5_advances_incumbent` rule against V67, and passes all six direct Pollen
head-to-head checks. Hardware validation is still required.

## Reproduce

Build the policy from the immutable V67 artifact:

```bash
python tools/build_command_gated_dynamic_residual_policy.py \
  releases/v67/duckwing-v67-joint-specialist-fusion.onnx \
  releases/v80/duckwing-v80-high-command-residual.onnx \
  --pos-gain -0.075 --vel-gain -0.010 --last-gain -0.070 \
  --command-threshold 0.5
```

The complete deterministic Race5 output is in `evaluation-summary.json`.
Randomized-start evidence is in `noisy-start-evaluation.json`, and the isolated
clean-start result is in `clean-screen-evaluation.json`.
