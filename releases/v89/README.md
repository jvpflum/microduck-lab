# DuckWing V89: learned high-speed braking

V89 is the latest qualified braking-control variant. It embeds the V80 driving
actor and a bounded learned ten-leg correction for high-speed-to-STOP
transitions. Non-STOP and low-speed control preserve V80. It improves braking
rather than claiming a new speed record or hardware readiness.

Independent final seeds 89901–89916 were separated from checkpoint-selection
seeds 89201–89208. Checkpoint 450 was selected from four saved checkpoints.
All final gates passed. V89 had 16/16 healthy high-speed brakes versus V80's
14/16. On the 14 healthy baseline pairs, mean lateral drift improved from
0.598869 to 0.325468 m, worst drift from 1.179300 to 0.787444 m, stopping time
from 2.930000 to 2.371429 s, and stopping path from 1.812486 to 1.493102 m.
Race5 and manual-agility phase metrics were exactly equal to V80 on all final
seeds. Low-speed braking was unchanged. See `metrics.json`,
`evaluation-summary.json`, and the frozen `evaluation-contract.json`.

Official evidence uses ARM64 deterministic CPU MuJoCo, wheel frictionloss
0.003, motor limit 1.75 A, and 50 Hz control. Separate x86 worker replays are
portability checks: trajectories differ between architectures and must not be
pooled with official results. The 5 mph goal remains unmet. No physical-robot
validation or unassisted front-flip success is claimed.

## Input contract

The model is inference-complete; no donor files are needed. Input is float32
shape `[1,15616]`: 256 frames of eight observation terms with widths
`3,3,14,14,14,3,4,6`. Layout is term-major, oldest to newest within each term.
At reset, repeat the first frame across the entire history. Append once per
50 Hz control step and reset history whenever simulation resets. The newest
32 frames feed the learned encoder; the longer command history captures the
high-speed-to-STOP edge and its five-second decaying authority. Leg correction
is bounded to ±0.08 rad; head control is protected.

`tools/evaluation_policy.py` supplies `ObservationHistory` and the metadata-aware
`configure_policy_history(controller)` adapter. Existing 61-input consumers
must adopt this contract before using V89; compatibility with every existing
browser deployment is not claimed.

```python
history = ObservationHistory(256)
input_values = history.observe(frame_61, simulation_time)
action = session.run(None, {session.get_inputs()[0].name: input_values[None]})[0]
```

This release contains the actor only, not optimizer/critic state. Training used
600 iterations × 128 environments × 64 steps = 4,915,200 transitions, including
1,329,075 active braking samples, scheduled as a Ray GPU job. Metadata containing
private training paths is removed from the public artifact; graph parameters
and outputs are preserved. `privacy-normalization.json` records original and
published hashes. The original evaluated artifact remains private and unchanged.

Licensed under the repository's Apache-2.0 license.

```bash
(cd releases/v89 && sha256sum -c SHA256SUMS)
```
