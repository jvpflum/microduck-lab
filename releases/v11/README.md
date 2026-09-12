# Race5 V11 — public simulation baseline

This directory contains the compact, inference-only ONNX export for the current
all-around skating champion:

`ducklab-race5-v11-drag-launch-i10-s42`

It is intentionally **not** a raw PyTorch training checkpoint. No optimizer
state, TensorBoard logs, captured sessions, absolute source paths, SSH details,
or local report bundles are published here.

## Verified simulator result

The policy was evaluated deterministically in CPU MuJoCo against Pollen's
official roller policy, using the same measured line-hold controller for both
policies. This is a simulation result, not a hardware claim.

| Metric | V11 | Pollen baseline |
| --- | ---: | ---: |
| Sustained forward speed | 1.42 mph | 1.07 mph |
| Verified top speed (0.5 s) | 1.65 mph | 1.28 mph |
| 100-ft elapsed time | 44.06 s | 57.59 s |
| 100-ft trap speed | 1.60 mph | 1.22 mph |
| First-second acceleration | 1.12 mph/s | 0.72 mph/s |
| Maximum lateral drift | 1.06 ft | 1.25 ft |
| Maximum heading error | 7.30° | 11.06° |

All 14 retained stability, braking, long-run, and turning checks passed in the
recorded simulation evaluation. It is the public all-around baseline while
faster, less complete experimental policies remain unpromoted.

## Files

- `microduck-race5-v11.onnx` — inference-only policy export.
- `SHA256SUMS` — integrity hash for the export.
- `evaluation-summary.json` — portable, scrubbed measurement summary.

Verify after download:

```bash
sha256sum -c SHA256SUMS
```

Use the repository's Policy Bench dashboard to open the matching simulator
preview. The exact simulation configuration, controller behavior, and scoring
code are in source; private local run directories are deliberately excluded.
