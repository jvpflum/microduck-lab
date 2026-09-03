# V70 teacher-guided residual search

V70 tested the next strategy suggested by V68/V69: train a deployable
observation-space residual in MuJoCo instead of searching static ONNX blends.
The actor started from the exact V65 high branch, kept its deployment
normalizer frozen, and was trained with a frozen control teacher plus the V47
official-friction speed teacher. Mild air-time, single-support, glide,
bilateral-symmetry, and heading-hold terms supplied contact-aware gradients.

Every rollout used the official wheel `frictionloss=0.003`, the `1.75 A` BAM
current ceiling, and the existing 61D actor / 78D critic interface. The run
completed 800 iterations at 2,048 environments with no NaN failures. The actor
exported normally, so this was a valid deployment-shaped experiment—not a
phase feature that disappears when the runtime pads the input.

## Result

V70 did not displace V67. The learned branch plateaued near 0.61 m/s in its
training averages. Two representative V70 checkpoints were composed behind
V67 with conservative joint authority and evaluated with the same deterministic
CPU MuJoCo harness and line controller.

| Metric | V67 leader | V70 i0 / low authority | V70 i600 / mid authority |
| --- | ---: | ---: | ---: |
| 100-ft time | 25.815 s | 25.982 s | 26.170 s |
| Long-run sustained speed | 2.727 mph | 2.706 mph | 2.686 mph |
| Verified top speed | 3.060 mph | 3.056 mph | 3.048 mph |
| Maximum lateral drift | 0.775 ft | **0.661 ft** | 0.766 ft |
| Maximum heading error | 10.22° | **9.85°** | 11.18° |
| Maximum tilt | 17.283° | 17.715° | 17.612° |
| Grounded fraction | 0.6519 | 0.6177 | 0.6272 |
| Low-speed stop | 0.88 s | 0.88 s | 0.88 s |

The low-authority checkpoint improved drift and heading but lost race time,
top speed, tilt, and contact. The mid-authority checkpoint also lost speed and
heading. Neither meets the strict promotion rule; **V67 remains the definitive
leader**.

## What this changes

The experiment rules out the current V70 recipe as a speed path: contact
shaping plus a strong V65/V47 action anchor mostly consolidates the existing
gait. The next high-value run should import the actual V67 high-route actor
(rather than the older V65 branch), optimize acceleration/top speed with a
much smaller teacher floor, and select checkpoints by the complete Race5
scorecard. V67's idle, turn, and moving-brake routes should remain immutable.

Reproduce the training recipe with `make train-speed-v70-residual`. Generated
checkpoints, exports, compositions, and raw evaluations are kept under the
ignored `reports/v70-residual-eval/` directory.
