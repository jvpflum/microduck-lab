# V69 body-yaw state-guard search

V69 tested whether the best V68 speed challenger could be used only during
calm body-yaw states, with a continuous fallback to V67 as measured rotation
increased. This was a lossless ONNX composition experiment: it did not
distill, retrain, or overwrite either source actor.

The search first measured V67's `abs(base_ang_vel_z)` distribution, then swept
four full-authority thresholds (`0.00`, `0.15`, `0.30`, `0.50 rad/s`), four
fallback thresholds (`0.40`, `0.70`, `1.10`, `1.60 rad/s`), and ten challenger
authorities (`0.01` through `1.00`). Invalid threshold pairs were excluded,
leaving **150** deterministic candidates. Every replay used wheel
`frictionloss=0.003`, current limit `1.75 A`, and V67's
`0.70/0.22/0.07/0.15` line controller.

No candidate passed the no-regression promotion rule, so **V67 remains the
definitive leader**.

## Fastest V69 challenger

The fastest candidate used a `0.50→1.10 rad/s` fallback band and `0.70`
challenger authority.

| Metric | V67 leader | Fastest V69 | Result |
| --- | ---: | ---: | --- |
| 100-ft time | 25.815 s | 25.571 s | 0.95% sooner |
| Long-run sustained speed | 2.727 mph | 2.755 mph | 1.03% faster |
| Verified top speed | 3.060 mph | 3.110 mph | 1.63% higher |
| First-second acceleration | 0.4714 m/s² | 0.4950 m/s² | 5.02% higher |
| Maximum lateral drift | 0.775 ft | 0.768 ft | 0.96% lower |
| Maximum heading error | **10.22°** | **12.81°** | **20.2% worse** |
| Maximum tilt | **17.283°** | **17.525°** | **1.38% worse** |
| Grounded fraction | **0.6519** | **0.6469** | **0.77% worse** |
| Low-speed stop | 0.88 s | 0.88 s | equal |

Only two very-low-authority candidates improved heading, and both lost speed,
acceleration, tilt, and at least one other retained measurement. The body-yaw
guard therefore moved the trade-off but did not remove it. The result supports
V68's conclusion: the next meaningful speed gain needs a phase/contact-aware
learned residual whose objective includes heading and stability, not another
static or single-state ONNX blend.

Reproduce the complete public search with `make v69-search`. Generated policies
and raw evaluations remain under ignored `reports/`; the builder, sweep grid,
source policies, leader metrics, and this result are committed.
