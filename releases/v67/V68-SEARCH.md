# V68 local-evolution search

V68 tested whether finer action fusion could improve V67 without sacrificing a
single promoted measurement. The search kept V67's command routing and V65
moving-brake route fixed, then independently tuned symmetric hip-yaw, hip-roll,
hip-pitch, knee, and ankle authority from the V47 speed specialist. Follow-up
searches optimized line-hold gains and swept the boundary between speed- and
tracking-favored controllers.

All 586 scored configurations used wheel frictionloss `0.003`, current limit
`1.75 A`, and deterministic CPU MuJoCo. The promotion gate required a faster
100-foot result and higher sustained speed, with no regression in top speed,
acceleration, lateral drift, heading error, tilt, grounded contact, or stopping.
No candidate passed that gate, so V67 remains the definitive leader.

## Best research challenger

The strongest interpretable near-miss used V47 authorities
`0.247352/0.347641/1.050326/1.039918/1.181325` for hip yaw, hip roll, hip
pitch, knee, and ankle respectively, with V67's `0.70/0.22/0.07/0.15`
line-hold controller.

| Metric | V67 leader | V68 challenger | Result |
| --- | ---: | ---: | --- |
| 100-ft time | 25.815 s | 25.659 s | 0.61% sooner |
| Long-run sustained speed | 2.727 mph | 2.745 mph | 0.65% faster |
| Verified top speed | 3.060 mph | 3.138 mph | 2.54% higher |
| First-second acceleration | 0.4714 m/s² | 0.5255 m/s² | 11.5% higher |
| Maximum lateral drift | 0.775 ft | 0.670 ft | 13.6% lower |
| Maximum heading error | **10.22°** | **13.26°** | **29.7% worse** |
| Maximum tilt | 17.283° | 17.272° | 0.06% lower |
| Grounded fraction | 0.6519 | 0.6853 | 5.1% higher |
| Low-speed stop | 0.88 s | 0.88 s | equal |

Controller recovery reduced heading error, but every recovered configuration
gave back at least one of acceleration, tilt, contact, top speed, or sustained
speed. Fine interpolation showed discontinuous gait changes rather than a
smooth overlap region.

## Next run

Do not spend another long run on global PPO or static post-hoc action mixing.
Keep V67's idle/cruise/turn and brake routes immutable, train only a bounded
phase-aware propulsion residual, and include world-frame heading, tilt, and
bilateral blade contact directly in selection. Seed the residual from the V68
challenger direction, use a small trust region, retain multiple contact-diverse
elites, and evaluate checkpoints against `leader-metrics.json` at exact physics
before any promotion.
