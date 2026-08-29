# Swizzle evaluation

The swizzle policy is qualified in the CPU MuJoCo deployment-rehearsal model,
not from PPO reward alone. Run:

```bash
make evaluate-swizzle
```

Pass an explicit ONNX path to compare checkpoints:

```bash
./scripts/evaluate-swizzle.sh /absolute/path/to/policy.onnx
```

The deterministic battery runs settle, forward, coast, reverse, and left/right
heading phases. It reports:

- body-frame forward and lateral speed;
- torso-height variation and body tilt;
- fraction of time both blades contact the floor;
- skate-separation amplitude and estimated swizzle cycles;
- passive-wheel angular speed; and
- action acceleration as a smoothness indicator.

## Qualification questions

Compare checkpoints near iterations 3000, 5000, and 8000. A candidate must:

1. move in the commanded direction both forward and backward;
2. keep both blades grounded for most of each powered phase;
3. show repeated expansion and contraction of skate separation;
4. avoid large vertical torso excursions and excessive tilt;
5. rotate its wheels rather than translating mainly through lateral slip;
6. remain smooth and stable through command transitions; and
7. pass visual inspection in Viser—the metrics cannot detect every reward hack.

Do not set universal numeric pass thresholds until the first mature swizzle run
establishes physically plausible baselines. Record the thresholds after visual
review so later policies can be gated automatically without moving the goalposts.
