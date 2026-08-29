# Upstream-first architecture audit

Date: 2026-08-29

DuckLab defaults to Pollen's working product surface. We maintain code only
where it adds measurable value around Pollen's stack.

## Adopt, retain, retire

| Product need | Decision | Source of truth |
| --- | --- | --- |
| Factory walking, skating, sit/stand, recovery, kicks, ground pick, crouch | Adopt | `upstream/microduck/policies/` |
| Browser physics arena and keyboard/touch/Xbox input | Adopt | `upstream/microduck-simulator/` |
| Training environments, PPO configuration, export contract | Adopt | `upstream/microduck_rl/` |
| Hardware control, motor ownership, safety, deadman, skills, update/rollback | Adopt | `upstream/microduck/` |
| Run registry, immutable hashes, curves, scoring, comparisons, review and promotion | Retain | DuckLab Policy Bench |
| Resource-aware local training and local assistant | Retain | DuckLab orchestration |
| Custom Viser/gamepad arena | Advanced fallback | Exact-checkpoint diagnosis only |
| New locomotion tasks that duplicate shipped behavior | Retire | Train only against a named unmet gate |

## Pinned evidence

- Pollen runtime: `590b986bd8c0d50ae02cb3ea2f59c463b6828168`
- Pollen browser simulator: `1261013e7e28ba2a6878bd76ae573751c0e4b457`
- DuckLab training fork: `7b07bd8` (Pollen `develop` plus the Roller Hop task;
  upstream base `d424a0c899f6b33cbd3daeb279913134349c0b63`)
- Factory roller SHA-256:
  `cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd`

The policy bytes in the pinned runtime and browser simulator are identical.
In DuckLab's 50 Hz CPU MuJoCo deployment battery, the factory roller produced:

- steady forward speed: 0.364 m/s;
- steady reverse speed: -0.333 m/s;
- forward/reverse stop times: 0.98 s / 1.18 s;
- left/right yaw response: +2.288 / -2.370 rad/s;
- maximum tilt: 12.65 degrees;
- 11/12 detected skating cycles;
- score: 80.9/100, all qualification gates passed.

The latest custom swizzle candidate (iteration 11,749) scored 42.72/100 and failed the basic
motion gates. The factory roller is therefore the `sim-qualified` champion.

The pinned training checkout also already registers ball kick, ground pick,
roller crouch, roller slope, roller stand-up, roulade, sit/stand, spin,
stand-up, walking/rolling velocity, rough-terrain, and backlash variants. A
custom DuckLab environment must not be added until this catalog has been
checked for the requested behavior.

## Development gate

Every capability request follows this sequence:

1. Search the pinned factory policy catalog.
2. Test the shipped behavior in Pollen's official browser arena.
3. Measure it with a repeatable evaluation.
4. If it fails, define one observable target and the corresponding test.
5. Only then add or tune an upstream-compatible RL task.
6. Verify checkpoint-to-ONNX parity and compare against the factory baseline.
7. Require human review before promotion. Hardware stages additionally require
   physical safety sign-off.

Policy/runtime/training revisions are pinned together. A 61-to-14 tensor shape
is necessary but not sufficient for compatibility: observation construction,
normalization, action scaling, filtering, model assets, and runtime tuning must
all match the policy's provenance.
