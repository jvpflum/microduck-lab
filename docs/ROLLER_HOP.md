# Roller Hop v1

Roller Hop is a specialist policy, not a replacement for Pollen's factory
roller controller. The supervisor sequence is:

```text
FACTORY ROLLER -> brake and stabilize -> ROLLER HOP -> confirm landing -> FACTORY ROLLER
```

## V1 target

- stationary start on both skates;
- preload without falling or drifting;
- both skates leave the floor;
- at least 15 mm qualification clearance, with 20 mm as the training target;
- land on both skates, remain within 15 degrees of upright, settle below
  0.10 m/s, and drift less than 50 mm.

The initial target is intentionally modest. A forward-moving hop, higher
clearance, and obstacle traversal are later curricula after the stationary
landing is repeatable.

## Anti-cheat reward structure

The roller model starts slightly above the plane to avoid geometry
interpenetration. A support latch therefore requires both skates to touch the
floor before takeoff can count; the initial settling fall is not a jump.

Clearance is paid once as potential progress and only while both skates are
airborne. Landing and stillness rewards remain locked until genuine takeoff.
Standing still at reset cannot farm the landing annuity. Low-weight drift,
angular-velocity, action-rate, torque, collision, and vertical-impact terms
shape the maneuver without blocking the high-force extension needed for
discovery. Smoothness and impact pressure increase later in the curriculum.

## Commands

```bash
make hop-smoke   # required 64 environments x 5 PPO iterations
make train-hop   # 4096 environments x 1500 iterations by default
```

Override a guarded full run with `DUCKLAB_ENVS` and `DUCKLAB_ITERATIONS`.
Finished runs are discovered under `logs/rsl_rl/roller_hop`, verified through
the official normalizer-aware ONNX export, evaluated headlessly, and shown in
Policy Bench. Human review is required before promotion or hardware testing.
