# Roller policy suite

MicroDuck should use specialized, hot-swappable policies with the shared 61D
observation and 14-action deployment contract. This avoids forcing incompatible
objectives into one PPO reward landscape before each behavior works reliably.

## Behaviors

1. **Swizzle locomotion** — symmetric grounded propulsion, forward and reverse.
2. **Active brake** — stop from either direction without interpreting reverse as
   braking.
3. **Spin** — antisymmetric swizzle for controlled rotation in place.
4. **Roller stand-up** — recover from face-down, face-up, and side falls.
5. **Slope/crouch** — specialized terrain and posture behaviors where needed.

## Active-brake command contract

Swizzle currently uses `twist[0]` as signed propulsion: positive is forward,
negative is reverse, and zero is coast. Preserve those semantics. Use the first
otherwise-zero body-command slot (`body_command[0]`) as `brake_intensity` in the
brake policy:

- `0.0`: no braking request;
- `0.0 < value <= 1.0`: active braking strength;
- all other unused body-command slots remain zero.

The runtime supervisor must explicitly zero the propulsion command before
switching to the brake policy. This keeps reverse and brake unambiguous while
retaining the common 61D observation shape.

## Brake training episodes

Train a brake specialist initialized from the mature swizzle actor. Generate
balanced episodes from forward and reverse motion, several starting speeds,
wheel-friction ranges, actuator delays, battery states, and small heading errors.
Include slow and near-zero starts so the policy learns to hold rather than twitch.

Use a staged curriculum:

1. moderate speed, nominal physics, straight heading;
2. forward and reverse speed buckets;
3. stronger/weaker brake requests and randomized friction;
4. actuator/battery/CoM randomization and external disturbances;
5. policy-transition rehearsals from the swizzle actor.

## Brake objective

Reward measured behavior, not a prescribed joint animation:

- potential-based reduction in absolute body-frame speed while braking;
- low terminal speed after a finite deadline;
- upright posture and both blades grounded;
- heading preservation;
- bounded stopping distance;
- low lateral slip, impact, action acceleration, and joint-limit proximity.

Gate progress and terminal rewards on a nontrivial initial speed. Otherwise the
policy can farm rewards by starting or remaining stationary. Avoid a perpetual
per-step reward for zero speed; it creates a jackpot for doing nothing.

## Supervisor state machine

```text
RECOVERY -> HOLD -> SWIZZLE <-> BRAKE -> HOLD
                    |
                    +-> BRAKE -> SPIN -> BRAKE -> HOLD
```

Transitions require hysteresis and measured state:

- enter `BRAKE` when requested or before another maneuver at unsafe speed;
- leave `BRAKE` only after speed stays below a threshold for a dwell period;
- enter `SPIN` only from a stable, near-stationary, upright state;
- enter `RECOVERY` immediately after a confirmed fall;
- return to locomotion only after recovery stability is sustained.

The supervisor owns timeouts, command zeroing, policy handoff, and emergency
fallback. A policy must never infer its operating mode only from stale commands.

## Release sequence

1. Qualify swizzle checkpoints in simulation and visual playback.
2. Implement and smoke-test the explicit brake command.
3. Train and qualify braking from both directions.
4. Qualify the existing spin task and transitions through brake/hold.
5. Qualify roller recovery.
6. Run randomized multi-policy transition batteries.
7. Deploy at conservative current and speed limits with a physical e-stop.
