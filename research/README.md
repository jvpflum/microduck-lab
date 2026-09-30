# Research ledger

This section records work that has not earned a new general-purpose model.
V89 remains the recommended skating/braking model; standalone 180, 360 and 720
degree turns retain their separate published evidence.

| Work | Current evidence | Disposition |
|---|---|---|
| [V90 Laguna bundle](../releases/v90-laguna/) | 42/48 clean laps at each tested heading; combined transitions 10/48 | Racing research bundle, not a general-purpose replacement |
| Later racing continuation | Starting control 60/64; trained continuation 44/64 | Rejected; its weights are excluded |
| [Extended transition recovery](transition-recovery-r1/) | Recovery checks improved from 10/96 to 76/96; race-tail speed/heading still failed | Diagnostic only; synthetic route observations did not prove track rejoining |
| [Isaac Lab/Newton bridge](isaaclab-v89-parity/) | Repeatable, upright, but CPU-reference path/action gates failed | Migration paused; original MuJoCo-Warp resumed |
| [Preview-controller campaign](mujoco-warp-push-r1/) | Real GPU learning smoke passed; two fixed training seeds submitted 2026-09-29 | Training/evaluation pending; no new best-model claim |

Parallel worlds are not independent training seeds. GPU course results and
authoritative CPU skating/trick results use distinct evaluation contracts and
are not pooled. Preserved failed results are not retroactively passed by the
new preview-controller contract. All evidence here is simulation-only.

Ray schedules tested software and GPU capabilities; MinIO remains optional
private artifact storage. Public Git carries reviewed source, inference-only
models and scrubbed summaries. Operator credentials, machine addresses,
private run receipts, raw optimizer state and full logs remain private.
