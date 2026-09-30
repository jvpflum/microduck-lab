# Paused Isaac Lab / Newton migration pilot

The project returned to its original MuJoCo-Warp training stack on 2026-09-29.
The failed migration pilot is retained as research evidence. It tested Isaac
Lab's kitless Newton/MuJoCo-Warp backend; it did not qualify full Isaac Sim,
RTX rendering or PhysX. Framework popularity was not treated as a result.

Pinned pilot: Isaac Lab develop commit
`eaab09fd0a61ea1fe84067d25172797abc30b89a`, Isaac Lab 26.0.0, Newton 1.6.0,
MuJoCo/MuJoCo-Warp 3.12.0 and PyTorch 2.12.0+cu130.

The bridge repaired standing-pose/reset alignment, xyzw/wxyz quaternion
interpretation, Euler integration, legacy IMU timing, contact material,
self-collision and mask handling. Passive damping was already imported;
adding it to drive damping duplicated it and was rejected. Active GPU
tolerance was corrected after upload silently floored the source value.
Six adapter tests passed.

Two deterministic repeats matched at recorded metrics and snapshots, remained
finite and upright, and started with exactly matching observations/actions.
They still failed the unchanged CPU-reference gates:

| Measure | Observed | Required |
|---|---:|---:|
| Final CPU/Isaac XY separation | 0.566202 m | <=0.25 m |
| Mean action difference | 0.181062 | <=0.15 |
| Original MJCF direct-GPU/CPU separation | 0.479757 m | diagnostic control |

The direct-GPU control bypassed the importer yet also diverged, so importer-only
causation is unproved. This does not exclude remaining import differences.
Earlier r3-r6 Isaac fall classifications are invalidated because their test
passed xyzw quaternions to a wxyz calculation. No failing run was relabeled
as a successful transfer and no model was promoted.

`evaluation-summary.json` preserves the corrected compact roster and outcomes.
The replay and pure adapter tests are archived here; they need the pinned pilot
environment and operator-supplied asset/policy/source inputs. Private raw
receipts, deployment identities, credentials and full logs remain outside Git.

```sh
python3 -m unittest discover -s research/isaaclab-v89-parity -p test_replay_contract.py
```
