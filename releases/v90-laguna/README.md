# DuckWing V90 Laguna

V90 Laguna adds a learned racing command controller to the unchanged V89 braking-control policy and published 180°, 360° and 720° trick controller. It expands the available behaviors; it does not establish a new general-purpose speed, braking or agility record.

**Research bundle: racing validation passed; combined mode-transition validation failed.** Keep V89 as the recommended general-purpose release. Use the established trick workflow separately; reliable automatic trick-to-race recovery and arbitrary on-track trick activation are not qualified.

## Measured Laguna improvement

| Comparison | Earlier course driver over V89 | V90 Laguna |
| --- | ---: | ---: |
| Original evaluation: clean laps | 15/64 | **55/64** |
| Original evaluation: fastest clean lap | 5:49.06 | **5:44.00** |
| Fresh -3° heading: clean laps | 12/48 | **42/48** |
| Fresh -3° heading: fastest clean lap | 5:47.58 | **5:39.24** |
| Fresh +3° heading: clean laps | 11/48 | **42/48** |
| Fresh +3° heading: fastest clean lap | 5:51.84 | **5:40.26** |

Fresh completion rates rose from 25.0% and 22.9% to **87.5%** each. Off-track failures fell from 1 to 0 and 2 to 1 respectively; all four fresh cohorts had zero body contacts. These compare course controllers over the same V89 base, not V89 manual driving or Pollen on Laguna. No Pollen Laguna comparison was run.

A later continuation re-evaluated this retained model at **60/64**, fastest **5:35.68**, then regressed to 44/64 and 5:43.14 after training. That continuation is rejected and its weights are not included. The 5:35.68 figure is an observed fastest clean lap from a separate cohort, not the time from the paired original comparison.

The course is an approximate 59.84 m flat layout, 0.70 m wide, with 200 ordered checkpoints and a 360-second timeout. It does not reproduce Corkscrew elevation or the full-size circuit. Success requires ordered completion without off-track, body-contact or excessive-tilt termination. GPU MuJoCo-Warp evaluates the race; published V89 driving/tricks use CPU MuJoCo. One trained seed and multiple simulated worlds do not establish cross-seed robustness. All results are simulation-only.

## Retained V89 motion and tricks

`policy.onnx` is byte-identical to V89 Tricks 2. Existing V89 braking and Pollen comparison figures remain evidence for that unchanged base, not new V90 measurements. `trick_controller.py` is also byte-identical: the published standalone 180°, 360° and 720° tests each passed 16/16 trials in CPU MuJoCo, in both directions. See [V89 evidence](../v89/), [trick evidence](../v89-tricks-2/) and the [labeled simulation compilation](tricks-showreel.mp4).

The separate GPU skate → trick → race → skate test passed **10/48** behavior checks although command-mode handoffs were correct. This failed the combined release gate. Retaining component files does not prove reliable behavior across their transitions. Per-case failures are preserved in [evaluation-summary.json](evaluation-summary.json).

## Inference integration

Requires Python with NumPy and an ONNX runtime appropriate to the caller. Run `python3 test_adapter.py` and verify `SHA256SUMS` before integration. The bundle is inference-only: no optimizer state, private endpoints or training credentials.

`Controller` starts in `skate`, passing through 13 user commands. `set_mode('race')` selects route control. `start_trick(heading, degrees=180, direction=1)` selects a trick; cancellation is explicit. Treat the automatic combined-mode path as experimental due to the failed transition test.

The caller owns V89's float32 `[1,15616]` input: 256 frames of 61 observations in term-major order. Repeat the initial frame on reset; update at 50 Hz and preserve history through modes. Use the [V89 history adapter](../../tools/evaluation_policy.py). Controller output supplies commands to that observation builder, never direct joint positions.

For racing, evaluate route control at 10 Hz and hold its commands across the 50 Hz base-policy steps. The 14 route observations in order are: the next three checkpoint offsets in body-frame XY divided by 3 (six values); body-frame forward and lateral velocities; body yaw rate; three body-frame gravity components; checkpoint index divided by point count; nearest-polyline distance in metres. `course.json` supplies the checkpoint geometry. This requires simulator localization and route state; it is not camera-based navigation.

Guidance uses the body-frame bearing `a` to the next checkpoint: yaw = clip(1.2*a, -0.6, 0.6); forward = clip(0.55*(1-0.65*clip(abs(a)/1.57,0,1)),0.15,0.55). `RacingActor` is a 14→64→64→2 network with tanh hidden layers. Tanh output residuals scale forward by 0.35 and yaw by 0.6, then clip commands to [0,1] and [-1.2,1.2]. The exported NumPy actor matched Torch to a maximum absolute error of 3.73e-8 on 100 sampled observations.

## Training and provenance

Ray scheduled GPU simulation and PPO training; MinIO remains the platform's private artifact-storage service. The racing controller used 12 full-lap rounds with 64 worlds, 3,600 driver steps per attempt, completion/time reward, potential shaping and a separate critic. These changes were applied together; their individual effects were not isolated. Failed apex/trail-braking experiments and the rejected continuation are excluded.

Source racing checkpoint SHA256: `2d1a028358803e845e58c10d22bcb80b4c9ae343f679e2b4c4d56846b2cfe3cd`. The public export contains actor parameters only. See `manifest.json` and `SHA256SUMS` for the actual distributed bytes.

Course layout adapted from [AlexJ's public-domain Laguna Seca layout](https://commons.wikimedia.org/wiki/File:Laguna_seca_layout.svg). Course width and robot scale are simulation choices. Code and model licensing follow the repository's Apache-2.0 license. No physical-robot validation is claimed.
