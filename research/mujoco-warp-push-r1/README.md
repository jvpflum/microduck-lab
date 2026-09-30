# MuJoCo-Warp preview-controller campaign

Started 2026-09-29. Status at publication: one training seed running, one queued.
No new policy is qualified or promoted. V89 remains the recommended skating
model, including its established braking and standalone trick evidence.

Hypothesis: a bounded controller residual with longer track preview can learn
better corner entry, turning and acceleration on exit while retaining the
preserved racing controller's completion and safety results. More PPO iterations
alone previously regressed racing; this campaign changes the controller's
information and protects its starting behavior.

## Simulator and learning

The original source MJCF runs directly in MuJoCo-Warp CUDA. Physics, V89
inference and PPO optimization all use the GPU. This is the original stack,
outside Isaac Lab, Newton and PhysX. The passing learning smoke used MuJoCo
3.10.0, Warp 1.15.0 and PyTorch 2.9.1+cu128 on an RTX 5090.

The 14-input racing actor is frozen. A new 28-input residual sees its original
route/sensor observations, four farther route points, four signed turn-preview
features, and the last forward/yaw commands. Its initially zero output preserves
the anchor's action mean exactly. Additional latent actions are bounded to
0.4 forward and 0.5 yaw; the existing command tanh/scales bound changes to at
most 0.14 forward-command units and 0.30 yaw-command units. The motor policy,
torque limits, wheel friction and published trick controller are unchanged.

The critic has separate parameters. PPO uses actor learning rate 3e-5, critic
learning rate 1e-4, clip 0.08, fixed exploration std 0.035, approximate-KL stop
0.015, three epochs, gradient clipping, and command anchoring. Idle steps and
terminal worlds are excluded from actor likelihood updates. Full attempts keep
terminal clean-finish/failure reward, potential shaping, and time cost from the
previous clean-lap objective. No rewards are paid merely for a requested brake
or steering command.

Reset-only training variation is +/-3 degrees heading and +/-0.003 radians
joint noise. No pose, velocity or joint-state correction occurs during a lap.
The original approximately 59.84 m flat, 0.70 m wide course, ordered gates,
wheel boundaries and 360-second deadline remain. This does not implement
Corkscrew elevation or physical-robot navigation.

## Fixed campaign and comparison

Two independent seeds, 99001 and 99002, each run 128 worlds, 24 full-attempt
updates, up to 3,600 decisions per update. Upper budget: **22,118,400 driving
decisions / 110,592,000 low-level control steps**, excluding tests and evaluation.
Each Ray job reserves one full GPU, four CPUs and 12 GiB logical memory, has a
12-hour execution limit, saves updates 8/16/24, and has no automatic retries.
The `sim:duckwing-v89-mjwarp` software capability limits placement to a tested
environment; operators advertise it only after readiness checks.

Every saved checkpoint faces the same two independent 64-world reset cohorts.
The new predeclared advancement gate requires no loss of clean-lap count and no
increase in off-track, body-contact, fall, nonfinite or barrel-collision counts;
at least 16 paired successful laps; median paired time ratio <=0.95; and faster
times in at least 60% of those pairs. A deterministic selection rule chooses
among eligible checkpoints before fresh 48-world final cohorts at -3, +3 and
+6 degrees. The +6-degree stress cohort requires completion/safety retention.

This is a **new experiment contract**, not a revision of earlier results. The
old continuation's requirement for eight additional finishes was unreachable
when its starting control achieved 60/64. Its failed result remains rejected.
This campaign targets time advancement while retaining safety and completion.

`campaign.json` freezes the complete budget, cohorts, limits and gates.
`qualification.py` and its five tests reject survivor-only speed gains, added
falls, insufficient paired evidence and unchanged performance presented as
advancement. A passing racing result would still require standalone and actual
course skate/trick/recovery/race/skate qualification before a general release.

## Verified readiness and source

The actual Ray GPU learning smoke completed 128 driver steps in eight worlds,
with 864 actor-controlling samples and three actor updates. Both the preview
encoder and output layer changed. It had zero body contacts and off-track
failures. Initial anchored action error was exactly zero; low-level CUDA
inference/history packing matched ONNX Runtime in the physical rollout.
See `learning-smoke.json`. This is readiness evidence, not a successful lap
or advancement result.

The worker Python files are exact research sources, with hashes
in `source-provenance.json`. They require the prepared operator environment:
the existing MicroDuck audit imports and CPU-contract helpers, pinned upstream
robot source, privately supplied preserved racing checkpoint (`anchor.pt`),
and the evaluated V89 graph. Their relative home-directory layout is an
environment template, not a shared filesystem or an automatic installer.
Raw optimizer checkpoints, credential files, addresses, private job receipts,
and training logs are excluded from Git. The public V89 ONNX is separately
metadata-sanitized, so its published hash differs from the recorded evaluated
training copy. The graph-equivalence privacy record remains in the release.

Local gate tests:

```sh
python3 -m unittest discover -s research/mujoco-warp-push-r1 -p test_qualification.py
```

Ray's successful command exit is not scientific qualification. Inspect the
complete checkpoint comparisons and fresh final verdict before making claims.

## Ray submission for prepared operators

`schedule.py` uses the public `compute/home_ray` client and operator-managed
`HOME_RAY_URL` / `HOME_RAY_TOKEN_FILE`. Supply the preserved private anchor
whose digest is frozen in the source; it is not part of this public bundle.
The submitter saves receipts and returned raw artifacts under ignored
`private-results/`. Its transport-path adaptation does not change worker code.
Do not advertise the software capability before verifying the required CUDA
environment, robot imports and policy hashes.

```sh
python3 research/mujoco-warp-push-r1/schedule.py smoke --anchor /operator/private/anchor.pt
# Inspect the passing GPU learning smoke and its source hashes first.
python3 research/mujoco-warp-push-r1/schedule.py training --anchor /operator/private/anchor.pt
```

The training command requires its own successful, unchanged-source smoke and
refuses duplicate local submission receipts. It does not silently fall back to
CPU or change the fixed two-seed roster.
