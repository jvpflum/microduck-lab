# DuckLab Robotics RL

DuckLab trains, evaluates and runs small-robot policies. A coding agent owns
the research loop; the local dashboard shows jobs, measured results and simulator
links. Ray schedules compute across GPU machines, and MinIO stores private
artifacts.

## DuckWing V89

**V89 is the recommended DuckWing roller-skating model for Pollen MicroDuck.**
It combines high-speed driving, steering, low-speed control and learned braking
in one inference-complete ONNX file. No separate donor models are required.

### What V89 improves

- **19.1% faster high-speed stopping.**
- **17.6% shorter stopping path.**
- **45.7% less mean lateral drift while braking.**
- **33.2% less worst-case braking drift.**
- **More reliable stops:** 16/16 healthy held-out high-speed brakes, up from 14/16.
- **Retained driving control:** Race5 and manual-agility phase metrics matched
  the reference exactly on all 16 paired seeds; low-speed braking was unchanged.

Braking improvements compare V89 with the prior validated DuckWing V80
reference, using the 14 paired episodes where both brakes were healthy.
Reliability counts include all 16 held-out episodes. These percentages are
separate from the Pollen driving comparison below; matching Pollen braking
measurements are not available. See the [braking evidence](releases/v89/evaluation-summary.json).

### V89 compared with Pollen's official roller

| Simulation metric | V89 | Pollen baseline | V89 advantage |
| --- | ---: | ---: | ---: |
| Race sustained speed | **2.268 mph** | 1.068 mph | **2.12×** |
| Verified top speed (0.5 s) | **3.084 mph** | 1.283 mph | **2.40×** |
| 100-ft elapsed time | **25.729 s** | 57.589 s | **55.3% sooner** |
| First-second acceleration | **0.466 m/s²** | 0.323 m/s² | **44.0% higher** |
| Maximum lateral drift | **0.403 ft** | 1.247 ft | **67.7% less** |
| Maximum heading error | **9.54°** | 11.06° | **13.7% less** |

These figures use deterministic CPU MuJoCo, wheel `frictionloss=0.003`,
`1.75 A` motor limits and the recorded policy-specific line-hold settings.
Driving figures come from the frozen benchmark and were verified unchanged
in 16 paired V89 Race5 replays. See the [comparison data and provenance](docs/V89_POLLEN_COMPARISON.json).

V89 also completed **16/16 independent held-out high-speed brakes healthily**,
with mean stopping time **2.406 s**, mean stopping path **1.529 m**, and mean
lateral drift **0.327 m** across the entire cohort. Matching Pollen braking
measurements are not available. See the [evaluation evidence](releases/v89/evaluation-summary.json).

These are simulation results. The 5 mph target and physical-robot validation
remain open.

### Download the model

Download [DuckWing V89 ONNX](releases/v89/duckwing-v89-braking-control.onnx).
The [release directory](releases/v89/) includes the input contract, manifest,
checksums and measured evidence; the model uses the repository's Apache-2.0 license.

```bash
(cd releases/v89 && sha256sum -c SHA256SUMS)
```

V89 takes a float32 `[1,15616]` input: 256 frames of 61 observations in
term-major order, oldest to newest. Repeat the first frame on reset and append
once per 50 Hz control step. The [history adapter](tools/evaluation_policy.py)
and [release notes](releases/v89/README.md) describe integration. Existing
single-frame consumers need this adapter before using V89.

## Skating tricks: V89 Tricks 2

**V89 can now perform 180°, 360° and 720° skating turns in simulation.**
The [V89 Tricks 2 bundle](releases/v89-tricks-2.zip) includes the unchanged
V89 ONNX model, a selectable heading-feedback trick controller, checksums,
per-trial validation data and replay videos. Tricks are composed around V89;
these are not newly trained neural weights.

| Trick | Fresh simulation trials | Time to turn target | Replay |
| --- | ---: | ---: | --- |
| 180° turn and rolling exit | **16/16 passed** | **0.86–1.16 s** | [Video](releases/v89-tricks-2/turn-180.mp4) |
| 360° skating turn and rolling exit | **16/16 passed** | **1.52–2.24 s** | [Video](releases/v89-tricks-2/skate-360.mp4) |
| 720° double spin and rolling exit | **16/16 passed** | **2.86–4.40 s** | [Video](releases/v89-tricks-2/skate-720.mp4) |

Each trick was tested in both directions from eight fresh initial joint-noise
seeds in CPU MuJoCo at 200 Hz physics and 50 Hz control, with the existing
0.003 wheel friction and 1.75 A motor limits. Passing requires the requested
signed rotation within 15°, no body ground contact, tilt below 45°, and two
final seconds with heading error below 20°, mean absolute yaw rate below
0.5 rad/s and rolling speed above 0.15 m/s. Turn time measures reaching that
rotation threshold; recovery is checked separately.

Use `TrickController.start(heading_radians, direction=1, degrees=720)` and feed
its updated commands into the normal V89 observation builder, preserving
history. The release notes explain cancellation and return to ordinary control.
The controller uses simulator heading in these tests; hardware heading
estimation, physical execution and app-button integration remain open.

See the [release notes and integration](releases/v89-tricks-2/README.md),
[180°/360° measurements](releases/v89-tricks-2/validation.json) and
[720° measurements](releases/v89-tricks-2/validation-720.json).
The bar-hop and flat-ground hop experiments are research, with no qualified
hop included in this release.

```bash
(cd releases/v89-tricks-2 && sha256sum -c SHA256SUMS)
```

## Quick start

Requirements: Git, Python 3.12, Node/npm for the browser arena, and a supported
Linux NVIDIA GPU. Windows GPU workers run Linux through WSL2.

```bash
git clone --recurse-submodules https://github.com/jvpflum/microduck-lab.git
cd microduck-lab
make bootstrap
make preflight
make test
make bench-dashboard
```

Open `http://localhost:8091`. For a remote checkout, forward the dashboard over
SSH: `ssh -L 8091:localhost:8091 <ssh-user>@<private-address>`.

## Ray and MinIO

The [compute package](compute/README.md) provides authenticated submission,
durable job receipts, resource-based Ray placement, logs, cancellation and
bounded retries. The head can run jobs alongside workers. ARM64 GB10 and x86
RTX workers share compute, with application environments validated separately
on each architecture.

Optional MinIO/S3 transfers store artifacts by SHA-256 and verify downloads.
Credentials and private endpoints stay in operator configuration outside Git.
See the [privacy guide](docs/PRIVACY.md).

## Working with an agent

Read [AGENTS.md](AGENTS.md), then give the agent a concrete robotics goal.
It inspects prior evidence, designs a bounded experiment, submits it through
Ray, evaluates saved checkpoints and publishes results to the dashboard.
Skating, front flips and walking keep their own evaluation contracts.

See the [agent workflow](docs/CODING_AGENT_WORKFLOW.md),
[platform design](docs/AGENTIC_RL_PLATFORM.md) and
[Policy Bench guide](docs/POLICY_BENCH.md). Raw training checkpoints remain
private; public releases contain reviewed inference models and evidence.
