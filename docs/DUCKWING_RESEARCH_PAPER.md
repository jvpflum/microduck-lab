# From Baseline Roller Locomotion to DuckWing V67

## An artifact-first, agent-driven reinforcement-learning study of speed, control, and policy composition on Pollen MicroDuck

**DuckLab technical report · 2 September 2026**

### Abstract

This report describes how an agent-driven robotics reinforcement-learning
workflow improved the simulated roller-skating performance of Pollen Robotics'
MicroDuck while preserving steering, stability, idle behavior, and braking.
The work began with Pollen's official roller policy and simulator as the fixed
reference. We added a reproducible 100-foot Race5 task, deterministic CPU
MuJoCo evaluation, immutable policy snapshots, and explicit promotion gates.
The research then combined three complementary techniques: conservative PPO
continuation for control-aware locomotion, deliberately permissive speed
discovery to expose gait headroom, and command/state-aware policy composition
to transfer specialist behavior without erasing mature skills.

DuckWing V67 completes 100 feet in 25.815 seconds versus 57.589 seconds for the
Pollen roller reference under the measured comparison, a 55.2% reduction in
elapsed time. Its race-phase sustained speed is 2.240 mph versus 1.066 mph
(2.10×), and its verified 0.5-second top speed is 3.060 mph versus 1.283 mph
(2.39×). Maximum lateral drift falls from 1.25 feet to 0.775 feet and maximum
heading error falls from 11.06° to 10.22°. V67 passes all 15 retained control
gates and wins all nine comparable dimensions in the local benchmark. These
are deterministic simulation results, not physical-robot claims or independent
certification.

The negative results were equally informative. Policies that were faster under
frictionless discovery physics did not transfer directly to official wheel
friction; global fine-tuning often erased braking or launch behavior; and 586
V68 mixtures plus 150 V69 state-gated mixtures could not improve speed without
at least one regression. The central lesson is to separate discovery, transfer,
and qualification, and to treat the deployment evaluator as the final objective
rather than PPO reward alone.

## 1. Research question

The practical goal was easy to state and difficult to measure: make a miniature
humanoid on passive wheel blades skate toward 5 mph while remaining useful as a
robot. “Useful” meant more than surviving one fast rollout. The policy needed to
start from rest, track a line, avoid excessive yaw and lateral drift, keep both
blades in meaningful contact, respond to turns, hold still at zero command, and
brake from speed.

That creates a multi-objective problem with conflicting gradients. A gait can
gain velocity by accepting larger yaw oscillations. An actor can achieve a high
instantaneous peak while failing to cover distance. A reward can increase even
as deployable behavior gets worse because regularization, curriculum, or
normalization terms dominate the scalar total. We therefore asked:

1. How much speed headroom exists in the MicroDuck passive-roller morphology?
2. How much can be transferred into the official-physics, control-complete
   policy without regressing any promoted measurement?

## 2. Experimental system

### 2.1 Robot and software stack

The study uses Pollen Robotics' open MicroDuck runtime, robot model, browser
arena, and reinforcement-learning environment as upstream components. DuckLab
adds experiment orchestration, evaluation, provenance, artifact handling, and a
local dashboard. Training uses the upstream 61-dimensional deployable actor
observation and 14 joint-position targets at 50 Hz. Actors are exported as
normalizer-aware ONNX graphs and replayed in CPU MuJoCo.

Experiments ran on a DGX Spark/GB10 and an RTX 5090 worker. The Spark's memory is
unified, so jobs were scheduled using host memory, GPU process attribution, and
swap together rather than assuming an independent VRAM pool. Git carried code
and curated inference artifacts between workers. Raw optimizer checkpoints and
local run metadata stayed private and were never treated as portable releases.

### 2.2 Frozen official-physics contract

The canonical Race5 comparison uses:

- wheel joint `frictionloss = 0.003`;
- actuator current limit `1.75 A`;
- 200 Hz physics and 50 Hz policy control;
- deterministic CPU MuJoCo replay;
- a 100-foot centreline race plus separate control phases;
- the same measured line-hold mechanism for comparable policies.

V67's line controller uses yaw proportional gain `0.70`, lateral proportional
gain `0.22`, yaw derivative gain `0.07`, and maximum yaw correction `0.15
rad/s`. Controller effort remains visible because a model that needs aggressive
external correction is not equivalent to one that is naturally straight.

### 2.3 Metrics and promotion rule

The benchmark records sustained and top speed, 100-foot elapsed and trap speed,
first-second acceleration, cross-track drift, heading error, tilt, bilateral
blade contact, stopping, idle creep, cruise, and left/right turn response. The
current battery contains 15 qualification gates and nine measurements that can
be compared directly with the Pollen reference.

A candidate cannot be promoted on aggregate score alone. It must complete the
100-foot race sooner, increase sustained speed, preserve or improve top speed
and acceleration, and show no regression in drift, heading, tilt, grounded
fraction, or stop time. Independent idle and high-speed-brake checks must also
pass. This deliberately conservative rule makes “leader” mean a strict Pareto
advance over the incumbent.

## 3. Agent-driven research method

Codex was used as the primary research and coding agent. The agent inspected run
artifacts, wrote task/reward/evaluation changes, launched bounded experiments,
monitored jobs, scored checkpoints, and updated the repository. The dashboard
was the human-facing evidence surface rather than the source of research logic.

The working loop was:

1. State one bottleneck and a falsifiable hypothesis.
2. Run the smallest smoke test that can expose wiring or normalization errors.
3. Launch one memory-bounded training job at a time.
4. Save frequent checkpoints and evaluate them, not only the final iteration.
5. Compare deterministic deployment behavior with a frozen incumbent.
6. Preserve useful specialists even when they fail all-around qualification.
7. Promote only a complete no-regression result; document the rest as negative
   evidence.

Many of the strongest contributions came from recombining previously rejected
specialists, not from the final checkpoint of a single long PPO run.

## 4. Progression of the policy

### 4.1 Establishing Race5: V11

The initial Race5 work made world-forward progress and centreline behavior
measurable. A conservative PPO continuation produced V11, the first public
all-around DuckLab baseline. V11 increased sustained speed from 1.066 to 1.419
mph and reduced 100-foot time from 57.589 to 44.06 seconds. It reduced maximum
drift from 1.25 to 1.064 feet and heading error from 11.06° to 7.30°, while
passing 14 of 14 then-current retention checks.

V11 demonstrated that the official policy was not the morphology's performance
ceiling. It also became a control anchor: later speed policies were judged by
whether they retained V11-like behavior outside the straight full-command mode.

### 4.2 Separating discovery from qualification

The early Race5 reward combined roughly 30 incentives and penalties, including
action smoothness, heading, lane error, lateral velocity, pose, torque, gait
symmetry, lean, glide, and skating form. That was appropriate for qualification
but restricted aggressive gait discovery.

A separate SpeedDiscovery task removed most control/form terms, retained weak
anti-exploit costs, and rewarded measured forward chassis velocity. Its
curriculum raised performance milestones without sending literal
multi-metre-per-second commands outside the actor's known command distribution.
The best scout reached a recorded 5.41 mph instantaneous peak under zero wheel
frictionloss, approximately 5.05 mph over its best one-second window and 4.18
mph mean over 20 seconds.

That answered the first question: the simulated robot had gait headroom. It did
not answer the second. Different physics and missing control constraints meant
the scout was a teacher and transfer seed, never an official leader.

### 4.3 Transfer failures and the specialist library

Directly adapting fast scouts to `0.003` wheel friction exposed several failure
modes. Larger exploration noise could erase the deterministic skate within a
few updates. Literal high-speed commands caused inherited actors to crouch or
stop because those values were out of distribution. Some checkpoints gained
peak speed while losing displacement; others could accelerate but not hold idle
or brake safely.

Parallel RTX 5090 experiments produced V56–V60 and later V63/V65 specialists.
These were preserved with scrubbed evaluations instead of discarded. V57b/V59
were useful high-speed teachers. V63 supplied a stronger mid-speed branch. V65
combined a low-speed control branch, V63 at mid speed, and V59 at high speed.
An unsuccessful complete controller could still be a valuable capability donor.

### 4.4 Command-aware composition: V61 and V66

DuckLab next composed actors at the ONNX action level. A control-aware incumbent
handled idle, cruise, and turns; a specialist contributed during straight,
high-speed commands. Contributions tapered as commanded yaw increased, avoiding
a hard discontinuity during steering.

V61 combined the mature control policy with 85% V57b contribution. It completed
100 feet in 27.449 seconds and reached 3.026 mph top speed while passing 15 of
15 gates. Its first-second acceleration remained weak, so it was retained as a
qualified frontier candidate rather than declared the all-around champion.

V66 routed 96.5% of V65 through the control shell. It improved 100-foot time to
26.323 seconds, race-phase sustained speed to 2.204 mph, and long-run drift to
0.827 feet. It passed all 15 gates and won all nine Pollen comparison
dimensions. Structural routing transferred speed more reliably than asking one
global PPO update to remember every mode.

### 4.5 Joint-specialist fusion and braking: V67

The V47 official-friction specialist had a native 26.145-second 100-foot result
and useful propulsion, but its 18.2° heading error made it unsuitable alone.
V67 imported only the action components that moved the frontier:

- 25% specialist authority on symmetric hip yaw and hip roll;
- 105% authority on hip pitch, knee, and ankle propulsion joints;
- zero specialist authority on head/neck joints;
- activation only for forward command above `0.5 m/s`;
- yaw-command taper from `0.08` to `0.25 rad/s`;
- a separate V65 moving-brake route at effectively zero command.

V67 improved every primary V66 Race5 measurement in two identical deterministic
replays. It reduced 100-foot time by another 0.507 seconds, increased long-run
sustained speed to 2.727 mph and top speed to 3.060 mph, improved acceleration,
drift, heading, tilt, grounded fraction, and low-speed stop time, and passed
independent idle and high-speed braking gates.

## 5. Results

| Metric | Pollen roller | V11 | V61 | V66 | V67 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Race sustained speed (mph) | 1.066 | 1.419 | 1.733 | 2.204 | **2.240** |
| Verified top speed (mph) | 1.283 | 1.649 | 3.026 | 3.021 | **3.060** |
| 100-ft time (s) | 57.589 | 44.060 | 27.449 | 26.323 | **25.815** |
| First-second acceleration (m/s²) | 0.323 | 0.501 | 0.288 | 0.443 | **0.471** |
| Maximum lateral drift (ft) | 1.250 | 1.064 | 1.080 | 0.827 | **0.775** |
| Maximum heading error (deg) | 11.06 | 7.30 | 10.55 | 10.66 | **10.22** |
| Qualification gates | reference | 14/14 | 15/15 | 15/15 | **15/15** |

Relative to Pollen, V67's race sustained speed is 109.4% higher, verified top
speed is 138.5% higher, and elapsed time is 55.2% lower. Drift is 38.0% lower
and heading error is 7.6% lower. The gains are therefore not merely a speed
measurement: the policy covers a fixed distance sooner and preserves the
tracked control properties.

## 6. Negative results after V67

V68 independently tuned bilateral hip-yaw, hip-roll, hip-pitch, knee, and ankle
authority and then optimized the line controller. Across 586 configurations,
the strongest interpretable challenger reached 25.659 seconds, 2.745 mph
long-run sustained speed, and 3.138 mph top speed. It improved acceleration,
drift, tilt, and grounded fraction, but heading error rose to 13.26°, a 29.7%
regression. Controller recovery always surrendered another retained metric.

V69 measured V67 body yaw-rate and allowed the V68 challenger only during calm
yaw states, tapering continuously back to V67 at higher rotation. A
150-candidate grid swept gate thresholds and authorities down to 1%. The
fastest V69 reached 25.571 seconds, 2.755 mph sustained, 3.110 mph top, and
better acceleration and drift. Heading rose to 12.81°, tilt rose, and grounded
fraction fell. No candidate passed. Very small authorities could improve
heading, but then lost speed and acceleration.

These experiments reveal a discontinuous contact-dynamics boundary. Small
joint-target changes can switch the closed-loop gait to a different limit
cycle, so static interpolation need not trace a smooth Pareto front. The next
credible approach is a bounded, phase/contact-aware learned residual trained
with world-frame heading, tilt, and bilateral contact in its selection
objective while V67's non-propulsion routes remain frozen.

## 7. Why the process worked

Five practices produced the largest leverage:

1. **Freeze the deployment benchmark.** Exact friction, current, controller,
   and windows prevented incomparable runs from replacing the leader.
2. **Separate discovery and qualification.** Permissive training exposed gait
   headroom without weakening the definition of success.
3. **Preserve failed specialists.** V47, V57b/V59, V63, and V65 were more useful
   as donors than as complete controllers.
4. **Compose by command and joint role.** Routing protected idle, steering, and
   braking while allowing specialist propulsion where it helped.
5. **Let artifacts outrank reward curves.** Frequent exports and deterministic
   replay found useful intermediate actors and rejected high-reward regressions.

The coding agent accelerated this loop without requiring a custom UI for each
hypothesis. The app's strongest role was narrower: make status, evidence,
simulator access, and human evaluation easy.

## 8. Reproducibility and artifact policy

The repository pins the runtime, simulator, and training fork as submodules. It
includes V11, V61, V66, and V67 inference actors, SHA-256 files, scrubbed
summaries, specialist inputs for V69, and exact composition/evaluation code. A
clean clone can verify V67 and rerun V69 without private state.

Raw `.pt` checkpoints are excluded because they contain optimizer/local-run
state and because an inference ONNX cannot be losslessly reconstructed into the
original resumable checkpoint. PPO continuations must identify their actual
checkpoint provenance; a distilled approximation is not a continuation.

```bash
git clone --recurse-submodules https://github.com/jvpflum/microduck-lab.git
cd microduck-lab
make bootstrap
make preflight
make test
(cd releases/v67 && sha256sum -c SHA256SUMS)
./scripts/verify-artifact.sh \
  "$(pwd)/releases/v67/duckwing-v67-joint-specialist-fusion.onnx"
make v69-search
```

## 9. Limitations

All performance claims are simulation-only. Deterministic replay omits some
actuator, contact, backlash, compliance, thermal, and state-estimation effects.
The comparison is specific to the committed model, evaluator, controller, and
physics. It is not a universal humanoid-skating benchmark.

The 5 mph official-physics goal has not been reached. The frictionless 5.41 mph
scout establishes simulated headroom under a different condition, not proof of
official-friction or hardware performance. Randomized-physics uncertainty is
not characterized by the nominal leaderboard. Robustness and hardware safety
gates must follow.

No claim here should be read as endorsement, independent validation, or
certification by Pollen Robotics or Hugging Face unless a separate citable
record explicitly states it.

## 10. Platform implications and future work

DuckLab is being generalized from a skating scoreboard into an agentic robotics
RL workbench. The research agent remains free to create novel tasks, rewards,
evaluators, and viewers. It publishes a generic run receipt with status,
metrics, hashes, and links. The dashboard displays those receipts and keeps
hardened adapters for mature benchmarks.

The first additional program is an unassisted MicroDuck front flip. Its metrics
are different: success across 256 episodes, takeoff and landing rates, settled
upright behavior, body strikes, clearance, forward/off-axis rotation, and
horizontal drift. Keeping that evaluator separate illustrates the platform
principle: evidence has a common envelope, but each capability defines its own
physics and meaning of improvement.

For skating, the next run should train a small phase-aware propulsion residual,
not globally fine-tune V67 or repeat static mixing. Candidate selection should
use contact-diverse elites and explicitly score forward speed, heading, tilt,
blade contact, and the full deployment replay. Platform priorities are
content-addressed result bundles, generic agent heartbeats, adapter conformance
tests, and lightweight task-specific summaries.

## 11. Conclusion

DuckWing improved the Pollen roller reference by turning reinforcement learning
into an evidence-controlled engineering process. The largest gains came from
measuring the real objective, separating aggressive discovery from deployable
qualification, preserving specialist capabilities, and composing them behind
stable control routes. V67 is the current endpoint: 25.815 seconds over 100
feet, 2.240 mph race sustained, 3.060 mph verified top, and no loss across the
retained qualification battery.

The broader workflow is simple: let the coding agent adapt freely to each new
problem, but require every claimed breakthrough to arrive as a reproducible
artifact with frozen physics, explicit metrics, complete provenance, and a
simulator a human can open.
