# MicroDuck Arena campaigns

This program collects our Arena sprint experiments, portable training/evaluation
source, and their measured limitations. Evidence was reviewed for this publication
on **2026-10-25**. The results below come from retained local evaluations of the
pinned official runner; they are not a fresh leaderboard query or submitted scores.

| Campaign | Local result | What it establishes |
| --- | --- | --- |
| V80 driving exported into Arena's roller event | 32.780 s penalized mean; 1/5 finishes | DuckWing driving success did not transfer directly to this event. |
| V89-equivalent driving component with internal gyro feedback | 2.604 s, 5/5 selection starts; **10/20 held-out finishes**, 10 falls | Fast on selection starts, insufficient robustness. Held-out penalized mean: 21.307 s. |
| Own fresh PPO, iteration 2000 | 2.680 s, 5/5 starts | Retained local CPU-judged parent, not a competition record or general walker. |
| Additional 500-update continuation | Iteration 2250: 2.864 s, 5/5; iteration 2500 unjudged | Run failed its unchanged export-parity gate; 49,152,000 additional training transitions did not establish improvement. |
| Upright-progress smoke | Parent 2.680 s → 3.076 s, 5/5 | Five updates and 7,680 transitions passed pipeline checks but regressed race time. |

The reported **1.648 s** time has no independently verified raw receipt in this
review. It is recorded as user-reported in the inventory and is not used for a
record, leaderboard, or release claim. None of these listed campaigns submitted
an entry or changed a protected release.

## Separate Arena competition repository

The related [microduck-arena-work repository](https://github.com/jvpflum/microduck-arena-work)
contains a different campaign: evolution-strategy searches, motor-gain wrappers,
Arena job tooling, entered/reserve policies for sprint, hill and roller events,
and Got a Talent work. The inspected revision is
[`938eb907`](https://github.com/jvpflum/microduck-arena-work/tree/938eb9071aac972242815485873d4ab604e9eef2).
Its reported entry times are not independently re-evaluated by this publication.

This directory is a curated historical PPO/CPU-judge snapshot, **not the complete
Arena archive**. Topic overlap does not make the policies, experiments or results
identical. The separate repo's later entries must not be inferred absent from the
limited local inventory below. No code or policy from that repo has been merged,
deleted, synchronized or archived by this organization change.

`microduck-lab` is the overall project index and current Isaac workshop home;
the separate Arena repo remains linked competition history/tooling. Keep future
shared summaries here and identify the exact source repo/revision for each result.
Do not maintain silently diverging copies or combine the two score tables.

## Evidence and method

[The machine-readable inventory](evidence-inventory.json) preserves denominators,
failed starts, source-receipt hashes, the export failure, and event-specific scope.
These are curated numeric summaries; private operator records and raw policies
are not published here.

The historical judge was the unmodified Arena revision
[`221dc512`](https://huggingface.co/spaces/pollen-robotics/microduck-arena/tree/221dc512f64aad0814251062df0511543ab2b3b4),
using Python 3.12, NumPy 2.4.1, MuJoCo 3.10.0 and ONNX Runtime 1.24.4 on Linux
x86_64 CPU. Roller selection swept 18 forward commands from 0.3 through 2.0 m/s,
then evaluated three finalists on five prescribed starts. Failures incur the
40-second limit; a fast successful run is not the mean. The reference golden
policy scored 9.404 s with 5/5 finishes under this contract.

Learning used batched CUDA physics; it is not bit-identical to the CPU judge.
The source-owned environment models the Arena reset, actual previous actions,
lagged velocity, and four physics steps with five servo-drive evaluations per
50 Hz control. Startup timing and physics differences remain part of the
simulator-transfer question. A roller race score does not establish barefoot
walk/turn/stop control, camera navigation, recovery, or hardware performance.

## Source and reuse

- [Fresh sprint source and usage](fresh-sprint/README.md): PPO, roller environment,
  strict CPU judge, and focused tests; no checkpoints, installers, remote launchers
  or automatic submissions.
- [Existing sprint build](../../scripts/build-pollen-arena.sh),
  [historical training probe](../../scripts/train-sprint-probe.sh), and
  [sprint evaluator](../../scripts/evaluate-sprint.sh): earlier public tools with
  their own contracts. The historical policy-import probe is not a generic safe
  checkpoint conversion path or proof of simulator equivalence.
- [Pollen's MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) supplies
  the upstream MuJoCo Warp/PPO, BAM, backlash and domain-randomization foundation.
- [V80](../../releases/v80/) remains the immutable DuckWing speed benchmark;
  [V89](../../releases/v89/) is its qualified braking-control variant, not a new
  Arena speed record. [Published trick evidence](../../releases/v89-tricks-2/)
  belongs to its own retained control and rendering contracts.

## Event family and “Olympic” scope

The reviewed historical event snapshot contains `sprint-2m` (legs),
`roller-sprint-2m` (rollers), `hill-climb-1m5` (legs, 8° incline), and
`got-a-talent` (a voted performance, not a race). The inventory stores each exact
TOML hash and scoring method. Only the roller event has our measured results in
this publication. “Olympic” describes the broader event-family ambition here;
there is no verified aggregate medal, all-event score, or championship claim.

The current [Isaac workshop program](../isaac-workshop/README.md) is developing a
rounded barefoot locomotion foundation and camera navigation. It requires named
joint and observation semantics, exact normalization, walk/turn/stop and recovery
coverage, solved-contact evidence, and held-out starts. This sprint policy masks
non-forward command slots, so it must not be repurposed as a general turn-capable
walker merely because its actor also has 61 inputs.
