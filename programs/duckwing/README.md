# DuckWing roller locomotion

DuckWing remains the roller-skating program. **V89 is the recommended general-purpose
roller model**; V80 is the frozen speed benchmark and V67 its predecessor. The
current barefoot Isaac workshop effort does not replace these baselines or change
their training defaults.

## Models and history

Earlier archived releases remain available as [V11](../../releases/v11/),
[V61](../../releases/v61/) and [V66](../../releases/v66/). Their original
contracts and limitations apply; they are not the current recommendation.

- [V67](../../releases/v67/): frozen predecessor.
- [V80](../../releases/v80/): immutable speed benchmark.
- [V89](../../releases/v89/): retained driving plus qualified braking improvements.
- [V89 Tricks 2](../../releases/v89-tricks-2/): separately evaluated skating turns.
- [Roller policy suite](../../docs/ROLLER_POLICY_SUITE.md) and
  [research paper](../../docs/DUCKWING_RESEARCH_PAPER.md): earlier work and context.

## Training and evaluation

Use the existing [skating guide](../../docs/SKATING.md),
[Race5 contract](../../docs/RACE5.md), [training scripts](../../scripts/) and
[evaluation tools](../../tools/). The source remains in its established locations
so consumers and launchers do not break.

The official roller comparison uses deterministic CPU MuJoCo, wheel frictionloss
0.003 and motor current limit 1.75 A. Do not compare these scores directly with
barefoot Arena sprint times or workshop navigation. See the immutable release
summaries for denominators, controller settings and limitations.
