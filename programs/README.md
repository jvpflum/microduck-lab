# MicroDuck program map

One repository, separate research programs. **Current development focuses on
[Isaac workshop walking and vision](isaac-workshop/).** The directories below are
entry points, not duplicate copies of every historical release.

| Program | Robot/runtime | Source and evidence |
| --- | --- | --- |
| [Arena](arena/) | Task-specific competition contracts | Sprint and Arena research, separately labeled results |
| [DuckWing](duckwing/) | Roller MicroDuck, deterministic CPU MuJoCo reference | Existing training tools and immutable V67/V80/V89 releases |
| [Laguna Seca](laguna-seca/) | Higher-level race controller over V89 | V90 research release and failed continuation ledger |
| [Isaac workshop](isaac-workshop/) | Barefoot MicroDuck, Isaac Lab/Newton, Kit RTX cameras | Current portable research modules and scrubbed evidence |

## Related repositories

- [microduck-arena-work](https://github.com/jvpflum/microduck-arena-work): separate
  Arena search/entry/talent campaign; linked from the Arena section, not merged.
- [microduck_rl](https://github.com/jvpflum/microduck_rl): training-environment fork,
  already pinned as an upstream submodule.
- [microduck-simulator](https://github.com/jvpflum/microduck-simulator): browser
  simulator, already pinned as an upstream submodule.

The project index centralizes discovery; it does not claim all related repository
history has been physically consolidated into this checkout.

## Keep programs separate

- Put new program-specific research in its program directory. Keep shared tools
  generic; do not add workshop assumptions to DuckWing training defaults.
- Preserve existing release paths, checksums and model interfaces. A new result
  earns a new version; do not overwrite a historical model.
- Record the exact morphology, runtime, observation/action contract, baseline,
  evaluation conditions and failures with every claimed result.
- Distinguish source tests, numerical parity, native behavior and physical-robot
  validation. A successful job or rising reward is not skill qualification.
- Keep portable source, tests, compact evidence and upstream asset provenance here.
  Raw checkpoints, optimizer state, proprietary asset packs, private endpoints,
  full logs and execution receipts belong in private artifact storage.
- Public program status is a dated snapshot. It must not impersonate a live
  training dashboard or promise that a queued run has completed.

See [privacy rules](../docs/PRIVACY.md), [agent workflow](../AGENTS.md), and the
[historical research ledger](../research/README.md).
