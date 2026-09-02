# Coding-agent instructions

This repository is an agent-first robotics RL workbench and the reproducible
system of record for DuckWing MicroDuck roller-skating research. Read
`README.md`, `docs/AGENTIC_RL_PLATFORM.md`, and the relevant capability/release
notes before changing training or evaluation code.

## Agent-first project protocol

When a user states a robotics objective, own the research loop end to end:

1. Inspect the robot, existing policies, evaluator evidence, compute, and prior
   failures. State a falsifiable hypothesis and a bounded plan before a heavy
   run.
2. Choose the technique that best fits the evidence. New tasks may need new
   environments, rewards, curricula, residuals, policy composition, search, or
   evaluators; do not force them through a pre-existing dashboard workflow.
3. Create or adapt the minimum research code, smoke-test it, check resources,
   prevent duplicate trainers, and launch one attributable job with checkpoints
   and exact provenance.
4. Publish and refresh a schema-v1 agent run receipt with status, progress,
   meaningful task metrics, artifact paths, and a simulation action. Use
   `make publish-agent-run RECEIPT=/path/to/receipt.json`; the dashboard is the
   evidence surface, not the experiment planner.
5. Monitor through completion, evaluate saved checkpoints against a frozen
   task-specific contract, and label the result promoted, rejected, or
   exploratory. A failure still needs its limiting evidence and next hypothesis.
6. Add a built-in entry to `config/robotics-capabilities.json` only after the
   capability benefits from a stable one-click launcher/evaluator/viewer.

Different programs may have unrelated metrics. Never rank a front flip, walk,
manipulation task, or new robot by the skating score. The generic receipt is the
common envelope; each mature capability owns its physics, baseline, metrics,
and promotion rule.

## Source of truth

- The simulation-qualified leader is DuckWing V67 in `releases/v67/`.
- Never overwrite a release artifact. A new version must earn a new directory.
- The official comparison contract is deterministic CPU MuJoCo, wheel
  `frictionloss=0.003`, motor current limit `1.75 A`, and the committed Race5
  controller/evaluator settings.
- A candidate is not a leader because PPO reward or one speed number improved.
  It must finish faster, increase sustained speed, and avoid regression in top
  speed, acceleration, lateral drift, heading error, tilt, grounded contact,
  stopping, idle behavior, turning, and retained control checks.
- Raw checkpoints and local reports stay out of Git. Curated inference-only
  ONNX exports, checksums, and scrubbed summaries belong under `releases/`.

## Required workflow

1. Start with `git status --short --branch` and `git submodule status`.
2. Run `make bootstrap` in a fresh clone, then `make preflight` and `make test`.
3. For skating, verify the V67 checksum and replay contract before comparing a
   candidate. For another capability, identify and freeze its own benchmark.
4. Use unique run names, save exact parameters and commits, and evaluate saved
   checkpoints rather than judging only the final iteration.
5. Run `git diff --check`, relevant tests, checksum verification, and a secret/
   machine-path review before committing.
6. Update the README, release notes, and leader data only after all promotion
   gates pass. If any gate fails, record the result as research and retain V67.

## Hardware and services

DGX Spark/GB10 memory is unified. Check `free -h`, process attribution, disk,
and `nvidia-smi` before heavy GPU work; do not treat it as conventional separate
VRAM. Avoid concurrent memory-heavy inference and full training. Service pause/
restore must be explicit and verified, using the opt-in resource hooks described
in `docs/CODING_AGENT_WORKFLOW.md`. CPU composition/evaluation searches do not
benefit from stopping vLLM or Hermes.

Do not modify CUDA, NVIDIA drivers, Docker, the firewall, vLLM, or Hermes merely
to make an experiment run. Do not expose credentials, install untrusted code,
or commit files containing private paths, optimizer state, or session data.
