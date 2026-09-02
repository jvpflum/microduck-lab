# Reproducing and extending DuckWing with a coding agent

This guide is designed for Codex and other repository-aware coding agents. The
agent can inspect, edit, test, and evaluate the project, but the benchmark—not
the agent's opinion—decides whether a model improves.

## Fresh-clone reproduction

```bash
git clone --recurse-submodules https://github.com/jvpflum/microduck-lab.git
cd microduck-lab
make bootstrap
make preflight
make test

(cd releases/v67 && sha256sum -c SHA256SUMS)
./scripts/verify-artifact.sh \
  "$(pwd)/releases/v67/duckwing-v67-joint-specialist-fusion.onnx"
```

The public release is inference-complete: no private checkpoint is needed to
verify V67 or run the current V69 state-guard search. Raw PPO continuation does
require a private normalizer-aware `.pt` donor, as documented in
[`RACE5.md`](RACE5.md); do not approximate one by distilling the ONNX actor and
then call the result a lossless continuation.

Run the public next-step search with:

```bash
make v69-search
```

It reconstructs the best V68 research challenger from committed ONNX inputs,
wraps it with measured body-yaw fallback to V67, and evaluates every candidate
at `frictionloss=0.003` and `1.75 A`. Generated models and full reports remain
ignored under `reports/`; only reviewed release artifacts enter Git.

## Copy/paste agent prompt

```text
Work inside this DuckWing repository. Read AGENTS.md, README.md,
docs/CODING_AGENT_WORKFLOW.md, releases/v67/README.md, and
releases/v67/leader-metrics.json before acting.

Goal: improve DuckWing V67 toward a stable 5 mph roller-skating policy under
the exact official Race5 contract: wheel frictionloss 0.003, current limit
1.75 A, deterministic CPU MuJoCo evaluation, and the committed line controller.
Preserve V67 as immutable. First verify repository/submodule state, available
memory/disk, the V67 SHA-256, and the existing benchmark. Then run the smallest
reproducible experiment that directly tests a stated hypothesis. Evaluate saved
artifacts against V67 on 100-ft time, sustained/top speed, acceleration, drift,
heading, tilt, bilateral contact, braking, idle, cruise, and turns.

Do not promote a candidate unless it is faster in both 100-ft time and sustained
speed with zero regression on every retained metric and all control gates pass.
Treat reward-only gains, permissive-physics speed, and single-metric wins as
research results. Keep raw checkpoints/reports out of Git, record exact commands
and commits, update the research notes, run relevant tests and git diff --check,
and finish with both the main repository and submodule clean. If a candidate
fails, explain the limiting metric and propose the next falsifiable experiment.
```

For a narrower task, replace the goal paragraph while preserving the physics,
baseline, and promotion constraints. Good prompts name one bottleneck—for
example heading error during high-speed propulsion—and ask for an ablation that
can distinguish whether the cause is reward design, controller authority,
contact phase, or policy composition.

## Heavy GPU runs on a shared Spark

The repository defaults to `shared` and will not touch unrelated services. An
operator who explicitly wants training priority can supply reversible hooks:

```bash
export DUCKLAB_RESOURCE_PROFILE=training-priority
export DUCKLAB_RESOURCE_STOP_CMD='systemctl --user stop hermes-gateway.service && sudo docker stop qwen38-hermes-vllm'
export DUCKLAB_RESOURCE_RESTORE_CMD='sudo docker start qwen38-hermes-vllm && systemctl --user start hermes-gateway.service'
```

Use these only for a real GPU training command. Verify service health after the
restore hook, and never use them for CPU evaluation or ONNX composition. Check
for an already-running trainer before launch so duplicate or orphaned jobs do
not compete for unified memory.

## Definition of done

- Exact code, submodule revisions, seeds, physics, controller, and model hashes
  are recorded.
- A fresh clone can execute the documented public path.
- The comparison uses V67's committed leader data and all retained gates.
- The result is labeled either promoted, rejected, or exploratory—never vaguely
  described as “better.”
- Generated bulk data and private trainer state are absent from Git.
- Tests pass, documentation matches the measured result, and `git status` is
  clean after the reviewed commit is pushed.
