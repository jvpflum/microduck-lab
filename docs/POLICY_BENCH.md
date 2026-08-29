# MicroDuck Policy Bench

Policy Bench is the offline, open-source system of record for training
candidates. It snapshots model bytes, records their hashes and source commits,
attaches deployment-rehearsal evaluations, compares candidates, and promotes a
reviewed policy through explicit safety stages.

It does not use W&B, a hosted registry, an account, telemetry, or network
access. Future training launched by the lab sets `WANDB_MODE=disabled`; the
upstream project's compatibility hooks remain untouched but inactive. Policy
Bench's formats are JSON and self-contained HTML. Generated state lives in
`policy-bench/` and is intentionally excluded from Git because it contains
model checkpoints. Source code, schemas, tests, and documentation remain in
Git.

## Register training candidates

Discover every current RSL-RL run:

```bash
make bench-discover
make bench-list
```

Use `./scripts/policy-bench.sh list --task swizzle --latest` when automation
needs exactly the newest registered candidate.

An active training directory produces a new immutable candidate whenever its
latest checkpoint iteration changes. Policy Bench copies the checkpoint, ONNX
export, and training parameter files into that candidate and records SHA-256
hashes. Later changes to the training directory cannot silently alter a
registered candidate.

Register one directory explicitly when needed:

```bash
./scripts/policy-bench.sh register /absolute/path/to/training-run --task swizzle
```

## Evaluate and compare

Run the skating deployment-rehearsal battery for a registered candidate:

```bash
./scripts/policy-bench.sh evaluate <run-id>
```

After two candidates have evaluations from the same suite:

```bash
./scripts/policy-bench.sh compare <candidate-run-id> <baseline-run-id>
```

The comparison produces JSON and a standalone HTML table beneath the candidate
run. Deltas are deliberately descriptive rather than automatically labeled
good or bad: higher speed can be good while higher tilt or action acceleration
is a regression. Stable task-specific gates should be added only after mature
policies establish reviewed baselines.

## Review and promote

Attaching an evaluation moves an experiment to `evaluated`. All later moves
require an identified reviewer and a review note, one stage at a time:

```bash
./scripts/policy-bench.sh promote <run-id> sim-qualified \
  --approved-by ducklab-user \
  --note "Passed numerical battery and Viser/gamepad review"
```

Stages are:

1. `experimental`
2. `evaluated`
3. `sim-qualified`
4. `hardware-candidate`
5. `production`

Promotion to `hardware-candidate` or `production` additionally requires
`--hardware-signoff`. The flag records a deliberate decision; it does not
replace the physical emergency stop, restrained test bench, current and
temperature limits, runtime watchdog, or Pollen health gate.

The newest promoted policy at a stage becomes that task's registry selection.
`scripts/view-final-skate.sh` automatically loads the `sim-qualified` swizzle
checkpoint when one exists and passes its hash check. Until then it retains the
known roller fallback. Override the selected stage with
`DUCKLAB_POLICY_STAGE`.

## Dashboard

Serve the local report on port 8091:

```bash
make bench-dashboard
```

For remote access, add `-L 8091:localhost:8091` to the SSH command and open
`http://localhost:8091`. Reports contain no external JavaScript, fonts,
analytics, or network calls.

## End-of-training automation

The swizzle completion watcher verifies the final ONNX contract, registers its
immutable candidate, evaluates it, and leaves it at `evaluated`. It never
auto-promotes a policy. Visual review and an explicit human promotion remain
required.
