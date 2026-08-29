# MicroDuck Policy Bench

Policy Bench is the offline, open-source system of record for training
candidates. It snapshots model bytes, records their hashes and source commits,
attaches deployment-rehearsal evaluations, compares candidates, and promotes a
reviewed policy through explicit safety stages.

It is the engineering backbone beneath the simple product flow described in
[PRODUCT_VISION.md](PRODUCT_VISION.md); advanced users can use its CLI while a
future guided UI hides these details.

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

Ingest the TensorBoard scalar stream and render local SVG curves:

```bash
./scripts/policy-bench.sh metrics <run-id>
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

Each evaluated checkpoint receives a transparent 0–100 heuristic score. It
combines command tracking, ground contact, stability, smoothness, lateral slip,
and (for swizzle) reverse tracking and stroke cycles. Component scores and
weights are visible in the evaluation JSON; the score is a triage aid, never an
automatic promotion decision. Use the star button to keep a human shortlist.
Only one candidate per skill is starred at a time:

```bash
./scripts/policy-bench.sh star <run-id> --note "Best reverse stability so far"
./scripts/policy-bench.sh unstar <run-id>
```

Equivalent convenience targets are available as `make bench-metrics RUN=…`,
`make bench-score RUN=…`, and `make bench-star RUN=… NOTE='…'`.

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

## Dashboard control center

Serve the local report on port 8091:

```bash
make bench-dashboard
```

For remote interactive play, forward all three local services:

```bash
ssh -L 8080:localhost:8080 \
    -L 8090:localhost:8090 \
    -L 8091:localhost:8091 \
    <ssh-user>@<spark-address>
```

Open `http://localhost:8091`. Every immutable run has a **Play** button. A
click validates the checkpoint hash, launches the correct mjlab task on CPU,
and opens Viser and the Xbox controller in separate browser tabs. If the
browser blocks those tabs, allow pop-ups for `localhost:8091`. Only one viewer
is managed at a time; use **Stop viewer** before selecting another candidate.
The control center refuses to take over or kill an older viewer it did not
launch, so manually stop an existing terminal-launched Viser session first.

### DuckLab Assistant

The dashboard lists one row per human-level training run. Click **Open run** to
expand its saved versions; checkpoint files stay out of the main view.

The dashboard chat accepts plain-language goals such as:

```text
train swizzle for 8000 iterations with 4096 environments
help me train MicroDuck to skate backwards
help me train MicroDuck to do a backflip
play iteration 2250
what is running?
stop the viewer
```

Codex on the Spark can translate a natural-language goal into a validated task
plan. Only registered simulator tasks (currently walking, roller skating, and
swizzle skating) can become a launch action; a goal such as a backflip receives
an honest explanation that its environment and reward still need to be added.
The proposed configuration is shown back to the user and requires a separate
**Confirm training launch** click. Full training is refused while another
training process is detected. No arbitrary shell text is ever executed.

Codex is an interpreter, not an executor: its JSON is schema-checked, bounds are
re-checked, and the existing allowlisted launcher is the only process path. Set
`DUCKLAB_CODEX=0` for the deterministic offline parser.

Reports and controls contain no external JavaScript, fonts, analytics, or
network calls. Training continues in its own process if the dashboard closes.

## End-of-training automation

The swizzle completion watcher verifies the final ONNX contract, registers its
immutable candidate, evaluates it, and leaves it at `evaluated`. It never
auto-promotes a policy. Visual review and an explicit human promotion remain
required.
