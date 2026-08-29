# MicroDuck Policy Bench

Policy Bench is the offline, open-source system of record for training
candidates. It snapshots model bytes, records their hashes and source commits,
attaches deployment-rehearsal evaluations, compares candidates, and promotes a
reviewed policy through explicit safety stages.

It is not a replacement for Pollen's robot runtime or browser simulator.
Factory policies and the official playground are the default experience;
Policy Bench exists to prove when a custom candidate is actually better.

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
and (for roller/swizzle policies) reverse tracking, stopping, turning, and
stroke cycles. Component scores and
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

For remote interactive play, forward the dashboard plus the viewer pool. Put
this once in the SSH config on your laptop so every saved model can get its own
arena without reconnecting:

```bash
Host microduck-spark
    HostName <spark-address>
    User ducklab-user
    LocalForward 8091 localhost:8091
    LocalForward 8080 localhost:8080
    LocalForward 8090 localhost:8090
    LocalForward 8081 localhost:8081
    LocalForward 8092 localhost:8092
    LocalForward 8082 localhost:8082
    LocalForward 8093 localhost:8093
    LocalForward 8083 localhost:8083
    LocalForward 8094 localhost:8094
    LocalForward 8084 localhost:8084
    LocalForward 8095 localhost:8095
    LocalForward 8085 localhost:8085
    LocalForward 8096 localhost:8096
```

Connect with `ssh microduck-spark`, then open `http://localhost:8091`.
**Open simulator** on a finished run verifies its immutable ONNX artifact and
opens it in Pollen's colorful browser arena through the dashboard tunnel. The
arena selects feet/rollers and the correct policy slot automatically.
It also keeps a rolling six-second physics buffer. **Replay → Save backflip**
stores the previous six seconds of qpos, qvel, policy actions, and commands on
the Spark under `reports/demonstrations/`; **Ball off** persists across robot
resets for repeated clean-floor attempts.

Active jobs have a different action: **Watch training live** launches six
sample environments from the newest checkpoint in mjlab/Viser. The actual PPO
trainer continues running thousands of environments headlessly; the viewer is
a current checkpoint sample and never replaces the finished-model arena. A new
checkpoint supersedes the prior live preview for that experiment so the port
pool cannot fill during a long run. The live viewer also maintains a rolling
motion buffer: choose a skill and click **Save last attempt** immediately after
a successful manual maneuver to store five seconds under
`reports/demonstrations/`.

Each finished skating run also offers **Deployment check** when it has an ONNX
export. This is deliberately separate from the interactive training arena: it
runs the normalizer-aware ONNX through Pollen's CPU MuJoCo inference path at
50 Hz, exercises settle, forward, coast, reverse, and heading phases, then
stores the transparent score in the run report. Pollen's `infer_policy.py`
viewer is a native desktop window rather than an SSH-forwardable browser UI;
Policy Bench therefore uses its headless runtime path for repeatable deployment
qualification, Pollen's browser arena for finished-policy driving, and
mjlab/Viser only for sampled live-training visualization and advanced debugging.

## Resource modes

New jobs default to **Shared**, which leaves vLLM and Hermes inference online.
Choose **Training priority** for an overnight/max-throughput run. Policy Bench
then pauses the `qwen38-hermes-vllm` container before the trainer starts,
disables its Docker auto-restart for the training window, and restores it after
the trainer exits. The Hermes watchdog honors the live training marker instead
of treating the intentional pause as a failure.

The active mode and vLLM state are visible in the dashboard status. If the host
hard-resets during a priority run, recover explicitly with:

```bash
./scripts/resource-profile.sh restore
```

Do not remove the marker by hand: it records whether vLLM was online before the
run and therefore whether Policy Bench should restart it.

## Upstream provenance

Every candidate manifest records both repositories' commits and dirty states,
plus the Pollen checkout's remote and branch. The control-room header shows the
currently installed `microduck_rl` revision; each run card shows the revision
that produced its saved model. Upstream changes should be pulled, tested, and
pinned deliberately rather than automatically changing existing experiments.

The dashboard never guesses that an occupied port belongs to the requested
model. It skips externally occupied pairs and reports a clear error if the pool
is full. A normal dashboard shutdown also closes the simulations it owns, which
prevents stale arenas from surviving a restart.

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
plan. Only registered simulator tasks (currently walking, roller skating,
swizzle, and Roller Hop) can become a launch action; a goal such as a backflip
receives an honest explanation that its environment and reward still need to
be added.
The proposed configuration is shown back to the user and requires a separate
**Confirm training launch** click. Full training is refused while another
training process is detected. No arbitrary shell text is ever executed.

Codex is an interpreter, not an executor: its JSON is schema-checked, bounds are
re-checked, and the existing allowlisted launcher is the only process path. Set
`DUCKLAB_CODEX=0` for the deterministic offline parser.

Reports and controls contain no external JavaScript, fonts, analytics, or
network calls. Training continues in its own process if the dashboard closes.

## End-of-training automation

Every standard training launcher now owns its complete finish path. After a
successful skating run it verifies the ONNX contract, registers the immutable
candidate, executes the deployment simulation score, imports training curves,
and leaves the model at `evaluated`. Walking runs are verified, registered, and
receive curves while their dedicated locomotion scoring battery is still being
built. A failed trainer never scores a stale artifact.

Scoring is automatic; judgment is not. Policy Bench never auto-stars or
auto-promotes a model. Interactive controller review and explicit human
promotion remain required. The **Deployment check** button can rerun the same
deterministic score at any time.
