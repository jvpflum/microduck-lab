# MicroDuck Platform Product Contract

The platform is for people who want to teach a robot, not people who want to
become RL infrastructure engineers.

## The user journey

```text
Sign in
  ↓
Describe a behavior in plain language
  ↓
Review the proposed skill, simulator, curriculum, and safety limits
  ↓
Start training with one confirmation
  ↓
Watch understandable curves and checkpoint progress
  ↓
Play any checkpoint with Viser and an Xbox controller
  ↓
Compare candidates, star the favorite, and record notes
  ↓
Promote through simulation and hardware gates
  ↓
Use the robot, with rollback always available
```

The user should not need to know PPO, MJCF, actuator models, ONNX normalizers,
SSH, TensorBoard, or filesystem paths. Those remain visible as an advanced
diagnostic layer for engineers, never as required inputs.

## Product surfaces

1. **Home** — current robot, active training, latest candidates, and one clear
   next action.
2. **Teach** — a guided skill form plus plain-language assistant. It converts a
   request into a typed experiment specification and always shows the plan
   before launching work.
3. **Runs** — grouped experiments with checkpoint snapshots, progress, curves,
   scores, failures, and Play buttons.
4. **Review** — side-by-side candidate comparison, transparent score
   components, stars, notes, and promotion controls.
5. **Robot** — the currently promoted policy, Xbox play control, health, logs,
   and rollback.

## Safety and trust

Natural language may propose actions, but only validated typed requests can
start training or deployment. Every model is tied to source revisions,
configuration, metrics, hashes, and an explicit reviewer. Simulation promotion
never implies physical-robot approval. Hardware deployment remains behind a
physical e-stop, runtime watchdog, health gate, and reversible release.

## Open-source deployment model

The first version is a single-user local control center on the DGX Spark. It
uses local JSON, SQLite/filesystem artifacts, TensorBoard event files, and
self-contained HTML/SVG. A later multi-user deployment can add a self-hosted
OIDC provider (such as Keycloak) without changing experiment or policy
contracts. No hosted experiment tracker is required.

## Build order

Policy Bench is the foundation: immutable runs, evaluation, scores, stars,
promotion, and play are implemented first. Next come the user-facing skill
wizard, richer RL curve dashboards, failure-to-regression capture, and a
self-hosted login/roles layer. The low-level simulator and deployment
contracts remain stable underneath that UX.
