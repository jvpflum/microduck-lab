#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_V19_ENVS:-4096}"
iterations="${DUCKWING_V19_ITERATIONS:-1000}"
seed="${DUCKWING_V19_SEED:-1901}"
run_name="${DUCKWING_V19_RUN_NAME:-duckwing-v19-v59-refinement-e${env_count}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKWING_V19_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/protected_checkpoints/duckwing_v18/v17-frontier-i250.pt}"
warmstart_run="stage19_warmstart_v59_refinement_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5_v59_refinement/${warmstart_run}/model_0.pt"

[[ -f "${source_checkpoint}" ]] || { echo "V19 warm-start checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
[[ -f "${UPSTREAM_DIR}/protected_checkpoints/duckwing_v18/rtx5090-v57b-i50.onnx" ]] || { echo "V19 base speed teacher assets missing" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" --learning-rate 5.0e-7

mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5V59Refinement-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V19 complete. Evaluate saved checkpoints only after composing each candidate into V66's control shell."
