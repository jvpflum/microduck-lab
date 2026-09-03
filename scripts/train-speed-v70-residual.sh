#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_V70_RESIDUAL_ENVS:-2048}"
iterations="${DUCKWING_V70_RESIDUAL_ITERATIONS:-800}"
seed="${DUCKWING_V70_RESIDUAL_SEED:-2701}"
run_name="${DUCKWING_V70_RESIDUAL_RUN_NAME:-duckwing-v70-residual-e${env_count}-i${iterations}-s${seed}}"
exact_checkpoint="${UPSTREAM_DIR}/protected_checkpoints/duckwing_v24/v65-high-exact-import.pt"
warmstart_run="stage70_v65_high_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_v70_residual/${warmstart_run}/model_0.pt"

[[ -f "${exact_checkpoint}" ]] || {
    echo "V65 exact checkpoint missing: ${exact_checkpoint}" >&2
    exit 1
}
[[ -f "${LAB_ROOT}/incoming/rtx5090/v47-official-friction-speed-specialist/policy.onnx" ]] || {
    echo "V47 speed teacher missing" >&2
    exit 1
}

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${exact_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 1.0e-7 --actor-std 0.012

mark_training_start
"${UV_BIN}" run train Mjlab-SpeedV70Residual-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V70 residual training complete. Export checkpoints, compose with V67, and require strict Race5/retention gates before promotion." \
    | tee -a "${REPORT_DIR}/train-${run_name}.log"
