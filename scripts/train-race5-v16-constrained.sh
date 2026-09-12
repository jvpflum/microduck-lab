#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_RACE5_CONSTRAINED_ENVS:-4096}"
iterations="${DUCKLAB_RACE5_CONSTRAINED_ITERATIONS:-3000}"
seed="${DUCKLAB_RACE5_CONSTRAINED_SEED:-1616}"
run_name="${DUCKLAB_RACE5_CONSTRAINED_RUN_NAME:-ducklab-race5-v16-v11-constrained-e${env_count}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKLAB_RACE5_CONSTRAINED_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/model_10.pt}"
warmstart_run="stage16_warmstart_v11_constrained"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5_constrained/${warmstart_run}/model_0.pt"

cd "${UPSTREAM_DIR}"
if [[ ! -f "${source_checkpoint}" ]]; then
    echo "V11 warm-start checkpoint not found: ${source_checkpoint}" >&2
    exit 1
fi
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 2.0e-6
mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5Constrained-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

"${LAB_ROOT}/scripts/finalize-training.sh" race5
