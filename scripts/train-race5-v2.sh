#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_ENVS:-2048}"
iterations="${DUCKLAB_ITERATIONS:-150}"
seed="${DUCKLAB_SEED:-42}"
source_checkpoint="${DUCKLAB_RACE5_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/model_10.pt}"
warmstart_run="stage14_warmstart_v11_i10_lean_glide"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/${warmstart_run}/model_0.pt"

cd "${UPSTREAM_DIR}"
if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Race5 warm-start checkpoint not found: ${source_checkpoint}" >&2
    exit 1
fi
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 5.0e-6
mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "ducklab-race5-v14-lean-glide-2048-s${seed}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-race5-v14-lean-glide-2048-s${seed}.log"

"${LAB_ROOT}/scripts/finalize-training.sh" race5
