#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_RACE5_FRONTIER_ENVS:-4096}"
iterations="${DUCKLAB_RACE5_FRONTIER_ITERATIONS:-4000}"
seed="${DUCKLAB_RACE5_FRONTIER_SEED:-1701}"
run_name="${DUCKLAB_RACE5_FRONTIER_RUN_NAME:-ducklab-race5-v17-controlaware-frontier-e${env_count}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKLAB_RACE5_FRONTIER_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/model_10.pt}"
speed_teacher="${UPSTREAM_DIR}/protected_checkpoints/microduck_5090_transfer/v47-speed-specialist/policy.onnx"
warmstart_run="stage17_warmstart_controlaware_frontier_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/${warmstart_run}/model_0.pt"

[[ -f "${source_checkpoint}" ]] || { echo "Control teacher checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
[[ -f "${speed_teacher}" ]] || { echo "V47 speed teacher not found: ${speed_teacher}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 1.0e-6

mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5Frontier-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

if [[ "${DUCKLAB_RACE5_FRONTIER_FINALIZE:-1}" == "1" ]]; then
    "${LAB_ROOT}/scripts/finalize-training.sh" race5
fi
