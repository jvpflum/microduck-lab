#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_V18_ENVS:-4096}"
iterations="${DUCKWING_V18_ITERATIONS:-3000}"
seed="${DUCKWING_V18_SEED:-1801}"
run_name="${DUCKWING_V18_RUN_NAME:-duckwing-v18-v57b-fusion-e${env_count}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKWING_V18_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/protected_checkpoints/duckwing_v18/v17-frontier-i250.pt}"
speed_teacher="${UPSTREAM_DIR}/protected_checkpoints/duckwing_v18/rtx5090-v57b-i50.onnx"
warmstart_run="stage18_warmstart_v57b_fusion_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5_fusion/${warmstart_run}/model_0.pt"

[[ -f "${source_checkpoint}" ]] || { echo "V18 warm-start checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
[[ -f "${speed_teacher}" ]] || { echo "V57b speed teacher not found: ${speed_teacher}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 1.0e-6

mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5Fusion-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V18 finished. Export selected checkpoints, wrap them with the control-aware champion, and evaluate the composed policy before promotion."
