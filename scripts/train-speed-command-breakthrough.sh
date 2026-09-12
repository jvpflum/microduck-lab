#!/usr/bin/env bash
# Six-hour official-friction speed-discovery continuation from transfer i500.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv

seed="${DUCKLAB_SPEED_SEED:-42}"
num_envs="${DUCKLAB_SPEED_ENVS:-4096}"
iterations="${DUCKLAB_SPEED_ITERATIONS:-8000}"
run_name="${DUCKLAB_SPEED_RUN_NAME:-ducklab-speed-command-breakthrough-overnight-e${num_envs}-i${iterations}-s${seed}}"
log_path="${LAB_ROOT}/reports/train-${run_name}.log"
source_checkpoint="${DUCKLAB_SPEED_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_friction_transfer/2026-08-31_06-01-10_microduck-speed-friction-transfer-pilot2-calibrated-e1024-i600-s42/model_500.pt}"
warmstart_run="command_breakthrough_i500_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_command_breakthrough/${warmstart_run}/model_0.pt"

if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Command-breakthrough warm-start checkpoint not found: ${source_checkpoint}" >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 5e-6 --actor-std 0.08

exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedCommandBreakthrough-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt >"${log_path}" 2>&1
