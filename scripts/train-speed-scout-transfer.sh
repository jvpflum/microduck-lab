#!/usr/bin/env bash
# Direct autonomous-line-hold fine-tune of the preserved 5.41 mph speed scout.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

seed="${DUCKLAB_SCOUT_SEED:-541}"
num_envs="${DUCKLAB_SCOUT_ENVS:-4096}"
iterations="${DUCKLAB_SCOUT_ITERATIONS:-4000}"
run_name="${DUCKLAB_SCOUT_RUN_NAME:-ducklab-speed-scout-transfer-e${num_envs}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKLAB_SCOUT_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_straightening/2026-08-31_05-18-27_microduck-speed-straightening-pilot1-e1024-i200-s42/model_160.pt}"
warmstart_run="speed_scout_transfer_i160_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_scout_transfer/${warmstart_run}/model_0.pt"
log_path="${REPORT_DIR}/train-${run_name}.log"

[[ -f "${source_checkpoint}" ]] || { echo "Speed-scout checkpoint not found: ${source_checkpoint}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" --learning-rate 5e-6 --actor-std 0.08
exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedScoutTransfer-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt 2>&1 | tee "${log_path}"
