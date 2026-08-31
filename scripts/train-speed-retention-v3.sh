#!/usr/bin/env bash
# Long straight-line/control continuation from the current 3.10 mph speed leader.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv

seed="${DUCKLAB_RETENTION_SEED:-42}"
num_envs="${DUCKLAB_RETENTION_ENVS:-4096}"
iterations="${DUCKLAB_RETENTION_ITERATIONS:-6000}"
source_run="2026-08-31_05-40-47_microduck-speed-retention-pilot2-brake-line-e1024-i200-s42"
source_checkpoint="${DUCKLAB_RETENTION_CHECKPOINT:-model_160.pt}"
run_name="${DUCKLAB_RETENTION_RUN_NAME:-ducklab-speed-retention-v3-straight-e${num_envs}-i${iterations}-s${seed}}"
log_path="${LAB_ROOT}/reports/train-${run_name}.log"

if [[ ! -f "${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_retention/${source_run}/${source_checkpoint}" ]]; then
    echo "Speed-leader checkpoint not found: ${source_run}/${source_checkpoint}" >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedRetention-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${source_run}" \
    --agent.load-checkpoint "${source_checkpoint}" >"${log_path}" 2>&1
