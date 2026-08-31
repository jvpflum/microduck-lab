#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv

seed="${DUCKLAB_BOOST_SEED:-42}"
num_envs="${DUCKLAB_BOOST_ENVS:-4096}"
iterations="${DUCKLAB_BOOST_ITERATIONS:-4000}"
source_run="2026-08-31_09-06-07_ducklab-speed-retention-v3-straight-e4096-i6000-s42"
source_checkpoint="${DUCKLAB_BOOST_CHECKPOINT:-model_6159.pt}"
run_name="${DUCKLAB_BOOST_RUN_NAME:-ducklab-speed-retention-boost-e${num_envs}-i${iterations}-s${seed}}"
log_path="${LAB_ROOT}/reports/train-${run_name}.log"
warmstart_run="speed_retention_boost_i6159_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_retention_boost/${warmstart_run}/model_0.pt"

if [[ ! -f "${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_retention/${source_run}/${source_checkpoint}" ]]; then
    echo "Straight donor checkpoint not found: ${source_run}/${source_checkpoint}" >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_retention/${source_run}/${source_checkpoint}" \
    "${warmstart_checkpoint}" --learning-rate 5e-6
exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedRetentionBoost-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt >"${log_path}" 2>&1
