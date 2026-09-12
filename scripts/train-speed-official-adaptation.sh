#!/usr/bin/env bash
# Preserve the 5.41 mph donor while adapting only to official Race5 friction.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

seed="${DUCKLAB_OFFICIAL_SEED:-541}"
num_envs="${DUCKLAB_OFFICIAL_ENVS:-4096}"
iterations="${DUCKLAB_OFFICIAL_ITERATIONS:-4000}"
recipe="${DUCKLAB_OFFICIAL_RECIPE:-balanced}"
learning_rate="${DUCKLAB_OFFICIAL_LR:-2e-6}"
actor_std="${DUCKLAB_OFFICIAL_STD:-0.06}"
run_name="${DUCKLAB_OFFICIAL_RUN_NAME:-ducklab-speed-official-adaptation-e${num_envs}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKLAB_OFFICIAL_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_straightening/2026-08-31_05-18-27_microduck-speed-straightening-pilot1-e1024-i200-s42/model_160.pt}"
warmstart_run="speed_official_adaptation_i160_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_official_adaptation/${warmstart_run}/model_0.pt"
log_path="${REPORT_DIR}/train-${run_name}.log"

[[ -f "${source_checkpoint}" ]] || { echo "Speed-scout checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" --learning-rate "${learning_rate}" --actor-std "${actor_std}"
exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedOfficialAdaptation-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt 2>&1 | tee "${log_path}"
