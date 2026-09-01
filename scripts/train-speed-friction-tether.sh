#!/usr/bin/env bash
# Long, low-plasticity preservation run: 5.4 mph scout -> official drag.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

seed="${DUCKLAB_TETHER_SEED:-707}"
num_envs="${DUCKLAB_TETHER_ENVS:-4096}"
iterations="${DUCKLAB_TETHER_ITERATIONS:-12000}"
run_name="${DUCKLAB_TETHER_RUN_NAME:-ducklab-speed-friction-tether-e${num_envs}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKLAB_TETHER_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_straightening/2026-08-31_05-18-27_microduck-speed-straightening-pilot1-e1024-i200-s42/model_160.pt}"
warmstart_run="speed_friction_tether_i160_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_friction_tether/${warmstart_run}/model_0.pt"
log_path="${REPORT_DIR}/train-${run_name}.log"

[[ -f "${source_checkpoint}" ]] || { echo "Speed-scout checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" --learning-rate 7.5e-7 --actor-std 0.03
exec env WANDB_MODE=disabled "${UV_BIN}" run train \
    Mjlab-SpeedFrictionTether-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${num_envs}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt 2>&1 | tee "${log_path}"
