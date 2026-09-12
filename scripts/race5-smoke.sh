#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"

source_checkpoint="${DUCKLAB_RACE5_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/2026-08-31_01-21-45_ducklab-race5-v5-s42/model_104.pt}"
warmstart_run="warmstart-race5-v6"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/${warmstart_run}/model_0.pt"

cd "${UPSTREAM_DIR}"
if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Race5 warm-start checkpoint not found: ${source_checkpoint}" >&2
    exit 1
fi
mkdir -p "$(dirname "${warmstart_checkpoint}")"
cp -a "${source_checkpoint}" "${warmstart_checkpoint}"
"${UV_BIN}" run train Mjlab-Velocity-Race5-MicroDuck \
    --env.scene.num-envs 128 \
    --agent.seed 42 \
    --agent.max_iterations 30 \
    --agent.run-name ducklab-race5-v6-graded-smoke-s42 \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/race5-v6-graded-smoke.log"
