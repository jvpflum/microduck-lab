#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-Velocity-Race5-MicroDuck \
    --env.scene.num-envs 128 \
    --agent.seed 42 \
    --agent.max_iterations 30 \
    --agent.run-name ducklab-race5-from-scratch-smoke-s42 \
    2>&1 | tee "${REPORT_DIR}/race5-v6-graded-smoke.log"
