#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-Velocity-Flat-MicroDuck-Rollers \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5 \
    --agent.run-name ducklab-v1.1-skate-smoke \
    2>&1 | tee "${REPORT_DIR}/skate-smoke.log"
