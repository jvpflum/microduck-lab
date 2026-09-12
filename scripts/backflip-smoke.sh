#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-RollerBackflip-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5 \
    --agent.run-name ducklab-v1-roller-backflip-smoke \
    2>&1 | tee "${REPORT_DIR}/backflip-smoke.log"
