#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"

env_count="${DUCKLAB_ENVS:-4096}"
iterations="${DUCKLAB_ITERATIONS:-4000}"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name ducklab-v1-walk \
    2>&1 | tee "${REPORT_DIR}/train-baseline.log"
