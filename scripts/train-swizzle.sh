#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_ENVS:-4096}"
iterations="${DUCKLAB_ITERATIONS:-8000}"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-Velocity-Swizzle-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name ducklab-v1.2-swizzle \
    2>&1 | tee "${REPORT_DIR}/train-swizzle.log"
