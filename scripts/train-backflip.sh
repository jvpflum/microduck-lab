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
iterations="${DUCKLAB_ITERATIONS:-2500}"
mark_training_start

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-RollerBackflip-Flat-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name ducklab-v1-roller-backflip \
    2>&1 | tee "${REPORT_DIR}/train-backflip.log"

"${LAB_ROOT}/scripts/finalize-training.sh" backflip
