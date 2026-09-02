#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

cd "${UPSTREAM_DIR}"
exec "${UV_BIN}" run python "${LAB_ROOT}/tools/optimize_v68_joint_fusion.py" "$@"
