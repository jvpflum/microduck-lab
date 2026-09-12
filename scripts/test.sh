#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run --with pytest pytest tests/ | tee "${REPORT_DIR}/tests.log"
