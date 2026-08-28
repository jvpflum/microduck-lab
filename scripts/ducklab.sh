#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
cd "${UPSTREAM_DIR}"
exec "${UV_BIN}" run "$@"
