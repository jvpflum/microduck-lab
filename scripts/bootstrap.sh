#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout

git -C "${LAB_ROOT}" submodule update --init --recursive

if [[ ! -x "${UV_BIN}" ]]; then
    python3 -m venv "${LAB_ROOT}/.tools/uv"
    "${LAB_ROOT}/.tools/uv/bin/pip" install uv
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" sync --frozen
"${LAB_ROOT}/scripts/build-pollen-arena.sh"
echo "Bootstrap complete. Run: make preflight"
