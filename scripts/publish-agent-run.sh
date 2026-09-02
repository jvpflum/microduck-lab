#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_uv

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/agent-run-receipt.json" >&2
    exit 2
fi
cd "${UPSTREAM_DIR}"
exec "${UV_BIN}" run python "${LAB_ROOT}/tools/agent_runs.py" "$1"
