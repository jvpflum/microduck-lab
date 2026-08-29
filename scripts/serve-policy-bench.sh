#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
prepare_dirs

port="${DUCKLAB_BENCH_PORT:-8091}"
exec python3 "${LAB_ROOT}/tools/policy_bench_server.py" --port "${port}"
