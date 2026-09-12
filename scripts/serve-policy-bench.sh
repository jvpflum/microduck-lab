#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
prepare_dirs

port="${DUCKLAB_BENCH_PORT:-8091}"
arena_dist="${LAB_ROOT}/upstream/microduck-simulator/app/dist"

if [[ ! -f "${arena_dist}/index.html" ]]; then
    echo "Pollen factory playground is not built. Run: ./scripts/build-pollen-arena.sh" >&2
    exit 1
fi

exec python3 "${LAB_ROOT}/tools/policy_bench_server.py" --port "${port}"
