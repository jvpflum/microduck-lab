#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
prepare_dirs

port="${DUCKLAB_BENCH_PORT:-8091}"
"${LAB_ROOT}/scripts/policy-bench.sh" dashboard >/dev/null
echo "Policy Bench: http://127.0.0.1:${port}"
cd "${LAB_ROOT}/policy-bench"
exec python3 -m http.server "${port}" --bind 127.0.0.1
