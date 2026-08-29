#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
prepare_dirs

port="${DUCKLAB_BENCH_PORT:-8091}"
arena_port="${DUCKLAB_ARENA_PORT:-8070}"
arena_dist="${LAB_ROOT}/upstream/microduck-simulator/app/dist"

if [[ ! -f "${arena_dist}/index.html" ]]; then
    echo "Pollen factory playground is not built. Run: ./scripts/build-pollen-arena.sh" >&2
    exit 1
fi

python3 -m http.server "${arena_port}" --bind 127.0.0.1 --directory "${arena_dist}" \
    >"${REPORT_DIR}/pollen-arena.log" 2>&1 &
arena_pid=$!
dashboard_pid=""

sleep 0.2
if ! kill -0 "${arena_pid}" 2>/dev/null; then
    wait "${arena_pid}" || true
    echo "Pollen factory playground failed to start on port ${arena_port}:" >&2
    tail -n 20 "${REPORT_DIR}/pollen-arena.log" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${dashboard_pid}" ]]; then
        kill "${dashboard_pid}" 2>/dev/null || true
    fi
    kill "${arena_pid}" 2>/dev/null || true
    wait "${arena_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

python3 "${LAB_ROOT}/tools/policy_bench_server.py" --port "${port}" &
dashboard_pid=$!
wait "${dashboard_pid}"
