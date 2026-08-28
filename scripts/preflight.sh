#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
swap_total_kib="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_kib="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"

if (( available_kib < 20 * 1024 * 1024 )); then
    echo "Refusing GPU work: less than 20 GiB unified memory is available." >&2
    exit 1
fi

if (( swap_total_kib > 0 && (swap_total_kib - swap_free_kib) * 100 / swap_total_kib > 50 )); then
    echo "Refusing GPU work: swap usage is over 50%." >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/preflight.py" | tee "${REPORT_DIR}/preflight.json"
