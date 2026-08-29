#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

policy_path="${1:-}"
if [[ -z "${policy_path}" ]]; then
    policy_path="$(find "${UPSTREAM_DIR}/logs/rsl_rl/velocity_swizzle" \
        -type f -name '*.onnx' ! -path '*smoke*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${policy_path}" || ! -f "${policy_path}" ]]; then
    echo "No swizzle ONNX policy found; provide an explicit path." >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy_path}" \
    --output "${REPORT_DIR}/swizzle-evaluation.json"
