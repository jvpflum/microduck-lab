#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv

policy="${1:?usage: evaluate-race5.sh POLICY.onnx [OUTPUT.json]}"
output="${2:-${REPORT_DIR}/race5-evaluation.json}"
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy}" --profile race-5mph --line-hold --output "${output}"
