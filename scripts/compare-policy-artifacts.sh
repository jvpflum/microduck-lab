#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CHECKPOINT.pt POLICY.onnx [OUTPUT.json]" >&2
    exit 2
fi

checkpoint="$1"
policy="$2"
output="${3:-${REPORT_DIR}/policy-parity.json}"

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/compare_policy_artifacts.py" \
    "${checkpoint}" "${policy}" --output "${output}"
