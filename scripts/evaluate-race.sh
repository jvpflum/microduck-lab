#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

policy_path="${1:-}"
output_path="${2:-${REPORT_DIR}/race-evaluation.json}"
if [[ -z "${policy_path}" || ! -f "${policy_path}" ]]; then
    echo "Usage: $0 /absolute/path/to/policy.onnx [output.json]" >&2
    exit 2
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy_path}" \
    --profile race \
    --output "${output_path}"
