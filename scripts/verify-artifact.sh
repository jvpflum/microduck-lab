#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

policy_path="${1:-}"
if [[ -z "${policy_path}" ]]; then
    policy_path="$(find "${UPSTREAM_DIR}/logs" -type f -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${policy_path}" || ! -f "${policy_path}" ]]; then
    echo "No ONNX policy found. Run ./scripts/smoke.sh first or provide a path." >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/verify_policy.py" "${policy_path}" \
    | tee "${REPORT_DIR}/policy-verification.json"
