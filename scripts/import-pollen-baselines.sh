#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

# Factory policy comes from the pinned official runtime submodule.  The digest
# is checked independently so a changed upstream pointer is always explicit.
runtime="${LAB_ROOT}/upstream/microduck"
revision="$(git -C "${runtime}" rev-parse HEAD)"
expected_sha256="cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd"
run_dir="${LAB_ROOT}/baselines/pollen-microduck-simulator/${revision}/velocity_rollers/pollen-factory-roller"
policy="${run_dir}/BEST_roller.onnx"
source_policy="${runtime}/policies/roller.onnx"

mkdir -p "${run_dir}"
if [[ ! -f "${policy}" ]]; then
    cp "${source_policy}" "${policy}"
fi
actual_sha256="$(sha256sum "${policy}" | cut -d' ' -f1)"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "Pollen baseline checksum mismatch: ${actual_sha256}" >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/verify_policy.py" --roller "${policy}" \
    >"${REPORT_DIR}/pollen-factory-roller-verification.json"
bench_run="$("${LAB_ROOT}/scripts/policy-bench.sh" register "${run_dir}" --task roller | tail -n 1)"
"${LAB_ROOT}/scripts/policy-bench.sh" evaluate "${bench_run}"

echo "Imported pinned Pollen factory roller baseline: ${bench_run}"
