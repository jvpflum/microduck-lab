#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${LAB_ROOT}/upstream/microduck_rl"
UV_BIN="${LAB_ROOT}/.tools/uv/bin/uv"
ARTIFACT_DIR="${LAB_ROOT}/artifacts"
REPORT_DIR="${LAB_ROOT}/reports"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# Policy Bench is the system of record. Keep upstream W&B compatibility hooks
# disabled so training never requires or contacts a proprietary service.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

require_checkout() {
    if [[ ! -f "${UPSTREAM_DIR}/pyproject.toml" ]]; then
        echo "MicroDuck upstream checkout is missing. Run: git submodule update --init --recursive" >&2
        exit 1
    fi
}

require_uv() {
    if [[ ! -x "${UV_BIN}" ]]; then
        echo "uv is missing. Run ./scripts/bootstrap.sh first." >&2
        exit 1
    fi
}

prepare_dirs() {
    mkdir -p "${ARTIFACT_DIR}" "${REPORT_DIR}"
}

DUCKLAB_RESOURCE_ACTIVE=false

activate_resource_profile() {
    local profile="${DUCKLAB_RESOURCE_PROFILE:-shared}"
    case "${profile}" in
        shared)
            echo "Resource profile: shared (vLLM stays online)."
            ;;
        training-priority)
            "${LAB_ROOT}/scripts/resource-profile.sh" enter "$$"
            DUCKLAB_RESOURCE_ACTIVE=true
            ;;
        *)
            echo "Unknown DUCKLAB_RESOURCE_PROFILE: ${profile}" >&2
            return 2
            ;;
    esac
}

restore_resource_profile() {
    if [[ "${DUCKLAB_RESOURCE_ACTIVE}" == "true" ]]; then
        "${LAB_ROOT}/scripts/resource-profile.sh" restore "$$" || true
        DUCKLAB_RESOURCE_ACTIVE=false
    fi
}

install_resource_profile_trap() {
    trap 'training_status=$?; restore_resource_profile; exit "${training_status}"' EXIT
}
