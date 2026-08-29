#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {swizzle|roller|walking}" >&2
    exit 2
fi

task="$1"
lab_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="${lab_root}/upstream/microduck_rl"
uv="${lab_root}/.tools/uv/bin/uv"

case "${task}" in
    swizzle)
        experiment="velocity_swizzle"
        bench_task="swizzle"
        roller_flag=(--roller)
        auto_score=true
        ;;
    roller)
        experiment="velocity_rollers"
        bench_task="roller"
        roller_flag=(--roller)
        auto_score=true
        ;;
    walking)
        experiment="velocity"
        bench_task="walking"
        roller_flag=()
        auto_score=false
        ;;
    *)
        echo "Unknown training task: ${task}" >&2
        exit 2
        ;;
esac

policy_path="$(find "${upstream}/logs/rsl_rl/${experiment}" -type f -name '*.onnx' \
    ! -path '*smoke*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${policy_path}" || ! -f "${policy_path}" ]]; then
    echo "No final ${task} ONNX artifact found." >&2
    exit 1
fi
if [[ -n "${DUCKLAB_TRAINING_STARTED_AT:-}" ]]; then
    if [[ ! "${DUCKLAB_TRAINING_STARTED_AT}" =~ ^[0-9]+$ ]]; then
        echo "DUCKLAB_TRAINING_STARTED_AT must be an epoch timestamp." >&2
        exit 2
    fi
    if (( $(stat -c %Y "${policy_path}") < DUCKLAB_TRAINING_STARTED_AT )); then
        echo "The trainer did not produce a new ONNX artifact; refusing to score an older run." >&2
        exit 1
    fi
fi

echo "Finalizing ${task} policy: ${policy_path}"
cd "${upstream}"
"${uv}" run python "${lab_root}/tools/verify_policy.py" \
    "${roller_flag[@]}" "${policy_path}" \
    >"${lab_root}/reports/${task}-policy-verification.json"

"${lab_root}/scripts/policy-bench.sh" discover --task "${bench_task}"
bench_run="$("${lab_root}/scripts/policy-bench.sh" list --task "${bench_task}" --latest | cut -f1)"
if [[ -z "${bench_run}" ]]; then
    echo "Policy Bench did not register the completed ${task} run." >&2
    exit 1
fi

if [[ "${auto_score}" == "true" ]]; then
    "${lab_root}/scripts/policy-bench.sh" evaluate "${bench_run}"
fi
"${lab_root}/scripts/policy-bench.sh" metrics "${bench_run}"
echo "Finalized Policy Bench run: ${bench_run}"
