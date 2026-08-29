#!/usr/bin/env bash
set -u

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 TRAINING_PID" >&2
    exit 2
fi

training_pid="$1"
lab_root="$(cd "$(dirname "$0")/.." && pwd)"
upstream="${lab_root}/upstream/microduck_rl"

while kill -0 "${training_pid}" 2>/dev/null; do
    sleep 30
done

verification_status=0
policy_path="$(find "${upstream}/logs/rsl_rl/velocity_swizzle" -type f -name '*.onnx' ! -path '*smoke*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -n "${policy_path}" && -f "${policy_path}" ]]; then
    cd "${upstream}"
    "${lab_root}/.tools/uv/bin/uv" run python "${lab_root}/tools/verify_policy.py" \
        --roller "${policy_path}" \
        > "${lab_root}/reports/swizzle-policy-verification.json" \
        || verification_status=$?
    if [[ "${verification_status}" -eq 0 ]]; then
        "${lab_root}/scripts/policy-bench.sh" discover --task swizzle
        bench_run="$("${lab_root}/scripts/policy-bench.sh" list --task swizzle --latest | cut -f1)"
        if [[ -n "${bench_run}" ]]; then
            "${lab_root}/scripts/policy-bench.sh" evaluate "${bench_run}" \
                || verification_status=$?
            "${lab_root}/scripts/policy-bench.sh" metrics "${bench_run}" \
                || verification_status=$?
        fi
    fi
else
    echo "No final swizzle ONNX artifact found." >&2
    verification_status=1
fi

sudo docker update --restart=unless-stopped qwen38-hermes-vllm
sudo docker start qwen38-hermes-vllm
systemctl --user start reachy-local-backend.service
/home/ducklab-user/.local/bin/hermes cron resume 82eef8b57ed7

exit "${verification_status}"
