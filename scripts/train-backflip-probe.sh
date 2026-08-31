#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

seed="${1:-11}"
env_count="${DUCKLAB_PROBE_ENVS:-2048}"
iterations="${DUCKLAB_PROBE_ITERATIONS:-300}"
eval_episodes="${DUCKLAB_PROBE_EVAL_EPISODES:-64}"
warmstart_run="${DUCKLAB_FRONTFLIP_WARMSTART_RUN:-warmstart-roller-hop-v1}"
warmstart_checkpoint="${DUCKLAB_FRONTFLIP_WARMSTART_CHECKPOINT:-model_1499.pt}"
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer." >&2
    exit 2
fi

started_at="$(date +%s)"
run_label="ducklab-v2-frontflip-probe-s${seed}"
log_path="${REPORT_DIR}/frontflip-probe-seed-${seed}.log"

cd "${UPSTREAM_DIR}"
train_command=("${UV_BIN}" run train Mjlab-RollerBackflip-Flat-MicroDuck
    --env.scene.num-envs "${env_count}" \
    --agent.max_iterations "${iterations}" \
    --agent.seed "${seed}" \
    --agent.run-name "${run_label}")
if [[ "${DUCKLAB_FRONTFLIP_FROM_SCRATCH:-0}" != "1" ]]; then
    warmstart_path="${UPSTREAM_DIR}/logs/rsl_rl/roller_backflip/${warmstart_run}/${warmstart_checkpoint}"
    if [[ ! -f "${warmstart_path}" ]]; then
        echo "Front-flip warm-start checkpoint does not exist: ${warmstart_path}" >&2
        exit 1
    fi
    echo "Warm-start: ${warmstart_path}"
    train_command+=(
        --agent.load-run "${warmstart_run}"
        --agent.load-checkpoint "${warmstart_checkpoint}"
        --agent.resume True
    )
fi
if [[ "${DUCKLAB_PROBE_QUIET:-0}" == "1" ]]; then
    echo "Training output: ${log_path}"
    "${train_command[@]}" >"${log_path}" 2>&1
else
    "${train_command[@]}" 2>&1 | tee "${log_path}"
fi

policy_path="$(find logs/rsl_rl/roller_backflip -type f -name "*${run_label}.onnx" \
    -newermt "@${started_at}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${policy_path}" || ! -f "${policy_path}" ]]; then
    echo "Probe did not produce an ONNX policy." >&2
    exit 1
fi

result_path="${REPORT_DIR}/frontflip-probe-seed-${seed}.json"
"${UV_BIN}" run "${LAB_ROOT}/tools/evaluate_frontflip.py" \
    "${policy_path}" --episodes "${eval_episodes}" --output "${result_path}"
echo "Probe result: ${result_path}"
