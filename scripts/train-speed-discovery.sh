#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile

env_count="${DUCKLAB_ENVS:-4096}"
iterations="${DUCKLAB_ITERATIONS:-4000}"
seed="${DUCKLAB_SEED:-42}"
actor_std="${DUCKLAB_SPEED_ACTOR_STD:-0.15}"
source_checkpoint="${DUCKLAB_SPEED_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/model_10.pt}"
mini_batches="${DUCKLAB_MINI_BATCHES:-4}"
if (( env_count >= 8192 )) && [[ -z "${DUCKLAB_MINI_BATCHES:-}" ]]; then
    mini_batches=8
fi

if [[ ! -f "${source_checkpoint}" ]]; then
    echo "Speed-discovery warm-start checkpoint not found: ${source_checkpoint}" >&2
    echo "Copy V11 to this machine or set DUCKLAB_SPEED_WARMSTART_CHECKPOINT explicitly." >&2
    exit 1
fi

warmstart_run="speed_discovery_v1_warmstart_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_discovery/${warmstart_run}/model_0.pt"
run_name="microduck-speed-discovery-v1-e${env_count}-i${iterations}-s${seed}"
training_log="${REPORT_DIR}/${run_name}.log"
gpu_log="${REPORT_DIR}/${run_name}-gpu.jsonl"

mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 3.0e-5 \
    --actor-std "${actor_std}"

python3 "${LAB_ROOT}/tools/monitor_nvidia_gpu.py" \
    --output "${gpu_log}" --interval 15 --parent-pid "$$" &
gpu_monitor_pid=$!

stop_gpu_monitor() {
    if kill -0 "${gpu_monitor_pid}" 2>/dev/null; then
        kill "${gpu_monitor_pid}" 2>/dev/null || true
        wait "${gpu_monitor_pid}" 2>/dev/null || true
    fi
}

cleanup() {
    stop_gpu_monitor
    restore_resource_profile
}
trap 'training_status=$?; cleanup; exit "${training_status}"' EXIT

echo "Speed discovery batch: $((env_count * 24)) transitions/update, ${mini_batches} minibatches, $((env_count * 24 / mini_batches)) transitions/minibatch"
echo "GPU telemetry: ${gpu_log}"
mark_training_start
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run train Mjlab-SpeedDiscovery-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max-iterations "${iterations}" \
    --agent.algorithm.num-mini-batches "${mini_batches}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${training_log}"

# Stop telemetry before checkpoint export/evaluation so 4096-vs-8192 GPU
# utilization statistics describe training only.
stop_gpu_monitor

run_dir="$(find "${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_discovery" \
    -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "Could not locate completed speed-discovery run ${run_name}" >&2
    exit 1
fi

if [[ "${DUCKLAB_SELECT_BEST:-1}" == "1" ]]; then
    "${UV_BIN}" run python "${LAB_ROOT}/tools/select_speed_discovery_checkpoint.py" \
        "${run_dir}" \
        --episodes "${DUCKLAB_SELECTION_EPISODES:-5}" \
        --checkpoint-stride "${DUCKLAB_SELECTION_STRIDE:-1}" \
        --wheel-friction "${DUCKLAB_SELECTION_WHEEL_FRICTION:-0.003}"
fi

"${UV_BIN}" run python "${LAB_ROOT}/tools/summarize_speed_discovery_run.py" \
    --training-log "${training_log}" \
    --gpu-log "${gpu_log}" \
    --run-dir "${run_dir}" \
    --envs "${env_count}" \
    --rollout 24 \
    --output "${REPORT_DIR}/${run_name}-summary.json"

echo "Completed speed-discovery run: ${run_dir}"
