#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_ENVS:-2048}"
iterations="${DUCKLAB_ITERATIONS:-50}"
seed="${DUCKLAB_SEED:-42}"
warmstart_run="warmstart-pollen-factory-roller-v3"
factory_onnx="${LAB_ROOT}/baselines/pollen-microduck-simulator/590b986bd8c0d50ae02cb3ea2f59c463b6828168/velocity_rollers/pollen-factory-roller/BEST_roller.onnx"
warmstart_dir="${UPSTREAM_DIR}/logs/rsl_rl/velocity_sprint/${warmstart_run}"
warmstart_checkpoint="model_0.pt"

if [[ ! -f "${factory_onnx}" ]]; then
    echo "Official Pollen factory policy is missing: ${factory_onnx}" >&2
    exit 1
fi
template_checkpoint="$(find "${UPSTREAM_DIR}/logs/rsl_rl/velocity_sprint" \
    -type f -path '*sprint-smoke*/model_4.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "${template_checkpoint}" || ! -f "${template_checkpoint}" ]]; then
    echo "No Sprint smoke checkpoint found; run ./scripts/sprint-smoke.sh first." >&2
    exit 1
fi
mkdir -p "${warmstart_dir}"
cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/import_pollen_actor.py" \
    "${factory_onnx}" "${template_checkpoint}" "${warmstart_dir}/${warmstart_checkpoint}"
mark_training_start

"${UV_BIN}" run train Mjlab-Velocity-Sprint-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name ducklab-v3-sprint-probe-s${seed} \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint "${warmstart_checkpoint}" \
    2>&1 | tee "${REPORT_DIR}/train-sprint-probe.log"
