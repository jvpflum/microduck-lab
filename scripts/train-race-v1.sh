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
iterations="${DUCKLAB_ITERATIONS:-100}"
seed="${DUCKLAB_SEED:-42}"
source_run_id="sprint-2026-08-30_23-22-10_ducklab-v3-sprint-probe-s42-i49"
source_dir="${LAB_ROOT}/policy-bench/runs/${source_run_id}/artifacts"
source_onnx="${source_dir}/sprint_model_49.onnx"
source_checkpoint="${source_dir}/model_49.pt"
warmstart_run="warmstart-sprint-v3-racer"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race/${warmstart_run}/model_0.pt"

if [[ ! -f "${source_onnx}" || ! -f "${source_checkpoint}" ]]; then
    echo "Qualified Sprint-v3 artifacts are missing from Policy Bench." >&2
    exit 1
fi

cd "${UPSTREAM_DIR}"
"${UV_BIN}" run python "${LAB_ROOT}/tools/import_pollen_actor.py" \
    "${source_onnx}" "${source_checkpoint}" "${warmstart_checkpoint}"
mark_training_start

"${UV_BIN}" run train Mjlab-Velocity-Race-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "ducklab-race-v1-s${seed}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-race-v1-s${seed}.log"

"${LAB_ROOT}/scripts/finalize-training.sh" race
