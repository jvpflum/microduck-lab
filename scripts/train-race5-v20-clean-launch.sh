#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_V20_ENVS:-4096}"
iterations="${DUCKWING_V20_ITERATIONS:-800}"
seed="${DUCKWING_V20_SEED:-2001}"
run_name="${DUCKWING_V20_RUN_NAME:-duckwing-v20-clean-launch-e${env_count}-i${iterations}-s${seed}}"
source_checkpoint="${DUCKWING_V20_WARMSTART_CHECKPOINT:-${UPSTREAM_DIR}/protected_checkpoints/duckwing_v18/v17-frontier-i250.pt}"
warmstart_run="stage20_warmstart_clean_launch_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5_clean_launch/${warmstart_run}/model_0.pt"

[[ -f "${source_checkpoint}" ]] || { echo "V20 warm-start checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
[[ -f "${LAB_ROOT}/incoming/rtx5090/top5-2026-09-01/v59-i99-promoted-speed-leader/policy.onnx" ]] || { echo "V20 V59 speed teacher missing" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${source_checkpoint}" "${warmstart_checkpoint}" --learning-rate 1.0e-6

mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5CleanLaunch-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V20 complete. Benchmark composed checkpoints against V66 before promotion."
