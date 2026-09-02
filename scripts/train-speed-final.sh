#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_FINAL_ENVS:-4096}"
iterations="${DUCKWING_FINAL_ITERATIONS:-5000}"
seed="${DUCKWING_FINAL_SEED:-2301}"
run_name="${DUCKWING_FINAL_RUN_NAME:-duckwing-v23-exact-v59-final-e${env_count}-i${iterations}-s${seed}}"
v59_onnx="${DUCKWING_FINAL_V59_ONNX:-${LAB_ROOT}/incoming/rtx5090/top5-2026-09-01/v59-i99-promoted-speed-leader/policy.onnx}"
scaffold="${DUCKWING_FINAL_SCAFFOLD:-${UPSTREAM_DIR}/protected_checkpoints/microduck_speed/official_friction_champion_model_6159.pt}"
exact_checkpoint="${UPSTREAM_DIR}/protected_checkpoints/duckwing_v23/v59-exact-import.pt"
warmstart_run="stage23_exact_v59_final_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_final/${warmstart_run}/model_0.pt"

[[ -f "${v59_onnx}" ]] || { echo "V59 ONNX donor missing: ${v59_onnx}" >&2; exit 1; }
[[ -f "${scaffold}" ]] || { echo "Checkpoint scaffold missing: ${scaffold}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${exact_checkpoint}")" "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/import_pollen_actor.py" \
    "${v59_onnx}" "${scaffold}" "${exact_checkpoint}" \
    --exploration-std 0.06
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${exact_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 7.5e-7 --actor-std 0.06

mark_training_start
"${UV_BIN}" run train Mjlab-SpeedFinal-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V23 complete. Compose selected checkpoints into V66 and run both Race5 and high-speed brake batteries before promotion."
