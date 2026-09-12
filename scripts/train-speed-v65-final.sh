#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKWING_V65_FINAL_ENVS:-4096}"
iterations="${DUCKWING_V65_FINAL_ITERATIONS:-1200}"
seed="${DUCKWING_V65_FINAL_SEED:-2401}"
run_name="${DUCKWING_V65_FINAL_RUN_NAME:-duckwing-v24-exact-v65-head-e${env_count}-i${iterations}-s${seed}}"
v65_composite="${DUCKWING_V65_POLICY:-${LAB_ROOT}/incoming/rtx5090/v65-v63-immediate-switch-2026-09-01/policy.onnx}"
scaffold="${DUCKWING_V65_SCAFFOLD:-${UPSTREAM_DIR}/protected_checkpoints/microduck_speed/official_friction_champion_model_6159.pt}"
exact_checkpoint="${UPSTREAM_DIR}/protected_checkpoints/duckwing_v24/v65-high-exact-import.pt"
warmstart_run="stage24_exact_v65_high_s${seed}"
warmstart_checkpoint="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_v65_final/${warmstart_run}/model_0.pt"

[[ -f "${v65_composite}" ]] || { echo "V65 composite missing: ${v65_composite}" >&2; exit 1; }
[[ -f "${scaffold}" ]] || { echo "Checkpoint scaffold missing: ${scaffold}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "$(dirname "${exact_checkpoint}")" "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/import_pollen_actor.py" \
    "${v65_composite}" "${scaffold}" "${exact_checkpoint}" \
    --initializer-prefix high/ \
    --verify-command-x 0.80 \
    --verify-command-yaw 0.0 \
    --exploration-std 0.015
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${exact_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 2.0e-8 --actor-std 0.015

mark_training_start
"${UV_BIN}" run train Mjlab-SpeedV65Final-Flat-MicroDuck-Rollers \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V24 complete. Export every 100 iterations, wrap in V66, and require Race5 plus high-speed-brake promotion gates." \
    | tee -a "${REPORT_DIR}/train-${run_name}.log"
