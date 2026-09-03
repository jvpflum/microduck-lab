#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs
"${LAB_ROOT}/scripts/preflight.sh"
activate_resource_profile
install_resource_profile_trap

env_count="${DUCKLAB_RACE5_V71_ENVS:-2048}"
iterations="${DUCKLAB_RACE5_V71_ITERATIONS:-600}"
seed="${DUCKLAB_RACE5_V71_SEED:-2711}"
run_name="${DUCKLAB_RACE5_V71_RUN_NAME:-ducklab-race5-v71-all-around-e${env_count}-i${iterations}-s${seed}}"
composite="${LAB_ROOT}/releases/v67/duckwing-v67-joint-specialist-fusion.onnx"
exact_checkpoint="${UPSTREAM_DIR}/protected_checkpoints/duckwing_v24/v65-high-exact-import.pt"
warmstart_run="stage71_v65_high_s${seed}"
run_root="${UPSTREAM_DIR}/logs/rsl_rl/velocity_race5_v71_all_around"
warmstart_checkpoint="${run_root}/${warmstart_run}/model_0.pt"
v67_high="${run_root}/v67-speed-high.onnx"
v67_control="${run_root}/v67-control.onnx"
v67_control_checkpoint="${run_root}/v67-control-teacher.pt"

[[ -f "${composite}" ]] || { echo "V67 composite missing: ${composite}" >&2; exit 1; }
[[ -f "${exact_checkpoint}" ]] || { echo "V65 exact checkpoint missing: ${exact_checkpoint}" >&2; exit 1; }

cd "${UPSTREAM_DIR}"
mkdir -p "${run_root}"

# V67 is a deployment router.  Extract the two standard actors it embeds so
# the teacher runs as Torch on the GPU and remains exactly reproducible.
"${UV_BIN}" run python "${LAB_ROOT}/tools/extract_composite_actor.py" \
    "${composite}" "${v67_high}" --prefix "drive_incumbent_speed_high/"
"${UV_BIN}" run python "${LAB_ROOT}/tools/extract_composite_actor.py" \
    "${composite}" "${v67_control}" --prefix "drive_incumbent_control_control_"
"${UV_BIN}" run python "${LAB_ROOT}/tools/import_pollen_actor.py" \
    "${v67_control}" "${exact_checkpoint}" "${v67_control_checkpoint}" \
    --exploration-std 0.015

mkdir -p "$(dirname "${warmstart_checkpoint}")"
"${UV_BIN}" run python "${LAB_ROOT}/tools/prepare_warmstart.py" \
    "${exact_checkpoint}" "${warmstart_checkpoint}" \
    --learning-rate 5.0e-7 --actor-std 0.015

mark_training_start
"${UV_BIN}" run train Mjlab-Velocity-Race5V71AllAround-MicroDuck \
    --env.scene.num-envs "${env_count}" \
    --agent.seed "${seed}" \
    --agent.max_iterations "${iterations}" \
    --agent.run-name "${run_name}" \
    --agent.resume True \
    --agent.load-run "${warmstart_run}" \
    --agent.load-checkpoint model_0.pt \
    2>&1 | tee "${REPORT_DIR}/train-${run_name}.log"

echo "V71 all-around training complete. Evaluate every command route against V67 before promotion." \
    | tee -a "${REPORT_DIR}/train-${run_name}.log"
