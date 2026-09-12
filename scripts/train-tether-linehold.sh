#!/usr/bin/env bash
# Phase two: qualify a selected tether champion under the strict official task.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

source_checkpoint="${DUCKLAB_TETHER_CHAMPION:?Set DUCKLAB_TETHER_CHAMPION to the selected tether .pt checkpoint}"
[[ -f "${source_checkpoint}" ]] || { echo "Tether checkpoint not found: ${source_checkpoint}" >&2; exit 1; }
exec env \
  DUCKLAB_OFFICIAL_WARMSTART_CHECKPOINT="${source_checkpoint}" \
  DUCKLAB_OFFICIAL_SEED="${DUCKLAB_TETHER_LINE_SEED:-808}" \
  DUCKLAB_OFFICIAL_ENVS="${DUCKLAB_TETHER_LINE_ENVS:-4096}" \
  DUCKLAB_OFFICIAL_ITERATIONS="${DUCKLAB_TETHER_LINE_ITERATIONS:-3000}" \
  DUCKLAB_OFFICIAL_RECIPE="line_hold" \
  DUCKLAB_OFFICIAL_LR="${DUCKLAB_TETHER_LINE_LR:-1e-6}" \
  DUCKLAB_OFFICIAL_STD="${DUCKLAB_TETHER_LINE_STD:-0.03}" \
  DUCKLAB_OFFICIAL_RUN_NAME="${DUCKLAB_TETHER_LINE_RUN_NAME:-ducklab-tether-linehold-e4096-i3000-s808}" \
  "${LAB_ROOT}/scripts/train-speed-official-adaptation.sh"
