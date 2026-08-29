#!/usr/bin/env bash
set -euo pipefail

lab_root="/home/ducklab-user/projects/microduck-lab"
upstream="${lab_root}/upstream/microduck_rl"
uv_bin="${lab_root}/.tools/uv/bin/uv"
checkpoint="${upstream}/logs/rsl_rl/velocity_rollers/2026-08-28_20-50-40_ducklab-v1.1-skate/model_4999.pt"
log_file="${lab_root}/reports/view-final-skate.log"

cd "${upstream}"
"${uv_bin}" run "${lab_root}/tools/play_viser_compat.py" Mjlab-Velocity-Flat-MicroDuck-Rollers \
  --checkpoint-file "${checkpoint}" \
  --num-envs 1 \
  --device cpu \
  --viewer viser 2>&1 | tee "${log_file}"

status="${PIPESTATUS[0]}"
echo "Viewer stopped with exit code ${status}. Log: ${log_file}"
exit "${status}"
