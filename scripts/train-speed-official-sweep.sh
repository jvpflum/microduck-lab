#!/usr/bin/env bash
# Sequential, evidence-first sweep for the 5.41 mph skate donor.
#
# Spark runs exactly ONE training job at a time.  Every candidate gets the
# same short official-friction certification; then the generated scoreboard
# identifies the Pareto set instead of chasing one opaque PPO reward.
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

num_envs="${DUCKLAB_SWEEP_ENVS:-4096}"
iterations="${DUCKLAB_SWEEP_ITERATIONS:-350}"
stamp="${DUCKLAB_SWEEP_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
sweep_dir="${REPORT_DIR}/official-speed-sweep-${stamp}"
mkdir -p "${sweep_dir}"

# recipe seed learning-rate actor-std
variants=(
  "balanced 601 2e-6 0.06"
  "speed_retention 602 1e-6 0.06"
  "speed_retention 603 2e-6 0.03"
  "line_hold 604 2e-6 0.06"
  "balanced 605 5e-6 0.10"
  "line_hold 606 1e-6 0.03"
)

printf 'Official speed sweep: %s variants, %s envs, %s PPO iterations each\n' "${#variants[@]}" "${num_envs}" "${iterations}"
printf 'Results: %s\n' "${sweep_dir}"

for variant in "${variants[@]}"; do
    read -r recipe seed learning_rate actor_std <<<"${variant}"
    run_name="official-sweep-${recipe}-lr${learning_rate}-std${actor_std}-s${seed}-e${num_envs}-i${iterations}"
    printf '\n=== %s ===\n' "${run_name}"
    DUCKLAB_OFFICIAL_RECIPE="${recipe}" \
    DUCKLAB_OFFICIAL_SEED="${seed}" \
    DUCKLAB_OFFICIAL_ENVS="${num_envs}" \
    DUCKLAB_OFFICIAL_ITERATIONS="${iterations}" \
    DUCKLAB_OFFICIAL_LR="${learning_rate}" \
    DUCKLAB_OFFICIAL_STD="${actor_std}" \
    DUCKLAB_OFFICIAL_RUN_NAME="${run_name}" \
      "${LAB_ROOT}/scripts/train-speed-official-adaptation.sh"

    # rsl_rl prefixes its run directory with a timestamp. Resolve it only
    # after the trainer exits so the evaluator cannot accidentally inspect a
    # same-named run from an earlier sweep.
    mapfile -t run_matches < <(find "${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_official_adaptation" \
      -maxdepth 1 -mindepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' | sort -n)
    (( ${#run_matches[@]} > 0 )) || { echo "Could not locate completed run ${run_name}" >&2; exit 1; }
    run_dir="${run_matches[-1]#* }"

    # Fast screening: evaluate six evenly-spaced saves, with the exact official
    # friction and the same controller users get from the dashboard.
    (
      cd "${UPSTREAM_DIR}"
      "${UV_BIN}" run python "${LAB_ROOT}/tools/select_speed_discovery_checkpoint.py" \
        "${run_dir}" \
        --task Mjlab-SpeedOfficialAdaptation-Flat-MicroDuck-Rollers \
        --episodes 3 --duration 12 --checkpoint-stride 5 \
        --wheel-friction 0.003 --race-line-control \
        --yaw-kp 0.80 --lateral-kp 0.16 --yaw-kd 0.10 --max-correction 0.22 \
        --rank-world-x --min-survival 0.80 \
        --max-lateral-deviation-m 1.0 --max-mean-heading-deg 12.0
    )
    mkdir -p "${sweep_dir}/${run_name}"
    cp -a "${run_dir}/best_speed_discovery.json" "${sweep_dir}/${run_name}/"
done

python3 "${LAB_ROOT}/tools/summarize_speed_official_sweep.py" "${sweep_dir}"
printf 'Sweep complete. Inspect: %s/official_sweep_scoreboard.json\n' "${sweep_dir}"
