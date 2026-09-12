#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

run_name="${DUCKWING_FINAL_RUN_NAME:-duckwing-v24-exact-v65-head-e4096-i1200-s2401}"
train_log="${REPORT_DIR}/train-${run_name}.log"
run_root="${UPSTREAM_DIR}/logs/rsl_rl/microduck_speed_v65_final"
result_dir="${REPORT_DIR}/duckwing-v24-final-screen"
baseline_policy="${LAB_ROOT}/releases/v66/duckwing-v66-v65-control-fusion.onnx"
control_policy="${LAB_ROOT}/policy-bench/runs/race5-microduck_hybrid_controlaware-export/artifacts/hybrid_v11_i6159_smooth_t02_t12_b100.onnx"
brake_policy="${LAB_ROOT}/incoming/rtx5090/v65-v63-immediate-switch-2026-09-01/policy.onnx"
task="Mjlab-SpeedV65Final-Flat-MicroDuck-Rollers"
done_marker="${result_dir}/.complete"
lock_dir="${result_dir}/.evaluation-lock"

# Cron calls this while training is live. Stay silent until evaluation can run.
if pgrep -af "${run_name}" >/dev/null; then
    exit 0
fi
if [[ ! -f "${train_log}" ]] || ! rg -q 'V24 complete' "${train_log}"; then
    printf 'DuckWing V24 is not running and has no completion marker: %s\n' "${train_log}" >&2
    exit 1
fi
if [[ -f "${done_marker}" ]]; then
    exit 0
fi

[[ -f "${baseline_policy}" ]] || { echo "V66 baseline policy missing: ${baseline_policy}" >&2; exit 1; }
[[ -f "${control_policy}" ]] || { echo "V66 inner control policy missing: ${control_policy}" >&2; exit 1; }
[[ -f "${brake_policy}" ]] || { echo "V65 brake policy missing: ${brake_policy}" >&2; exit 1; }
run_dir="$(find "${run_root}" -maxdepth 1 -type d -name "*${run_name}" -print | sort | tail -n 1)"
[[ -n "${run_dir}" ]] || { echo "Completed V24 run directory not found" >&2; exit 1; }

# Exporting each checkpoint creates a simulator and policy instance. Require a
# comfortable post-training memory/disk margin before starting the batch.
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
disk_available_kib="$(df --output=avail -k "${REPORT_DIR}" | tail -n 1 | tr -d ' ')"
if (( available_kib < 16777216 )); then
    echo "Deferring V24 evaluation: less than 16 GiB memory available" >&2
    exit 1
fi
if (( disk_available_kib < 10485760 )); then
    echo "Deferring V24 evaluation: less than 10 GiB disk available" >&2
    exit 1
fi

mkdir -p "${result_dir}/actors" "${result_dir}/hybrids" "${result_dir}/evaluations" "${result_dir}/logs"
if ! mkdir "${lock_dir}" 2>/dev/null; then
    exit 0
fi
cleanup_lock() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup_lock EXIT

evaluate_policy() {
    local policy="$1"
    local output="$2"
    local log="$3"
    local profile="${4:-race-5mph}"
    cd "${UPSTREAM_DIR}"
    "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
        "${policy}" \
        --profile "${profile}" \
        --line-hold \
        --line-yaw-kp 0.70 \
        --line-lateral-kp 0.14 \
        --line-yaw-kd 0.07 \
        --line-max-wz 0.15 \
        --output "${output}" >"${log}" 2>&1
}

evaluate_idle_steer() {
    local policy="$1"
    local output="$2"
    local log="$3"
    cd "${UPSTREAM_DIR}"
    "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
        "${policy}" \
        --profile idle-launch \
        --launch-yaw-command -0.30 \
        --launch-yaw-pulse-duration 0.10 \
        --launch-yaw-pulse-count 4 \
        --launch-yaw-pulse-gap 0.10 \
        --output "${output}" >"${log}" 2>&1
}

baseline_eval="${result_dir}/evaluations/v66-baseline.json"
baseline_brake_eval="${result_dir}/evaluations/v66-baseline-drive-retention.json"
baseline_idle_eval="${result_dir}/evaluations/v66-baseline-idle-steer.json"
if [[ ! -f "${baseline_eval}" ]]; then
    evaluate_policy \
        "${baseline_policy}" \
        "${baseline_eval}" \
        "${result_dir}/logs/v66-baseline.log"
fi
if [[ ! -f "${baseline_brake_eval}" ]]; then
    evaluate_policy \
        "${baseline_policy}" \
        "${baseline_brake_eval}" \
        "${result_dir}/logs/v66-baseline-drive-retention.log" \
        drive-retention
fi
if [[ ! -f "${baseline_idle_eval}" ]]; then
    evaluate_idle_steer \
        "${baseline_policy}" \
        "${baseline_idle_eval}" \
        "${result_dir}/logs/v66-baseline-idle-steer.log"
fi

latest_checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
[[ -n "${latest_checkpoint}" ]] || { echo "No V24 checkpoints found in ${run_dir}" >&2; exit 1; }
latest_iteration="${latest_checkpoint#model_}"
latest_iteration="${latest_iteration%.pt}"

# Early, middle, and late snapshots catch transient improvements that the final
# optimizer state may erase. The last available checkpoint is always included.
checkpoint_iterations=(0 100 200 300 400 500 600 700 800 900 1000 1100 "${latest_iteration}")
mapfile -t checkpoint_iterations < <(printf '%s\n' "${checkpoint_iterations[@]}" | sort -n -u)
blends=(0.925 0.950 0.965 0.980 1.000)

score_jsonl="${result_dir}/scorecard.jsonl"
: >"${score_jsonl}"

for iteration in "${checkpoint_iterations[@]}"; do
    checkpoint="${run_dir}/model_${iteration}.pt"
    [[ -f "${checkpoint}" ]] || continue
    actor="${result_dir}/actors/v24-model-${iteration}.onnx"
    if [[ ! -f "${actor}" ]]; then
        cd "${UPSTREAM_DIR}"
        "${UV_BIN}" run python scripts/export.py "${task}" \
            --checkpoint-file "${checkpoint}" \
            --onnx-file "${actor}" \
            --device cpu \
            --num-envs 1 >"${result_dir}/logs/export-${iteration}.log" 2>&1
    fi

    for blend in "${blends[@]}"; do
        blend_tag="${blend/./}"
        drive_hybrid="${result_dir}/hybrids/v66-v24-i${iteration}-b${blend_tag}-drive.onnx"
        hybrid="${result_dir}/hybrids/v66-v24-i${iteration}-b${blend_tag}-brake-safe.onnx"
        evaluation="${result_dir}/evaluations/v66-v24-i${iteration}-b${blend_tag}.json"
        brake_evaluation="${result_dir}/evaluations/v66-v24-i${iteration}-b${blend_tag}-drive-retention.json"
        idle_evaluation="${result_dir}/evaluations/v66-v24-i${iteration}-b${blend_tag}-idle-steer.json"
        if [[ ! -f "${drive_hybrid}" ]]; then
            cd "${UPSTREAM_DIR}"
            "${UV_BIN}" run python "${LAB_ROOT}/tools/build_hybrid_policy.py" \
                "${control_policy}" "${actor}" "${drive_hybrid}" \
                --speed-command-threshold 0.5 \
                --speed-blend "${blend}" \
                --smooth-turn-start 0.08 \
                --smooth-turn-end 0.25 >"${result_dir}/logs/build-i${iteration}-b${blend_tag}.log" 2>&1
        fi
        if [[ ! -f "${hybrid}" ]]; then
            cd "${UPSTREAM_DIR}"
            "${UV_BIN}" run python "${LAB_ROOT}/tools/build_brake_safe_policy.py" \
                "${drive_hybrid}" "${brake_policy}" "${hybrid}" \
                --zero-command-threshold 0.02 \
                --gate-mode joint_velocity \
                --joint-velocity-threshold 0.20 >"${result_dir}/logs/build-brake-i${iteration}-b${blend_tag}.log" 2>&1
        fi
        if [[ ! -f "${evaluation}" ]]; then
            evaluate_policy \
                "${hybrid}" \
                "${evaluation}" \
                "${result_dir}/logs/eval-i${iteration}-b${blend_tag}.log"
        fi
        if [[ ! -f "${brake_evaluation}" ]]; then
            evaluate_policy \
                "${hybrid}" \
                "${brake_evaluation}" \
                "${result_dir}/logs/eval-i${iteration}-b${blend_tag}-drive-retention.log" \
                drive-retention
        fi
        if [[ ! -f "${idle_evaluation}" ]]; then
            evaluate_idle_steer \
                "${hybrid}" \
                "${idle_evaluation}" \
                "${result_dir}/logs/eval-i${iteration}-b${blend_tag}-idle-steer.log"
        fi

        jq -cn \
            --argjson iteration "${iteration}" \
            --argjson blend "${blend}" \
            --arg policy "${hybrid}" \
            --slurpfile base "${baseline_eval}" \
            --slurpfile candidate "${evaluation}" \
            --slurpfile base_brake "${baseline_brake_eval}" \
            --slurpfile candidate_brake "${brake_evaluation}" \
            --slurpfile base_idle "${baseline_idle_eval}" \
            --slurpfile candidate_idle "${idle_evaluation}" '
              ($base[0]) as $b | ($candidate[0]) as $c |
              ($base_brake[0]) as $bb | ($candidate_brake[0]) as $cb |
              ($base_idle[0].phases.idle_launch) as $bi |
              ($candidate_idle[0].phases.idle_launch) as $ci |
              ($b.phases.max_speed) as $bm | ($c.phases.max_speed) as $cm |
              ($b.phases.stop_cruise) as $bs | ($c.phases.stop_cruise) as $cs |
              ($bb.phases.launch_high) as $bl | ($cb.phases.launch_high) as $cl |
              ($bb.phases.brake_high) as $bh | ($cb.phases.brake_high) as $ch |
              {
                iteration: $iteration,
                blend: $blend,
                policy: $policy,
                finished_100ft: $cm.finished_100ft,
                finish_time_100ft_s: ($cm.finish_time_100ft_s // 999.0),
                sustained_speed_mph: ($cm.steady_mean_world_forward_speed_mps * 2.2369362920544),
                verified_top_speed_mph: $cm.verified_top_speed_0_5s_mph,
                trap_speed_mph: ($cm.trap_speed_100ft_mph // 0.0),
                acceleration_mps2: $cm.acceleration_first_second_mps2,
                max_drift_ft: $cm.max_lateral_drift_ft,
                max_heading_error_deg: $cm.max_heading_error_deg,
                max_tilt_deg: $cm.tilt_max_deg,
                grounded_fraction: $cm.steady_both_blades_grounded_fraction,
                stop_time_s: ($cs.stop_time_below_0_05_mps_s // 999.0),
                high_speed_brake_time_s: ($ch.stop_time_below_0_05_mps_s // 999.0),
                high_speed_brake_end_mps: $ch.end_abs_forward_speed_mps,
                high_speed_brake_tilt_deg: $ch.tilt_max_deg,
                high_speed_brake_drift_ft: $ch.max_lateral_drift_ft,
                high_speed_brake_grounded_fraction: $ch.steady_both_blades_grounded_fraction,
                high_speed_launch_top_mph: $cl.verified_top_speed_0_5s_mph,
                idle_end_speed_mps: $ci.end_abs_forward_speed_mps,
                idle_forward_distance_m: $ci.forward_distance_m,
                idle_heading_error_deg: $ci.max_heading_error_deg,
                idle_tilt_deg: $ci.tilt_max_deg,
                baseline_finish_time_s: ($bm.finish_time_100ft_s // 999.0),
                baseline_sustained_mph: ($bm.steady_mean_world_forward_speed_mps * 2.2369362920544),
                baseline_top_speed_mph: $bm.verified_top_speed_0_5s_mph,
                baseline_drift_ft: $bm.max_lateral_drift_ft,
                baseline_heading_deg: $bm.max_heading_error_deg,
                baseline_tilt_deg: $bm.tilt_max_deg,
                baseline_high_speed_brake_time_s: ($bh.stop_time_below_0_05_mps_s // 999.0),
                baseline_high_speed_launch_top_mph: $bl.verified_top_speed_0_5s_mph,
                promotable: (
                  $cm.finished_100ft and
                  (($cm.finish_time_100ft_s // 999.0) < ($bm.finish_time_100ft_s // 999.0)) and
                  ($cm.steady_mean_world_forward_speed_mps > $bm.steady_mean_world_forward_speed_mps) and
                  ($cm.verified_top_speed_0_5s_mph >= (0.99 * $bm.verified_top_speed_0_5s_mph)) and
                  ($cm.max_lateral_drift_ft <= ($bm.max_lateral_drift_ft + 0.05)) and
                  ($cm.max_heading_error_deg <= ($bm.max_heading_error_deg + 1.0)) and
                  ($cm.tilt_max_deg <= ($bm.tilt_max_deg + 1.0)) and
                  ($cm.steady_both_blades_grounded_fraction >= ($bm.steady_both_blades_grounded_fraction - 0.03)) and
                  (($cs.stop_time_below_0_05_mps_s // 999.0) <= (($bs.stop_time_below_0_05_mps_s // 999.0) + 0.10)) and
                  ($ch.end_abs_forward_speed_mps <= 0.05) and
                  (($ch.stop_time_below_0_05_mps_s // 999.0) <= 4.0) and
                  ($ch.acceleration_first_second_mps2 <= -0.25) and
                  ($ch.tilt_max_deg <= 25.0) and
                  ($ch.steady_both_blades_grounded_fraction >= 0.95) and
                  ($ch.max_lateral_drift_ft <= ($bh.max_lateral_drift_ft + 0.25)) and
                  ($ci.end_abs_forward_speed_mps <= 0.01) and
                  ($ci.forward_distance_m <= ($bi.forward_distance_m + 0.02)) and
                  ($ci.max_heading_error_deg <= ($bi.max_heading_error_deg + 1.0)) and
                  ($ci.tilt_max_deg <= ($bi.tilt_max_deg + 1.0))
                ),
                score: (
                  4.0 * (($bm.finish_time_100ft_s // 999.0) / ($cm.finish_time_100ft_s // 999.0)) +
                  2.0 * ($cm.steady_mean_world_forward_speed_mps / $bm.steady_mean_world_forward_speed_mps) +
                  1.0 * ($cm.verified_top_speed_0_5s_mph / $bm.verified_top_speed_0_5s_mph) +
                  1.0 * ($bm.max_lateral_drift_ft / (($cm.max_lateral_drift_ft | if . < 0.01 then 0.01 else . end))) +
                  1.0 * ($bm.max_heading_error_deg / (($cm.max_heading_error_deg | if . < 0.1 then 0.1 else . end))) +
                  1.0 * ($bm.tilt_max_deg / (($cm.tilt_max_deg | if . < 0.1 then 0.1 else . end)))
                )
              }' >>"${score_jsonl}"
    done
done

jq -s 'sort_by(.score) | reverse' "${score_jsonl}" >"${result_dir}/scorecard.json"
jq -s '[.[] | select(.promotable)] | sort_by(.score) | reverse | .[0] // null' \
    "${score_jsonl}" >"${result_dir}/promotion-candidate.json"

best_any="$(jq -s 'sort_by(.score) | last' "${score_jsonl}")"
best_promotable="$(cat "${result_dir}/promotion-candidate.json")"
printf '%s\n' "${best_any}" >"${result_dir}/best-measured.json"

cat >"${result_dir}/README.md" <<EOF
# DuckWing V24 final checkpoint screen

- Incumbent: V66
- Training run: ${run_name}
- Checkpoints sampled: ${#checkpoint_iterations[@]}
- Router blends sampled: ${blends[*]}
- Physics: 1.75 A current limit, 0.003 wheel frictionloss
- Course control: line hold 0.70 / 0.14 / 0.07 / 0.15
- Promotion rule: faster 100-foot finish and sustained speed, at least 99% of
  incumbent top speed, with bounded regressions in drift, heading, tilt, blade
  contact, low-speed braking, high-speed braking, and
  zero-command steering stability. High-speed braking must remain below 25
  degrees tilt with at least 95% steady blade contact.

The full ranked data is in scorecard.json. A null promotion-candidate.json means
V66 remains the definitive model.
EOF

touch "${done_marker}"

if [[ "${best_promotable}" == "null" ]]; then
    printf 'DuckWing V24 evaluation complete: no checkpoint cleared the V66 promotion gates. V66 remains definitive.\nBest measured (rejected): iteration %s, blend %.3f, %.3f s/100ft, %.3f mph sustained, %.3f mph top.\n' \
        "$(jq -r '.iteration' <<<"${best_any}")" \
        "$(jq -r '.blend' <<<"${best_any}")" \
        "$(jq -r '.finish_time_100ft_s' <<<"${best_any}")" \
        "$(jq -r '.sustained_speed_mph' <<<"${best_any}")" \
        "$(jq -r '.verified_top_speed_mph' <<<"${best_any}")"
else
    printf 'DuckWing V24 produced a promotion candidate: iteration %s, blend %.3f, %.3f s/100ft, %.3f mph sustained, %.3f mph top. Candidate policy: %s\n' \
        "$(jq -r '.iteration' <<<"${best_promotable}")" \
        "$(jq -r '.blend' <<<"${best_promotable}")" \
        "$(jq -r '.finish_time_100ft_s' <<<"${best_promotable}")" \
        "$(jq -r '.sustained_speed_mph' <<<"${best_promotable}")" \
        "$(jq -r '.verified_top_speed_mph' <<<"${best_promotable}")" \
        "$(jq -r '.policy' <<<"${best_promotable}")"
fi
