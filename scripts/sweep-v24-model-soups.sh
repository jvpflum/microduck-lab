#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

screen_dir="${REPORT_DIR}/duckwing-v24-final-screen"
baseline="${screen_dir}/evaluations/v66-baseline.json"
control="${LAB_ROOT}/policy-bench/runs/race5-microduck_hybrid_controlaware-export/artifacts/hybrid_v11_i6159_smooth_t02_t12_b100.onnx"
brake="${LAB_ROOT}/incoming/rtx5090/v65-v63-immediate-switch-2026-09-01/policy.onnx"
output_dir="${REPORT_DIR}/duckwing-v24-model-soups"
mkdir -p "${output_dir}/actors" "${output_dir}/hybrids" "${output_dir}/evaluations" "${output_dir}/logs"

[[ -f "${baseline}" ]] || { echo "Baseline evaluation missing: ${baseline}" >&2; exit 1; }
score_jsonl="${output_dir}/scorecard.jsonl"
: >"${score_jsonl}"
cd "${UPSTREAM_DIR}"

for right_iteration in 0 200 600 900; do
    left="${screen_dir}/actors/v24-model-100.onnx"
    right="${screen_dir}/actors/v24-model-${right_iteration}.onnx"
    [[ -f "${left}" && -f "${right}" ]] || continue
    for alpha in 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90; do
        alpha_tag="${alpha/./}"
        tag="i100-i${right_iteration}-a${alpha_tag}"
        actor="${output_dir}/actors/${tag}.onnx"
        drive="${output_dir}/hybrids/${tag}-drive.onnx"
        policy="${output_dir}/hybrids/${tag}-brake-safe.onnx"
        evaluation="${output_dir}/evaluations/${tag}.json"
        if [[ ! -f "${actor}" ]]; then
            "${UV_BIN}" run python "${LAB_ROOT}/tools/blend_onnx_models.py" \
                "${left}" "${right}" "${actor}" --alpha "${alpha}" \
                >"${output_dir}/logs/blend-${tag}.log" 2>&1
        fi
        if [[ ! -f "${drive}" ]]; then
            "${UV_BIN}" run python "${LAB_ROOT}/tools/build_hybrid_policy.py" \
                "${control}" "${actor}" "${drive}" \
                --speed-command-threshold 0.5 \
                --speed-blend 1.0 \
                --smooth-turn-start 0.08 \
                --smooth-turn-end 0.25 >"${output_dir}/logs/build-${tag}.log" 2>&1
        fi
        if [[ ! -f "${policy}" ]]; then
            "${UV_BIN}" run python "${LAB_ROOT}/tools/build_brake_safe_policy.py" \
                "${drive}" "${brake}" "${policy}" \
                --zero-command-threshold 0.02 \
                --gate-mode joint_velocity \
                --joint-velocity-threshold 0.20 >"${output_dir}/logs/brake-${tag}.log" 2>&1
        fi
        if [[ ! -f "${evaluation}" ]]; then
            "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
                "${policy}" --profile race-5mph --line-hold \
                --line-yaw-kp 0.70 --line-lateral-kp 0.14 \
                --line-yaw-kd 0.07 --line-max-wz 0.15 \
                --output "${evaluation}" >"${output_dir}/logs/eval-${tag}.log" 2>&1
        fi

        jq -cn \
            --argjson right_iteration "${right_iteration}" \
            --argjson alpha "${alpha}" \
            --arg policy "${policy}" \
            --slurpfile base "${baseline}" \
            --slurpfile candidate "${evaluation}" '
              ($base[0].phases.max_speed) as $b |
              ($candidate[0].phases.max_speed) as $c |
              {
                left_iteration: 100,
                right_iteration: $right_iteration,
                alpha: $alpha,
                policy: $policy,
                finish_time_100ft_s: ($c.finish_time_100ft_s // 999.0),
                sustained_speed_mph: ($c.steady_mean_world_forward_speed_mps * 2.2369362920544),
                verified_top_speed_mph: $c.verified_top_speed_0_5s_mph,
                max_drift_ft: $c.max_lateral_drift_ft,
                max_heading_error_deg: $c.max_heading_error_deg,
                max_tilt_deg: $c.tilt_max_deg,
                grounded_fraction: $c.steady_both_blades_grounded_fraction,
                baseline_finish_time_s: ($b.finish_time_100ft_s // 999.0),
                baseline_sustained_mph: ($b.steady_mean_world_forward_speed_mps * 2.2369362920544),
                baseline_top_speed_mph: $b.verified_top_speed_0_5s_mph,
                baseline_drift_ft: $b.max_lateral_drift_ft,
                baseline_heading_deg: $b.max_heading_error_deg,
                baseline_tilt_deg: $b.tilt_max_deg,
                baseline_grounded_fraction: $b.steady_both_blades_grounded_fraction,
                race_promotable: (
                  $c.finished_100ft and
                  (($c.finish_time_100ft_s // 999.0) < ($b.finish_time_100ft_s // 999.0)) and
                  ($c.steady_mean_world_forward_speed_mps > $b.steady_mean_world_forward_speed_mps) and
                  ($c.verified_top_speed_0_5s_mph >= (0.99 * $b.verified_top_speed_0_5s_mph)) and
                  ($c.max_lateral_drift_ft <= ($b.max_lateral_drift_ft + 0.05)) and
                  ($c.max_heading_error_deg <= ($b.max_heading_error_deg + 1.0)) and
                  ($c.tilt_max_deg <= ($b.tilt_max_deg + 1.0)) and
                  ($c.steady_both_blades_grounded_fraction >= ($b.steady_both_blades_grounded_fraction - 0.03))
                ),
                score: (
                  4.0 * (($b.finish_time_100ft_s // 999.0) / ($c.finish_time_100ft_s // 999.0)) +
                  2.0 * ($c.steady_mean_world_forward_speed_mps / $b.steady_mean_world_forward_speed_mps) +
                  ($c.verified_top_speed_0_5s_mph / $b.verified_top_speed_0_5s_mph) +
                  ($b.max_lateral_drift_ft / (($c.max_lateral_drift_ft | if . < 0.01 then 0.01 else . end))) +
                  ($b.max_heading_error_deg / (($c.max_heading_error_deg | if . < 0.1 then 0.1 else . end))) +
                  ($b.tilt_max_deg / (($c.tilt_max_deg | if . < 0.1 then 0.1 else . end)))
                )
              }' >>"${score_jsonl}"
    done
done

jq -s 'sort_by(.score) | reverse' "${score_jsonl}" >"${output_dir}/scorecard.json"
jq -s '[.[] | select(.race_promotable)] | sort_by(.score) | reverse' \
    "${score_jsonl}" >"${output_dir}/promotion-candidates.json"
count="$(jq 'length' "${output_dir}/promotion-candidates.json")"
printf 'V24 model-soup sweep complete: %s race-promotable policies. Results: %s\n' \
    "${count}" "${output_dir}/scorecard.json"
