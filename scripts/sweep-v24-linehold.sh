#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

policy="${1:-${REPORT_DIR}/duckwing-v24-final-screen/hybrids/v66-v24-i100-b1000-brake-safe.onnx}"
baseline="${REPORT_DIR}/duckwing-v24-final-screen/evaluations/v66-baseline.json"
output_dir="${REPORT_DIR}/duckwing-v24-linehold-sweep"
mkdir -p "${output_dir}/evaluations" "${output_dir}/logs"

[[ -f "${policy}" ]] || { echo "Candidate policy missing: ${policy}" >&2; exit 1; }
[[ -f "${baseline}" ]] || { echo "Baseline evaluation missing: ${baseline}" >&2; exit 1; }

score_jsonl="${output_dir}/scorecard.jsonl"
: >"${score_jsonl}"

for yaw_kp in 0.70 0.90 1.10; do
    for lateral_kp in 0.14 0.22; do
        for yaw_kd in 0.07 0.14; do
            for max_wz in 0.15 0.20 0.25; do
                tag="yk${yaw_kp/./}-lk${lateral_kp/./}-yd${yaw_kd/./}-mw${max_wz/./}"
                evaluation="${output_dir}/evaluations/${tag}.json"
                log="${output_dir}/logs/${tag}.log"
                if [[ ! -f "${evaluation}" ]]; then
                    cd "${UPSTREAM_DIR}"
                    "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
                        "${policy}" \
                        --profile race-5mph \
                        --line-hold \
                        --line-yaw-kp "${yaw_kp}" \
                        --line-lateral-kp "${lateral_kp}" \
                        --line-yaw-kd "${yaw_kd}" \
                        --line-max-wz "${max_wz}" \
                        --output "${evaluation}" >"${log}" 2>&1
                fi

                jq -cn \
                    --argjson yaw_kp "${yaw_kp}" \
                    --argjson lateral_kp "${lateral_kp}" \
                    --argjson yaw_kd "${yaw_kd}" \
                    --argjson max_wz "${max_wz}" \
                    --arg policy "${policy}" \
                    --slurpfile base "${baseline}" \
                    --slurpfile candidate "${evaluation}" '
                      ($base[0].phases.max_speed) as $b |
                      ($candidate[0].phases.max_speed) as $c |
                      {
                        yaw_kp: $yaw_kp,
                        lateral_kp: $lateral_kp,
                        yaw_kd: $yaw_kd,
                        max_wz: $max_wz,
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
    done
done

jq -s 'sort_by(.score) | reverse' "${score_jsonl}" >"${output_dir}/scorecard.json"
jq -s '[.[] | select(.race_promotable)] | sort_by(.score) | reverse' \
    "${score_jsonl}" >"${output_dir}/promotion-candidates.json"

count="$(jq 'length' "${output_dir}/promotion-candidates.json")"
printf 'V24 line-hold sweep complete: %s race-promotable configurations. Results: %s\n' \
    "${count}" "${output_dir}/scorecard.json"
