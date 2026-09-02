#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

fusion_dir="${REPORT_DIR}/duckwing-v67-joint-fusion"
baseline="${REPORT_DIR}/duckwing-v24-final-screen/evaluations/v66-baseline.json"
output_dir="${REPORT_DIR}/duckwing-v67-linehold"
mkdir -p "${output_dir}/evaluations" "${output_dir}/logs"
score_jsonl="${output_dir}/scorecard.jsonl"
: >"${score_jsonl}"
cd "${UPSTREAM_DIR}"

for seed in speed balanced; do
    case "${seed}" in
        speed) policy="${fusion_dir}/policies/s000-p110-h000-brake-safe.onnx" ;;
        balanced) policy="${fusion_dir}/policies/s010-p100-h000-brake-safe.onnx" ;;
    esac
    [[ -f "${policy}" ]] || { echo "Seed policy missing: ${policy}" >&2; exit 1; }
    for yaw_kp in 0.70 0.80 0.90; do
        for lateral_kp in 0.14 0.18 0.22; do
            for yaw_kd in 0.07 0.10 0.14; do
                for max_wz in 0.15 0.18 0.20; do
                    tag="${seed}-yk${yaw_kp/./}-lk${lateral_kp/./}-yd${yaw_kd/./}-mw${max_wz/./}"
                    evaluation="${output_dir}/evaluations/${tag}.json"
                    if [[ ! -f "${evaluation}" ]]; then
                        "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
                            "${policy}" --profile race-5mph \
                            --current-limit 1.75 --wheel-friction 0.003 \
                            --line-hold --line-yaw-kp "${yaw_kp}" \
                            --line-lateral-kp "${lateral_kp}" \
                            --line-yaw-kd "${yaw_kd}" --line-max-wz "${max_wz}" \
                            --output "${evaluation}" >"${output_dir}/logs/${tag}.log" 2>&1
                    fi
                    jq -cn \
                        --arg seed "${seed}" \
                        --arg policy "${policy}" \
                        --argjson yaw_kp "${yaw_kp}" \
                        --argjson lateral_kp "${lateral_kp}" \
                        --argjson yaw_kd "${yaw_kd}" \
                        --argjson max_wz "${max_wz}" \
                        --slurpfile base "${baseline}" \
                        --slurpfile candidate "${evaluation}" '
                          ($base[0].phases.max_speed) as $b |
                          ($candidate[0].phases.max_speed) as $c |
                          ($base[0].phases.stop_cruise) as $bs |
                          ($candidate[0].phases.stop_cruise) as $cs |
                          {
                            seed: $seed, policy: $policy,
                            yaw_kp: $yaw_kp, lateral_kp: $lateral_kp,
                            yaw_kd: $yaw_kd, max_wz: $max_wz,
                            wheel_frictionloss: $candidate[0].wheel_frictionloss,
                            current_limit_a: $candidate[0].current_limit_a,
                            finish_time_100ft_s: ($c.finish_time_100ft_s // 999.0),
                            sustained_speed_mph: ($c.steady_mean_world_forward_speed_mps * 2.2369362920544),
                            verified_top_speed_mph: $c.verified_top_speed_0_5s_mph,
                            acceleration_mps2: $c.acceleration_first_second_mps2,
                            max_drift_ft: $c.max_lateral_drift_ft,
                            max_heading_error_deg: $c.max_heading_error_deg,
                            max_tilt_deg: $c.tilt_max_deg,
                            grounded_fraction: $c.steady_both_blades_grounded_fraction,
                            stop_time_s: ($cs.stop_time_below_0_05_mps_s // 999.0),
                            strict_promotable: (
                              ($candidate[0].wheel_frictionloss == 0.003) and
                              ($candidate[0].current_limit_a == 1.75) and
                              $c.finished_100ft and
                              (($c.finish_time_100ft_s // 999.0) < ($b.finish_time_100ft_s // 999.0)) and
                              ($c.steady_mean_world_forward_speed_mps > $b.steady_mean_world_forward_speed_mps) and
                              ($c.verified_top_speed_0_5s_mph >= $b.verified_top_speed_0_5s_mph) and
                              ($c.acceleration_first_second_mps2 >= $b.acceleration_first_second_mps2) and
                              ($c.max_lateral_drift_ft <= $b.max_lateral_drift_ft) and
                              ($c.max_heading_error_deg <= $b.max_heading_error_deg) and
                              ($c.tilt_max_deg <= $b.tilt_max_deg) and
                              ($c.steady_both_blades_grounded_fraction >= $b.steady_both_blades_grounded_fraction) and
                              (($cs.stop_time_below_0_05_mps_s // 999.0) <= ($bs.stop_time_below_0_05_mps_s // 999.0))
                            ),
                            score: (
                              4.0 * (($b.finish_time_100ft_s // 999.0) / ($c.finish_time_100ft_s // 999.0)) +
                              2.0 * ($c.steady_mean_world_forward_speed_mps / $b.steady_mean_world_forward_speed_mps) +
                              ($c.verified_top_speed_0_5s_mph / $b.verified_top_speed_0_5s_mph) +
                              ($c.acceleration_first_second_mps2 / $b.acceleration_first_second_mps2) +
                              ($b.max_lateral_drift_ft / (($c.max_lateral_drift_ft | if . < 0.01 then 0.01 else . end))) +
                              ($b.max_heading_error_deg / (($c.max_heading_error_deg | if . < 0.1 then 0.1 else . end))) +
                              ($b.tilt_max_deg / (($c.tilt_max_deg | if . < 0.1 then 0.1 else . end))) +
                              ($c.steady_both_blades_grounded_fraction / $b.steady_both_blades_grounded_fraction)
                            )
                          }' >>"${score_jsonl}"
                done
            done
        done
    done
done

jq -s 'sort_by(.score) | reverse' "${score_jsonl}" >"${output_dir}/scorecard.json"
jq -s '[.[] | select(.strict_promotable)] | sort_by(.score) | reverse' \
    "${score_jsonl}" >"${output_dir}/promotion-candidates.json"
count="$(jq 'length' "${output_dir}/promotion-candidates.json")"
printf 'V67 line-hold sweep complete: %s strict promotion candidates. Results: %s\n' \
    "${count}" "${output_dir}/scorecard.json"
