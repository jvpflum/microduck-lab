#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

incumbent="${LAB_ROOT}/releases/v66/duckwing-v66-v65-control-fusion.onnx"
specialist="${LAB_ROOT}/incoming/rtx5090/v47-official-friction-speed-specialist/policy.onnx"
brake="${LAB_ROOT}/incoming/rtx5090/v65-v63-immediate-switch-2026-09-01/policy.onnx"
baseline="${REPORT_DIR}/duckwing-v24-final-screen/evaluations/v66-baseline.json"
output_dir="${REPORT_DIR}/duckwing-v67-authority-refine"
mkdir -p "${output_dir}/drive" "${output_dir}/policies" "${output_dir}/evaluations" "${output_dir}/logs"
score_jsonl="${output_dir}/scorecard.jsonl"
: >"${score_jsonl}"
cd "${UPSTREAM_DIR}"

for controller in precision performance; do
    case "${controller}" in
        precision) yaw_kd=0.10 ;;
        performance) yaw_kd=0.07 ;;
    esac
    for steering in 0.050 0.075 0.100 0.125 0.150 0.175 0.200 0.225 0.250; do
        for propulsion in 0.950 0.975 1.000 1.025 1.050 1.075 1.100; do
            tag="${controller}-s${steering/./}-p${propulsion/./}"
            drive="${output_dir}/drive/${tag}.onnx"
            policy="${output_dir}/policies/${tag}-brake-safe.onnx"
            evaluation="${output_dir}/evaluations/${tag}.json"
            if [[ ! -f "${drive}" ]]; then
                "${UV_BIN}" run python "${LAB_ROOT}/tools/build_joint_fusion_policy.py" \
                    "${incumbent}" "${specialist}" "${drive}" \
                    --steering-authority "${steering}" \
                    --propulsion-authority "${propulsion}" \
                    --head-authority 0.0 \
                    --speed-command-threshold 0.5 \
                    --smooth-turn-start 0.08 --smooth-turn-end 0.25 \
                    >"${output_dir}/logs/build-${tag}.log" 2>&1
            fi
            if [[ ! -f "${policy}" ]]; then
                "${UV_BIN}" run python "${LAB_ROOT}/tools/build_brake_safe_policy.py" \
                    "${drive}" "${brake}" "${policy}" \
                    --zero-command-threshold 0.02 --gate-mode joint_velocity \
                    --joint-velocity-threshold 0.20 \
                    >"${output_dir}/logs/brake-${tag}.log" 2>&1
            fi
            if [[ ! -f "${evaluation}" ]]; then
                "${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
                    "${policy}" --profile race-5mph \
                    --current-limit 1.75 --wheel-friction 0.003 \
                    --line-hold --line-yaw-kp 0.70 --line-lateral-kp 0.22 \
                    --line-yaw-kd "${yaw_kd}" --line-max-wz 0.15 \
                    --output "${evaluation}" >"${output_dir}/logs/eval-${tag}.log" 2>&1
            fi

            jq -cn \
                --arg controller "${controller}" \
                --arg policy "${policy}" \
                --argjson steering "${steering}" \
                --argjson propulsion "${propulsion}" \
                --argjson yaw_kd "${yaw_kd}" \
                --slurpfile base "${baseline}" \
                --slurpfile candidate "${evaluation}" '
                  ($base[0].phases.max_speed) as $b |
                  ($candidate[0].phases.max_speed) as $c |
                  ($base[0].phases.stop_cruise) as $bs |
                  ($candidate[0].phases.stop_cruise) as $cs |
                  {
                    controller: $controller, policy: $policy,
                    steering_authority: $steering,
                    propulsion_authority: $propulsion,
                    yaw_kp: 0.70, lateral_kp: 0.22,
                    yaw_kd: $yaw_kd, max_wz: 0.15,
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

jq -s 'sort_by(.score) | reverse' "${score_jsonl}" >"${output_dir}/scorecard.json"
jq -s '[.[] | select(.strict_promotable)] | sort_by(.score) | reverse' \
    "${score_jsonl}" >"${output_dir}/race-promotion-candidates.json"
count="$(jq 'length' "${output_dir}/race-promotion-candidates.json")"
printf 'V67 authority refinement complete: %s strict race-promotable policies. Results: %s\n' \
    "${count}" "${output_dir}/scorecard.json"
