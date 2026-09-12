#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_checkout
require_uv
prepare_dirs

policy="${1:-${REPORT_DIR}/duckwing-v67-authority-refine/policies/performance-s0250-p1050-brake-safe.onnx}"
output_dir="${REPORT_DIR}/duckwing-v67-final-validation"
baseline_dir="${REPORT_DIR}/duckwing-v24-final-screen/evaluations"
mkdir -p "${output_dir}/logs"

[[ -f "${policy}" ]] || { echo "V67 candidate missing: ${policy}" >&2; exit 1; }
cd "${UPSTREAM_DIR}"

"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy}" --profile race-5mph \
    --current-limit 1.75 --wheel-friction 0.003 \
    --line-hold --line-yaw-kp 0.70 --line-lateral-kp 0.22 \
    --line-yaw-kd 0.07 --line-max-wz 0.15 \
    --output "${output_dir}/race5.json" >"${output_dir}/logs/race5.log" 2>&1

"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy}" --profile drive-retention \
    --current-limit 1.75 --wheel-friction 0.003 \
    --line-hold --line-yaw-kp 0.70 --line-lateral-kp 0.22 \
    --line-yaw-kd 0.07 --line-max-wz 0.15 \
    --output "${output_dir}/drive-retention.json" \
    >"${output_dir}/logs/drive-retention.log" 2>&1

"${UV_BIN}" run python "${LAB_ROOT}/tools/evaluate_swizzle.py" \
    "${policy}" --profile idle-launch \
    --current-limit 1.75 --wheel-friction 0.003 \
    --launch-yaw-command -0.30 --launch-yaw-pulse-duration 0.10 \
    --launch-yaw-pulse-count 4 --launch-yaw-pulse-gap 0.10 \
    --output "${output_dir}/idle-steer.json" >"${output_dir}/logs/idle-steer.log" 2>&1

jq -n \
    --arg policy "${policy}" \
    --slurpfile base "${baseline_dir}/v66-baseline.json" \
    --slurpfile race "${output_dir}/race5.json" \
    --slurpfile base_brake "${baseline_dir}/v66-baseline-drive-retention.json" \
    --slurpfile brake "${output_dir}/drive-retention.json" \
    --slurpfile base_idle "${baseline_dir}/v66-baseline-idle-steer.json" \
    --slurpfile idle "${output_dir}/idle-steer.json" '
      ($base[0].phases.max_speed) as $b |
      ($race[0].phases.max_speed) as $r |
      ($base[0].phases.stop_cruise) as $bs |
      ($race[0].phases.stop_cruise) as $rs |
      ($base_brake[0].phases.launch_high) as $bl |
      ($brake[0].phases.launch_high) as $cl |
      ($base_brake[0].phases.brake_high) as $bb |
      ($brake[0].phases.brake_high) as $cb |
      ($base_idle[0].phases.idle_launch) as $bi |
      ($idle[0].phases.idle_launch) as $ci |
      {
        policy: $policy,
        physics: {
          wheel_frictionloss: $race[0].wheel_frictionloss,
          current_limit_a: $race[0].current_limit_a
        },
        controller: {yaw_kp: 0.70, lateral_kp: 0.22, yaw_kd: 0.07, max_wz: 0.15},
        incumbent: {
          finish_time_100ft_s: $b.finish_time_100ft_s,
          sustained_speed_mph: ($b.steady_mean_world_forward_speed_mps * 2.2369362920544),
          verified_top_speed_mph: $b.verified_top_speed_0_5s_mph,
          acceleration_mps2: $b.acceleration_first_second_mps2,
          max_drift_ft: $b.max_lateral_drift_ft,
          max_heading_error_deg: $b.max_heading_error_deg,
          max_tilt_deg: $b.tilt_max_deg,
          grounded_fraction: $b.steady_both_blades_grounded_fraction,
          low_speed_stop_time_s: $bs.stop_time_below_0_05_mps_s
        },
        candidate: {
          finish_time_100ft_s: $r.finish_time_100ft_s,
          sustained_speed_mph: ($r.steady_mean_world_forward_speed_mps * 2.2369362920544),
          verified_top_speed_mph: $r.verified_top_speed_0_5s_mph,
          acceleration_mps2: $r.acceleration_first_second_mps2,
          max_drift_ft: $r.max_lateral_drift_ft,
          max_heading_error_deg: $r.max_heading_error_deg,
          max_tilt_deg: $r.tilt_max_deg,
          grounded_fraction: $r.steady_both_blades_grounded_fraction,
          low_speed_stop_time_s: $rs.stop_time_below_0_05_mps_s,
          high_speed_launch_top_mph: $cl.verified_top_speed_0_5s_mph,
          high_speed_brake_time_s: $cb.stop_time_below_0_05_mps_s,
          high_speed_brake_end_mps: $cb.end_abs_forward_speed_mps,
          high_speed_brake_deceleration_mps2: $cb.acceleration_first_second_mps2,
          high_speed_brake_tilt_deg: $cb.tilt_max_deg,
          high_speed_brake_drift_ft: $cb.max_lateral_drift_ft,
          high_speed_brake_grounded_fraction: $cb.steady_both_blades_grounded_fraction,
          idle_end_speed_mps: $ci.end_abs_forward_speed_mps,
          idle_forward_distance_m: $ci.forward_distance_m,
          idle_heading_error_deg: $ci.max_heading_error_deg,
          idle_tilt_deg: $ci.tilt_max_deg
        },
        race_all_metrics_improved: (
          ($race[0].wheel_frictionloss == 0.003) and
          ($race[0].current_limit_a == 1.75) and
          $r.finished_100ft and
          ($r.finish_time_100ft_s < $b.finish_time_100ft_s) and
          ($r.steady_mean_world_forward_speed_mps > $b.steady_mean_world_forward_speed_mps) and
          ($r.verified_top_speed_0_5s_mph >= $b.verified_top_speed_0_5s_mph) and
          ($r.acceleration_first_second_mps2 >= $b.acceleration_first_second_mps2) and
          ($r.max_lateral_drift_ft <= $b.max_lateral_drift_ft) and
          ($r.max_heading_error_deg <= $b.max_heading_error_deg) and
          ($r.tilt_max_deg <= $b.tilt_max_deg) and
          ($r.steady_both_blades_grounded_fraction >= $b.steady_both_blades_grounded_fraction) and
          ($rs.stop_time_below_0_05_mps_s <= $bs.stop_time_below_0_05_mps_s)
        ),
        high_speed_brake_safe: (
          ($brake[0].wheel_frictionloss == 0.003) and
          ($cl.verified_top_speed_0_5s_mph >= (0.99 * $bl.verified_top_speed_0_5s_mph)) and
          ($cb.end_abs_forward_speed_mps <= 0.05) and
          (($cb.stop_time_below_0_05_mps_s // 999.0) <= 4.0) and
          ($cb.acceleration_first_second_mps2 <= -0.25) and
          ($cb.tilt_max_deg <= 25.0) and
          ($cb.steady_both_blades_grounded_fraction >= 0.95) and
          ($cb.max_lateral_drift_ft <= $bb.max_lateral_drift_ft)
        ),
        idle_safe: (
          ($ci.end_abs_forward_speed_mps <= 0.01) and
          ($ci.forward_distance_m <= ($bi.forward_distance_m + 0.02)) and
          ($ci.max_heading_error_deg <= ($bi.max_heading_error_deg + 1.0)) and
          ($ci.tilt_max_deg <= ($bi.tilt_max_deg + 1.0))
        )
      } |
      . + {full_promotable: (.race_all_metrics_improved and .high_speed_brake_safe and .idle_safe)}' \
    >"${output_dir}/verification-summary.json"

jq '.' "${output_dir}/verification-summary.json"
