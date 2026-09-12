#!/usr/bin/env bash
set -euo pipefail

# Optional resource coordination for a machine that runs other GPU services.
# DuckLab never assumes a particular inference server, container runtime, or
# service manager. The default "shared" profile does not change anything.
#
# To let a dedicated training run pause local services, the operator explicitly
# supplies commands, for example:
#   export DUCKLAB_RESOURCE_STOP_CMD='systemctl --user stop my-service'
#   export DUCKLAB_RESOURCE_RESTORE_CMD='systemctl --user start my-service'
# The commands are intentionally not stored in Git or the marker.

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="${DUCKLAB_RESOURCE_MARKER:-${LAB_ROOT}/policy-bench/training-priority.json}"
HOOK_LOCK="${DUCKLAB_RESOURCE_LOCK:-${LAB_ROOT}/policy-bench/training-priority.lock}"
STOP_CMD="${DUCKLAB_RESOURCE_STOP_CMD:-}"
RESTORE_CMD="${DUCKLAB_RESOURCE_RESTORE_CMD:-}"
HEALTH_CMD="${DUCKLAB_RESOURCE_HEALTH_CMD:-}"

marker_pid() {
  sed -n 's/.*"owner_pid": *\([0-9][0-9]*\).*/\1/p' "${MARKER}" 2>/dev/null | head -n 1
}

marker_stopped_service() {
  sed -n 's/.*"stop_hook_ran": *\(true\|false\).*/\1/p' "${MARKER}" 2>/dev/null | head -n 1
}

run_hook() {
  local label="$1"
  local command="$2"
  [[ -n "${command}" ]] || return 0
  echo "Resource profile: ${label}."
  bash -lc "${command}"
}

enter_priority() {
  local owner_pid="$1"
  local existing_pid=""
  local stop_hook_ran=false

  mkdir -p "$(dirname "${MARKER}")"
  if [[ -f "${MARKER}" ]]; then
    existing_pid="$(marker_pid)"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Training priority is already owned by process ${existing_pid}." >&2
      exit 1
    fi
    echo "Recovering stale training-priority marker."
    rm -f "${MARKER}"
  fi

  if [[ -n "${STOP_CMD}" ]]; then
    exec 9>"${HOOK_LOCK}"
    flock -w 30 9
    run_hook "running configured stop hook" "${STOP_CMD}"
    stop_hook_ran=true
  else
    echo "Training priority: no resource hooks configured; leaving other services unchanged."
  fi

  printf '{"owner_pid": %s, "stop_hook_ran": %s, "started_at": "%s"}\n' \
    "${owner_pid}" "${stop_hook_ran}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${MARKER}"
}

restore_priority() {
  local owner_pid="${1:-}"
  local recorded_pid=""
  local stop_hook_ran=false

  [[ -f "${MARKER}" ]] || return 0
  recorded_pid="$(marker_pid)"
  if [[ -z "${owner_pid}" && -n "${recorded_pid}" ]] \
    && kill -0 "${recorded_pid}" 2>/dev/null; then
    echo "Refusing manual restore while training owner ${recorded_pid} is still running." >&2
    exit 1
  fi
  if [[ -n "${owner_pid}" && -n "${recorded_pid}" && "${owner_pid}" != "${recorded_pid}" ]]; then
    echo "Refusing to restore a training-priority profile owned by process ${recorded_pid}." >&2
    exit 1
  fi
  stop_hook_ran="$(marker_stopped_service)"

  if [[ "${stop_hook_ran}" == "true" ]]; then
    exec 9>"${HOOK_LOCK}"
    flock -w 30 9
    if [[ -z "${RESTORE_CMD}" ]]; then
      echo "Configured stop hook ran, but DUCKLAB_RESOURCE_RESTORE_CMD is not set; marker retained at ${MARKER}." >&2
      return 1
    fi
    run_hook "running configured restore hook" "${RESTORE_CMD}"
    run_hook "running configured health hook" "${HEALTH_CMD}"
  fi

  rm -f "${MARKER}"
}

case "${1:-}" in
  enter)
    [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "Usage: $0 enter OWNER_PID" >&2; exit 2; }
    enter_priority "$2"
    ;;
  restore)
    restore_priority "${2:-}"
    ;;
  status)
    if [[ -f "${MARKER}" ]]; then
      printf 'training-priority owner=%s stop_hook_ran=%s\n' "$(marker_pid)" "$(marker_stopped_service)"
    else
      echo "shared"
    fi
    ;;
  *)
    echo "Usage: $0 {enter OWNER_PID|restore [OWNER_PID]|status}" >&2
    exit 2
    ;;
esac
