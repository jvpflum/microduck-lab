#!/usr/bin/env bash
set -euo pipefail

# Coordinates long GPU training with the local Docker-backed vLLM service.
# The marker is also understood by the Hermes vLLM watchdog, preventing it
# from undoing an intentional training-priority pause.

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="${DUCKLAB_RESOURCE_MARKER:-${LAB_ROOT}/policy-bench/training-priority.json}"
CONTAINER="${DUCKLAB_VLLM_CONTAINER:-qwen38-hermes-vllm}"
VLLM_URL="${DUCKLAB_VLLM_URL:-http://127.0.0.1:8000/v1/models}"
VLLM_LOCK="${DUCKLAB_VLLM_LOCK:-/home/ducklab-user/.hermes/run/vllm-launch.lock}"

docker_cmd() {
  sudo -n docker "$@"
}

marker_pid() {
  sed -n 's/.*"owner_pid": *\([0-9][0-9]*\).*/\1/p' "${MARKER}" 2>/dev/null | head -n 1
}

marker_was_running() {
  sed -n 's/.*"vllm_was_running": *\(true\|false\).*/\1/p' "${MARKER}" 2>/dev/null | head -n 1
}

enter_priority() {
  local owner_pid="$1"
  local existing_pid=""
  local was_running=false

  mkdir -p "$(dirname "${MARKER}")" "$(dirname "${VLLM_LOCK}")"
  if [[ -f "${MARKER}" ]]; then
    existing_pid="$(marker_pid)"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Training priority is already owned by process ${existing_pid}." >&2
      exit 1
    fi
    echo "Recovering stale training-priority marker."
    rm -f "${MARKER}"
  fi

  # Validate non-interactive Docker access before recording or changing state.
  docker_cmd info >/dev/null
  if docker_cmd inspect "${CONTAINER}" >/dev/null 2>&1; then
    if [[ "$(docker_cmd inspect --format '{{.State.Running}}' "${CONTAINER}")" == "true" ]]; then
      was_running=true
    fi
  fi

  printf '{"owner_pid": %s, "vllm_container": "%s", "vllm_was_running": %s, "started_at": "%s"}\n' \
    "${owner_pid}" "${CONTAINER}" "${was_running}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${MARKER}"

  if [[ "${was_running}" == "true" ]]; then
    exec 9>"${VLLM_LOCK}"
    flock -w 30 9
    echo "Training priority: pausing ${CONTAINER}; local Hermes inference will be unavailable."
    if ! docker_cmd update --restart=no "${CONTAINER}" >/dev/null \
      || ! docker_cmd stop --time 45 "${CONTAINER}" >/dev/null; then
      docker_cmd update --restart=unless-stopped "${CONTAINER}" >/dev/null 2>&1 || true
      rm -f "${MARKER}"
      echo "Could not pause vLLM safely; training was not started." >&2
      exit 1
    fi
  else
    echo "Training priority: vLLM was already offline; leaving it offline."
  fi
}

restore_priority() {
  local owner_pid="${1:-}"
  local recorded_pid=""
  local was_running=false

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
  was_running="$(marker_was_running)"

  if [[ "${was_running}" == "true" ]]; then
    exec 9>"${VLLM_LOCK}"
    flock -w 30 9
    echo "Training finished: restoring ${CONTAINER}."
    docker_cmd update --restart=unless-stopped "${CONTAINER}" >/dev/null
    docker_cmd start "${CONTAINER}" >/dev/null
    for _attempt in $(seq 1 48); do
      if curl -fsS --connect-timeout 3 "${VLLM_URL}" >/dev/null 2>&1; then
        echo "vLLM is healthy again."
        rm -f "${MARKER}"
        /home/ducklab-user/.hermes/scripts/hermes-health-verify.sh || \
          echo "WARNING: Hermes final health verification reported an issue." >&2
        return 0
      fi
      sleep 10
    done
    echo "WARNING: vLLM did not become ready within eight minutes; marker retained at ${MARKER}." >&2
    return 1
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
      printf 'training-priority owner=%s vllm_was_running=%s\n' "$(marker_pid)" "$(marker_was_running)"
    else
      echo "shared"
    fi
    ;;
  *)
    echo "Usage: $0 {enter OWNER_PID|restore [OWNER_PID]|status}" >&2
    exit 2
    ;;
esac
