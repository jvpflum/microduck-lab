#!/usr/bin/env bash
# macOS companion for DuckLab Policy Bench.
#
# Keeps the complete dashboard/viewer SSH tunnel alive and opens new viewer
# sessions in the local browser when Codex launches them on the Spark.

set -euo pipefail

readonly DASHBOARD_URL="http://127.0.0.1:8091"
readonly STATE_DIR="${HOME}/Library/Application Support/DuckLab"
readonly CONTROL_SOCKET="${STATE_DIR}/ssh-control"
readonly WATCHER_PID_FILE="${STATE_DIR}/viewer-watcher.pid"
readonly TARGET_FILE="${STATE_DIR}/ssh-target"
readonly SEEN_FILE="${STATE_DIR}/seen-viewers"
readonly LOG_FILE="${STATE_DIR}/companion.log"

readonly -a FORWARD_SPECS=(
  127.0.0.1:8091:127.0.0.1:8091
  127.0.0.1:8080:127.0.0.1:8080
  127.0.0.1:8090:127.0.0.1:8090
  127.0.0.1:8081:127.0.0.1:8081
  127.0.0.1:8092:127.0.0.1:8092
  127.0.0.1:8082:127.0.0.1:8082
  127.0.0.1:8093:127.0.0.1:8093
  127.0.0.1:8083:127.0.0.1:8083
  127.0.0.1:8094:127.0.0.1:8094
  127.0.0.1:8084:127.0.0.1:8084
  127.0.0.1:8095:127.0.0.1:8095
  127.0.0.1:8085:127.0.0.1:8085
  127.0.0.1:8096:127.0.0.1:8096
)

usage() {
  cat <<'EOF'
Usage:
  ducklab start <ssh-target>   Start/reuse the tunnel and open Policy Bench
  ducklab stop                 Stop the local tunnel and viewer watcher
  ducklab status               Show companion, tunnel, and dashboard status

Example:
  ducklab start <ssh-user>@<spark-address>

Leave the companion running. When Codex launches a Policy Bench viewer, macOS
will open it automatically in your default browser.
EOF
}

require_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This companion must run on the Mac, not on the Spark." >&2
    exit 2
  fi
}

validate_target() {
  local target="$1"
  if [[ -z "${target}" || "${target}" == -* || ! "${target}" =~ ^[A-Za-z0-9_.@:-]+$ ]]; then
    echo "Invalid SSH target: ${target}" >&2
    exit 2
  fi
}

saved_target() {
  if [[ -f "${TARGET_FILE}" ]]; then
    tr -d '\r\n' <"${TARGET_FILE}"
  fi
}

tunnel_alive() {
  local target="$1"
  ssh -S "${CONTROL_SOCKET}" -O check "${target}" >/dev/null 2>&1
}

ensure_local_forwards() {
  local target="$1"
  local spec bind_address local_port remote_address remote_port
  for spec in "${FORWARD_SPECS[@]}"; do
    IFS=: read -r bind_address local_port remote_address remote_port <<<"${spec}"
    if ! /usr/bin/nc -z "${bind_address}" "${local_port}" >/dev/null 2>&1; then
      ssh -S "${CONTROL_SOCKET}" -O forward -L "${spec}" "${target}" >/dev/null 2>&1 || true
    fi
  done
}

watcher_alive() {
  [[ -f "${WATCHER_PID_FILE}" ]] || return 1
  local pid
  pid="$(tr -d '\r\n' <"${WATCHER_PID_FILE}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" >/dev/null 2>&1
}

ensure_remote_dashboard() {
  local target="$1"
  ssh -S "${CONTROL_SOCKET}" "${target}" \
    "cd \"\$HOME/projects/microduck-lab\" && if ! curl -fsS http://127.0.0.1:8091/api/status >/dev/null 2>&1; then nohup ./scripts/serve-policy-bench.sh >reports/policy-bench-server.log 2>&1 </dev/null & fi"
}

wait_for_dashboard() {
  local attempt
  for ((attempt = 1; attempt <= 30; attempt++)); do
    if curl -fsS "${DASHBOARD_URL}/api/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "The SSH tunnel is up, but Policy Bench did not become ready within 30 seconds." >&2
  return 1
}

watch_viewers() {
  local target
  mkdir -p "${STATE_DIR}"
  touch "${SEEN_FILE}"
  target="$(saved_target)"
  while true; do
    local status urls url
    if [[ -n "${target}" ]] && tunnel_alive "${target}"; then
      # This also makes migration painless when an older hand-written tunnel is
      # still serving one of the ports: occupied ports are reused, and the
      # companion claims them automatically if that older SSH session closes.
      ensure_local_forwards "${target}"
    fi
    status="$(curl -fsS "${DASHBOARD_URL}/api/status" 2>/dev/null || true)"
    urls="$(printf '%s' "${status}" | tr ',' '\n' | sed -n 's/.*"open_url": "\([^"]*\)".*/\1/p')"
    while IFS= read -r url; do
      [[ -n "${url}" ]] || continue
      if ! grep -Fqx "${url}" "${SEEN_FILE}"; then
        printf '%s\n' "${url}" >>"${SEEN_FILE}"
        /usr/bin/open "${url}"
      fi
    done <<<"${urls}"
    sleep 2
  done
}

start_companion() {
  local target="$1"
  local previous_target
  validate_target "${target}"
  mkdir -p "${STATE_DIR}"
  previous_target="$(saved_target)"
  if [[ -n "${previous_target}" && "${previous_target}" != "${target}" ]] && \
      tunnel_alive "${previous_target}"; then
    ssh -S "${CONTROL_SOCKET}" -O exit "${previous_target}" >/dev/null 2>&1 || true
  fi
  printf '%s\n' "${target}" >"${TARGET_FILE}"

  if ! tunnel_alive "${target}"; then
    # Remove only this companion's stale control socket. It is never a user SSH
    # key or configuration file.
    [[ ! -S "${CONTROL_SOCKET}" ]] || rm -f "${CONTROL_SOCKET}"
    ssh -M -S "${CONTROL_SOCKET}" -fNT \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      "${target}"
  fi

  ensure_local_forwards "${target}"
  ensure_remote_dashboard "${target}"
  wait_for_dashboard

  if ! watcher_alive; then
    : >"${SEEN_FILE}"
    nohup "$0" watch-viewers >>"${LOG_FILE}" 2>&1 &
    printf '%s\n' "$!" >"${WATCHER_PID_FILE}"
  fi

  /usr/bin/open "${DASHBOARD_URL}"
  echo "DuckLab is connected. Dashboard and new viewers will open automatically."
}

stop_companion() {
  local target pid
  target="$(saved_target)"
  if watcher_alive; then
    pid="$(tr -d '\r\n' <"${WATCHER_PID_FILE}")"
    kill "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${WATCHER_PID_FILE}"
  if [[ -n "${target}" ]] && tunnel_alive "${target}"; then
    ssh -S "${CONTROL_SOCKET}" -O exit "${target}" >/dev/null 2>&1 || true
  fi
  echo "DuckLab companion stopped."
}

show_status() {
  local target
  target="$(saved_target)"
  if [[ -n "${target}" ]] && tunnel_alive "${target}"; then
    echo "SSH tunnel: running (${target})"
  else
    echo "SSH tunnel: stopped"
  fi
  if watcher_alive; then
    echo "Browser watcher: running"
  else
    echo "Browser watcher: stopped"
  fi
  if curl -fsS "${DASHBOARD_URL}/api/status" >/dev/null 2>&1; then
    echo "Policy Bench: ${DASHBOARD_URL}"
  else
    echo "Policy Bench: unavailable"
  fi
}

main() {
  require_macos
  local command="${1:-}"
  case "${command}" in
    start)
      [[ $# -eq 2 ]] || { usage >&2; exit 2; }
      start_companion "$2"
      ;;
    stop)
      [[ $# -eq 1 ]] || { usage >&2; exit 2; }
      stop_companion
      ;;
    status)
      [[ $# -eq 1 ]] || { usage >&2; exit 2; }
      show_status
      ;;
    watch-viewers)
      watch_viewers
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
