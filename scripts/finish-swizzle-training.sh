#!/usr/bin/env bash
set -u

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 TRAINING_PID" >&2
    exit 2
fi

training_pid="$1"
lab_root="$(cd "$(dirname "$0")/.." && pwd)"

while kill -0 "${training_pid}" 2>/dev/null; do
    sleep 30
done

exec "${lab_root}/scripts/finalize-training.sh" swizzle
