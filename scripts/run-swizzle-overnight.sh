#!/usr/bin/env bash
set -u

lab_root="$(cd "$(dirname "$0")/.." && pwd)"

"${lab_root}/scripts/train-swizzle.sh" &
training_pid="$!"

"${lab_root}/scripts/finish-swizzle-training.sh" "${training_pid}"
finish_status="$?"
wait "${training_pid}"
training_status="$?"

if (( training_status != 0 )); then
    echo "Swizzle training failed with exit code ${training_status}." >&2
    exit "${training_status}"
fi
exit "${finish_status}"
