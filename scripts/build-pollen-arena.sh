#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

arena_app="${LAB_ROOT}/upstream/microduck-simulator/app"
if [[ ! -f "${arena_app}/package-lock.json" ]]; then
    echo "Pinned Pollen simulator submodule is missing. Run git submodule update --init --recursive." >&2
    exit 1
fi

cd "${arena_app}"
npm ci
npm run build
echo "Pollen factory playground built at ${arena_app}/dist"
