#!/usr/bin/env bash
set -euo pipefail

lab_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${lab_root}"
python3 -m unittest tests.test_gamepad_bridge -v
