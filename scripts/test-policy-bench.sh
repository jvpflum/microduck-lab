#!/usr/bin/env bash
set -euo pipefail

lab_root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m unittest -v "${lab_root}/tests/test_policy_bench.py"
