#!/usr/bin/env bash
set -euo pipefail

lab_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${lab_root}"
python3 -m unittest -v \
    tests.test_agent_runs \
    tests.test_policy_bench \
    tests.test_policy_bench_server
