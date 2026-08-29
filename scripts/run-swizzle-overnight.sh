#!/usr/bin/env bash
set -euo pipefail

lab_root="$(cd "$(dirname "$0")/.." && pwd)"
exec "${lab_root}/scripts/train-swizzle.sh"
