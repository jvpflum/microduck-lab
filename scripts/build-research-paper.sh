#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_uv

source_path="${LAB_ROOT}/docs/DUCKWING_RESEARCH_PAPER.md"
output_path="${LAB_ROOT}/docs/DUCKWING_RESEARCH_PAPER.docx"
cd "${UPSTREAM_DIR}"
exec "${UV_BIN}" run python "${LAB_ROOT}/tools/render_research_paper.py" \
    "${source_path}" "${output_path}"
