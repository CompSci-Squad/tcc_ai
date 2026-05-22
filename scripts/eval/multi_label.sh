#!/usr/bin/env bash
# Phase C1: Multi-label validation panel (Chauvet-Piger, Sahm, CFNAI-MA3, OECD CLI).
# Requires: uv sync --extra labels && export FRED_API_KEY=<your_key>
set -euo pipefail
exec uv run tcc eval multi-label "$@"
