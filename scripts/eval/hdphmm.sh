#!/usr/bin/env bash
# Sticky / SDHDP-HMM baseline. Requires `uv sync --extra baselines`.
set -euo pipefail
exec uv run tcc eval hdphmm "$@"
