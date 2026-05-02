#!/usr/bin/env bash
# Train one model per (W, d) and evaluate every K post-hoc.
set -euo pipefail
exec uv run tcc train sweep "$@"
