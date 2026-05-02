#!/usr/bin/env bash
# Train one config end-to-end with full evaluation.
set -euo pipefail
exec uv run tcc train single "$@"
