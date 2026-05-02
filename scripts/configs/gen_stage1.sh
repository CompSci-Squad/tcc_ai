#!/usr/bin/env bash
# Generate stage-1 (LR x dropout) sweep YAMLs.
set -euo pipefail
exec uv run tcc configs gen-stage1 "$@"
