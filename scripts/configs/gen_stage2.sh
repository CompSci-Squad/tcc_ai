#!/usr/bin/env bash
# Generate stage-2 (W x d x K) sweep YAMLs from a frozen stage-1 winner.
set -euo pipefail
exec uv run tcc configs gen-stage2 "$@"
