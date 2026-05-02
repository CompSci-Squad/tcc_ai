#!/usr/bin/env bash
# Pick the stage-2 winner using the pre-registered tiebreak.
set -euo pipefail
exec uv run tcc winners stage2 "$@"
