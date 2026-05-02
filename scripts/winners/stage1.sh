#!/usr/bin/env bash
# Pick the stage-1 winner from completed SageMaker jobs.
set -euo pipefail
exec uv run tcc winners stage1 "$@"
