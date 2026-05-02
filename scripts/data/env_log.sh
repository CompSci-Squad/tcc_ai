#!/usr/bin/env bash
# Snapshot interpreter, GPU, package versions to docs/environment.json.
set -euo pipefail
exec uv run tcc data env-log "$@"
