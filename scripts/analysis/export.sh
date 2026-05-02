#!/usr/bin/env bash
# Export MLflow runs to LaTeX tables.
set -euo pipefail
exec uv run tcc analysis export "$@"
