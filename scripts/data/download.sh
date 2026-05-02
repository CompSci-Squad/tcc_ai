#!/usr/bin/env bash
# Download FRED-MD vintage CSV + SHA256.
set -euo pipefail
exec uv run tcc data download "$@"
