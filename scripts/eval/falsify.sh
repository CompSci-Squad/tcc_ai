#!/usr/bin/env bash
# B2 falsification: linear AE / MLP AE / SVD encoders matched to d_lat.
set -euo pipefail
exec uv run tcc eval falsify "$@"
