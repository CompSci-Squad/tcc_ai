#!/usr/bin/env bash
# Run the 4 baselines per config + locked panel CSV.
set -euo pipefail
exec uv run tcc eval baselines "$@"
