#!/usr/bin/env bash
# A1 confound check: NBER, pre-2008, |dINDPRO| chi-square + ARI w/o 2020-Q2.
set -euo pipefail
exec uv run tcc eval confound "$@"
