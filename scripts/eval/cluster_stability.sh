#!/usr/bin/env bash
# Phase C4: Cluster stability bootstrap (Ben-Hur 2002 Jaccard).
set -euo pipefail
exec uv run tcc eval cluster-stability "$@"
