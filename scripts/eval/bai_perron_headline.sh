#!/usr/bin/env bash
# Phase C3: Bai-Perron break agreement on INDPRO, PAYEMS, UNRATE, T10Y3M.
# Requires: export FRED_API_KEY=<your_key>
set -euo pipefail
exec uv run tcc eval bai-perron-headline "$@"
