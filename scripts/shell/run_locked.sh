#!/usr/bin/env bash
# Run the locked Phase C evaluation
# Usage: bash scripts/shell/run_locked.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Load env vars from tcc.env (skipping export lines that cause quoting issues)
if [ -f "../tcc.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ../tcc.env 2>/dev/null || true
    set +a
fi

echo "Starting Phase C locked evaluation..."
exec uv run python scripts/run_phase_c_locked.py
