#!/usr/bin/env bash
# Pull NBER USREC monthly recession indicator.
set -euo pipefail
exec uv run tcc data pull-nber "$@"
