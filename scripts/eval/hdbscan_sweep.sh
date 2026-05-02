#!/usr/bin/env bash
# A3 grid sweep over UMAP / t-SNE x HDBSCAN params.
set -euo pipefail
exec uv run tcc eval hdbscan-sweep "$@"
