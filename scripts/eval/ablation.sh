#!/usr/bin/env bash
# {PCA,UMAP,t-SNE} x {KMeans,HDBSCAN} ablation on cached embeddings.
set -euo pipefail
exec uv run tcc eval ablation "$@"
