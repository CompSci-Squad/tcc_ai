"""Evaluation metrics, statistical tests, and clustering analysis."""

from __future__ import annotations

from tcc_itransformer.evaluation.baselines import (
    LinearAEBaseline,
    RandomEmbeddingBaseline,
    RawPCABaseline,
    WindowedPCABaseline,
    run_all_baselines,
)
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    clustering_stability,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
    select_k,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    compute_effective_n,
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
    reconstruction_mse,
)
from tcc_itransformer.evaluation.statistical_tests import (
    kruskal_wallis_per_dim,
    moving_block_bootstrap,
    pairwise_mann_whitney,
    permutation_test_silhouette,
)

__all__ = [
    # baselines
    "LinearAEBaseline",
    "RandomEmbeddingBaseline",
    "RawPCABaseline",
    "WindowedPCABaseline",
    "run_all_baselines",
    # clustering
    "apply_pca",
    "clustering_stability",
    "compute_clustering_metrics",
    "fit_adaptive_pca",
    "fit_kmeans",
    "select_k",
    # effective_sample_size
    "compute_effective_n",
    "extract_non_overlapping_indices",
    # embedding_quality
    "check_embedding_collapse",
    "compute_effective_rank",
    "compute_isotropy",
    "reconstruction_mse",
    # statistical_tests
    "kruskal_wallis_per_dim",
    "moving_block_bootstrap",
    "pairwise_mann_whitney",
    "permutation_test_silhouette",
]
