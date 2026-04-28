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
    compute_regime_transitions,
    fit_adaptive_pca,
    fit_kmeans,
    select_k,
    select_k_combined,
)
from tcc_itransformer.evaluation.density_clustering import (
    HDBSCANResult,
    fit_hdbscan,
    optimize_hdbscan_dbcv,
)
from tcc_itransformer.evaluation.dim_reduction import (
    UMAPConfig,
    apply_umap,
    fit_umap,
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
from tcc_itransformer.evaluation.explain import (
    FeatureExplanation,
    RegimeExplanation,
    explain_assignment,
    explanations_to_frame,
)
from tcc_itransformer.evaluation.regime_validation import (
    CANONICAL_CRISIS_WINDOWS,
    NBEROverlapResult,
    bai_perron_alignment,
    crisis_window_coverage,
    nber_overlap,
    regime_conditional_moments,
    regime_durations,
    transition_matrix,
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
    # clustering (PCA + K-Means baseline)
    "apply_pca",
    "clustering_stability",
    "compute_clustering_metrics",
    "compute_regime_transitions",
    "fit_adaptive_pca",
    "fit_kmeans",
    "select_k",
    "select_k_combined",
    # density_clustering (HDBSCAN — principal)
    "HDBSCANResult",
    "fit_hdbscan",
    "optimize_hdbscan_dbcv",
    # dim_reduction (UMAP — principal)
    "UMAPConfig",
    "apply_umap",
    "fit_umap",
    # effective_sample_size
    "compute_effective_n",
    "extract_non_overlapping_indices",
    # embedding_quality
    "check_embedding_collapse",
    "compute_effective_rank",
    "compute_isotropy",
    "reconstruction_mse",
    # explain (Module 4)
    "FeatureExplanation",
    "RegimeExplanation",
    "explain_assignment",
    "explanations_to_frame",
    # regime_validation (NBER, Bai-Perron, moments, transitions)
    "CANONICAL_CRISIS_WINDOWS",
    "NBEROverlapResult",
    "bai_perron_alignment",
    "crisis_window_coverage",
    "nber_overlap",
    "regime_conditional_moments",
    "regime_durations",
    "transition_matrix",
    # statistical_tests
    "kruskal_wallis_per_dim",
    "moving_block_bootstrap",
    "pairwise_mann_whitney",
    "permutation_test_silhouette",
]
