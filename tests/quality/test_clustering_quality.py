"""Scientific quality gate tests for clustering quality.

These tests verify that the clustering pipeline produces meaningful
and stable results on the trained model's embeddings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    clustering_stability,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
    select_k,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.statistical_tests import kruskal_wallis_per_dim


@pytest.fixture()
def pca_embeddings_and_labels(trained_model_and_data: dict) -> dict:
    """Prepare PCA-reduced embeddings and K-Means labels for clustering tests."""
    train_emb = trained_model_and_data["train_emb"]
    config = trained_model_and_data["config"]

    non_overlap_idx = extract_non_overlapping_indices(
        n_windows=len(train_emb), window_size=config.window_size,
    )
    train_emb_no = train_emb[non_overlap_idx]

    pca, n_pca = fit_adaptive_pca(
        train_emb_no,
        config.latent_dim,
        variance_threshold=config.pca_variance_threshold,
        n_max=config.n_pca_max,
    )
    train_pca = apply_pca(train_emb_no, pca)

    km = fit_kmeans(train_pca, config.n_clusters, random_state=config.seed)
    labels = km.labels_

    return {
        "train_pca": train_pca,
        "labels": labels,
        "n_pca": n_pca,
        "config": config,
    }


@pytest.mark.quality
class TestSilhouetteAboveZero:
    def test_silhouette_above_zero(self, pca_embeddings_and_labels: dict) -> None:
        """Silhouette score must be > 0 (clusters better than random)."""
        metrics = compute_clustering_metrics(
            pca_embeddings_and_labels["train_pca"],
            pca_embeddings_and_labels["labels"],
        )
        assert metrics["silhouette"] > 0.0, (
            f"Silhouette score ({metrics['silhouette']:.4f}) is not positive. "
            "Clusters are not better than random assignment."
        )


@pytest.mark.quality
class TestKWSignificantDimensions:
    def test_kw_significant_dimensions(self, pca_embeddings_and_labels: dict) -> None:
        """At least ceil(d/2) PCA dimensions must be significant after BH correction."""
        data = pca_embeddings_and_labels
        n_pca = data["n_pca"]
        min_significant = math.ceil(n_pca / 2)

        kw = kruskal_wallis_per_dim(data["train_pca"], data["labels"])

        assert kw["n_significant"] >= min_significant, (
            f"Only {kw['n_significant']}/{n_pca} PCA dims significant after BH "
            f"(need >= {min_significant}). "
            f"p-values: {kw['p_values_corrected']}"
        )


@pytest.mark.quality
class TestClusteringStability:
    def test_clustering_stability(self, pca_embeddings_and_labels: dict) -> None:
        """Clustering stability (mean ARI across 5 re-initializations) must be > 0.7."""
        data = pca_embeddings_and_labels
        ari = clustering_stability(
            data["train_pca"],
            k=data["config"].n_clusters,
            n_runs=5,
            random_state=data["config"].seed,
        )
        assert ari > 0.7, (
            f"Clustering stability ARI ({ari:.4f}) is below threshold 0.7. "
            "K-Means results are not stable across random initializations."
        )


@pytest.mark.quality
class TestValidKRange:
    def test_valid_k_range(self, pca_embeddings_and_labels: dict) -> None:
        """Best K from silhouette selection must be in {3, 4, 5}."""
        data = pca_embeddings_and_labels
        result = select_k(data["train_pca"], k_range=[3, 4, 5])
        assert result["best_k"] in {3, 4, 5}, (
            f"Best K ({result['best_k']}) is outside valid range {{3, 4, 5}}. "
            f"Silhouette scores: {result['scores']}"
        )
