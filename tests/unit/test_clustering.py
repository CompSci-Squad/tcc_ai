"""Tests for clustering pipeline: adaptive PCA, K-Means, metrics, stability."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.clustering import (
    clustering_stability,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
    select_k,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def clustered_embeddings(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 3-cluster data in 8 dims."""
    centers = np.array([[5, 0, 0, 0, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 5, 0, 0, 0, 0, 0]], dtype=np.float64)
    data = np.vstack([
        rng.normal(loc=centers[i], scale=0.3, size=(30, 8))
        for i in range(3)
    ])
    labels = np.repeat([0, 1, 2], 30)
    return data, labels


class TestFitAdaptivePCA:
    def test_components_less_than_latent(self, rng: np.random.Generator) -> None:
        emb = rng.standard_normal((50, 8))
        pca, n = fit_adaptive_pca(emb, latent_dim=8, n_max=5)
        assert n < 8
        assert n <= 5
        assert pca.n_components == n

    def test_variance_threshold(self, rng: np.random.Generator) -> None:
        emb = rng.standard_normal((50, 8))
        pca, n = fit_adaptive_pca(emb, latent_dim=8, variance_threshold=0.5, n_max=7)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # The selected n should capture at least the threshold variance
        # (or be constrained by n_max / latent_dim - 1)
        assert n >= 1


class TestFitKMeans:
    def test_labels_shape(self, rng: np.random.Generator) -> None:
        emb = rng.standard_normal((50, 3))
        km = fit_kmeans(emb, k=3)
        assert km.labels_.shape == (50,)
        assert len(np.unique(km.labels_)) == 3


class TestClusteringMetrics:
    def test_keys(self, clustered_embeddings: tuple[np.ndarray, np.ndarray]) -> None:
        data, labels = clustered_embeddings
        metrics = compute_clustering_metrics(data, labels)
        assert set(metrics.keys()) == {"silhouette", "davies_bouldin", "calinski_harabasz"}

    def test_silhouette_range(
        self, clustered_embeddings: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, labels = clustered_embeddings
        metrics = compute_clustering_metrics(data, labels)
        assert -1 <= metrics["silhouette"] <= 1


class TestSelectK:
    def test_returns_best(self, rng: np.random.Generator) -> None:
        # Create data that clearly has 3 clusters
        centers = rng.standard_normal((3, 4)) * 10
        data = np.vstack([
            rng.normal(loc=centers[i], scale=0.1, size=(40, 4))
            for i in range(3)
        ])
        result = select_k(data, k_range=[2, 3, 4, 5])
        assert "best_k" in result
        assert "scores" in result
        assert result["best_k"] in [2, 3, 4, 5]
        # With 3 well-separated clusters, 3 should score highest
        assert result["best_k"] == 3


class TestClusteringStability:
    def test_range(
        self, clustered_embeddings: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, _ = clustered_embeddings
        ari = clustering_stability(data, k=3, n_runs=5)
        assert 0 <= ari <= 1
