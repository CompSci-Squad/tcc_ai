"""Smoke tests for UMAP + HDBSCAN principal pipeline (pre_projeto §4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.density_clustering import (
    fit_hdbscan,
    optimize_hdbscan_dbcv,
)
from tcc_itransformer.evaluation.dim_reduction import (
    UMAPConfig,
    apply_umap,
    fit_umap,
)


@pytest.fixture()
def two_blob_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    blob1 = rng.standard_normal((60, 16)) + 5
    blob2 = rng.standard_normal((60, 16)) - 5
    return np.vstack([blob1, blob2])


def test_fit_umap_returns_low_dim(two_blob_data):
    reducer = fit_umap(two_blob_data, UMAPConfig(n_components=3, n_neighbors=10))
    z = apply_umap(two_blob_data, reducer)
    assert z.shape == (two_blob_data.shape[0], 3)


def test_hdbscan_recovers_two_clusters(two_blob_data):
    reducer = fit_umap(two_blob_data, UMAPConfig(n_components=3, n_neighbors=10))
    z = apply_umap(two_blob_data, reducer)
    res = fit_hdbscan(z, min_cluster_size=10)
    assert res.n_clusters >= 2
    assert 0.0 <= res.noise_fraction <= 1.0


def test_optimize_hdbscan_dbcv_returns_best(two_blob_data):
    reducer = fit_umap(two_blob_data, UMAPConfig(n_components=3, n_neighbors=10))
    z = apply_umap(two_blob_data, reducer)
    best, log = optimize_hdbscan_dbcv(
        z,
        min_cluster_sizes=(5, 10),
        min_samples_grid=(None, 5),
    )
    assert best.n_clusters >= 2
    assert len(log) > 0
    assert any(entry["valid"] for entry in log)
