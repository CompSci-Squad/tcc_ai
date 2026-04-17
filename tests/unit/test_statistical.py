"""Tests for statistical tests: KW, Mann-Whitney, permutation, block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.statistical_tests import (
    kruskal_wallis_per_dim,
    moving_block_bootstrap,
    pairwise_mann_whitney,
    permutation_test_silhouette,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def separated_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated 3-cluster data in 4 dims."""
    data = np.vstack([
        rng.normal(loc=10, scale=0.1, size=(30, 4)),
        rng.normal(loc=0, scale=0.1, size=(30, 4)),
        rng.normal(loc=-10, scale=0.1, size=(30, 4)),
    ])
    labels = np.repeat([0, 1, 2], 30)
    return data, labels


@pytest.fixture()
def random_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random data with no cluster structure in 4 dims."""
    data = rng.standard_normal((90, 4))
    labels = np.repeat([0, 1, 2], 30)
    return data, labels


class TestKruskalWallisPerDim:
    def test_output_shape(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, labels = separated_data
        result = kruskal_wallis_per_dim(data, labels)
        assert result["h_stats"].shape == (4,)
        assert result["p_values_raw"].shape == (4,)
        assert result["p_values_corrected"].shape == (4,)
        assert result["rejected"].shape == (4,)
        assert result["effect_sizes"].shape == (4,)

    def test_significant_on_separated(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, labels = separated_data
        result = kruskal_wallis_per_dim(data, labels)
        # Well-separated clusters should yield significant results
        assert result["n_significant"] == 4

    def test_effect_size_range(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, labels = separated_data
        result = kruskal_wallis_per_dim(data, labels)
        assert np.all(result["effect_sizes"] >= 0)
        assert np.all(result["effect_sizes"] <= 1)

    def test_bh_correction_less_strict_than_bonferroni(
        self, rng: np.random.Generator
    ) -> None:
        """BH should reject >= what Bonferroni rejects (BH is less strict)."""
        # Create marginal data: some dimensions significant, some not
        data = np.hstack([
            np.vstack([
                rng.normal(loc=2, scale=1, size=(20, 2)),
                rng.normal(loc=-2, scale=1, size=(20, 2)),
                rng.normal(loc=0, scale=1, size=(20, 2)),
            ]),
            rng.standard_normal((60, 2)),  # noise dims
        ])
        labels = np.repeat([0, 1, 2], 20)
        result = kruskal_wallis_per_dim(data, labels)
        n_bh = result["n_significant"]
        # Bonferroni: apply manual correction
        n_bonf = int(np.sum(result["p_values_raw"] * 4 < 0.05))
        assert n_bh >= n_bonf


class TestPairwiseMannWhitney:
    def test_shape(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        data, labels = separated_data
        result = pairwise_mann_whitney(data, labels)
        n_pairs = len(result["pairs"])
        assert n_pairs == 3  # C(3,2) = 3
        assert result["u_stats"].shape == (3, 4)
        assert result["p_values_corrected"].shape == (3, 4)
        assert result["effect_sizes"].shape == (3, 4)


class TestPermutationTest:
    def test_random_data_not_significant(self, rng: np.random.Generator) -> None:
        """With random data, silhouette difference should not be significant."""
        from sklearn.cluster import KMeans

        emb = rng.standard_normal((60, 4))
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = km.fit_predict(emb)

        # Split into two halves
        emb_a, labels_a = emb[:30], labels[:30]
        emb_b, labels_b = emb[30:], labels[30:]

        result = permutation_test_silhouette(
            emb_a, labels_a, emb_b, labels_b, n_permutations=200
        )
        assert result["p_value"] > 0.05

    def test_separated_data_significant(self, rng: np.random.Generator) -> None:
        """Well-separated vs random should be significant."""
        from sklearn.cluster import KMeans

        # A: well-separated
        emb_a = np.vstack([
            rng.normal(loc=10, scale=0.1, size=(15, 4)),
            rng.normal(loc=-10, scale=0.1, size=(15, 4)),
            rng.normal(loc=0, scale=0.1, size=(15, 4)),
        ])
        labels_a = np.repeat([0, 1, 2], 15)

        # B: random
        emb_b = rng.standard_normal((45, 4))
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels_b = km.fit_predict(emb_b)

        result = permutation_test_silhouette(
            emb_a, labels_a, emb_b, labels_b, n_permutations=200
        )
        assert result["p_value"] < 0.05


class TestMovingBlockBootstrap:
    def test_ci_covers_mean(self, rng: np.random.Generator) -> None:
        data = rng.standard_normal(200)
        true_mean = float(np.mean(data))
        result = moving_block_bootstrap(
            statistic_fn=lambda x: float(np.mean(x)),
            data=data,
            block_length=5,
            n_bootstrap=2000,
            confidence_level=0.95,
        )
        assert result["ci_lower"] <= true_mean <= result["ci_upper"]
        assert result["bootstrap_distribution"].shape == (2000,)
