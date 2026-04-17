"""Unit tests for embedding quality metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
    reconstruction_mse,
)


class TestReconstructionMSE:
    def test_zero_for_identical(self) -> None:
        x = np.random.default_rng(42).standard_normal((20, 8))
        assert reconstruction_mse(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_different(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 8))
        x_hat = rng.standard_normal((20, 8))
        assert reconstruction_mse(x, x_hat) > 0.0

    def test_known_value(self) -> None:
        x = np.ones((2, 2))
        x_hat = np.zeros((2, 2))
        assert reconstruction_mse(x, x_hat) == pytest.approx(1.0)


class TestCheckEmbeddingCollapse:
    def test_no_collapse(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 8))
        result = check_embedding_collapse(emb)
        assert not result["is_collapsed"]
        assert result["n_collapsed"] == 0

    def test_detects_collapse(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 4))
        emb[:, 2] = 5.0  # constant column → variance 0
        result = check_embedding_collapse(emb)
        assert result["is_collapsed"]
        assert 2 in result["collapsed_dims"]

    def test_per_dim_variance_shape(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((30, 6))
        result = check_embedding_collapse(emb)
        assert result["per_dim_variance"].shape == (6,)


class TestComputeEffectiveRank:
    def test_positive(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 8))
        rank = compute_effective_rank(emb)
        assert rank > 0

    def test_high_for_full_rank(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((100, 4))
        rank = compute_effective_rank(emb)
        assert rank > 3.0  # nearly 4 for full-rank random

    def test_low_for_near_rank1(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.standard_normal((50, 1))
        emb = np.hstack([base, base * 1.01, base * 0.99, base * 1.005])
        rank = compute_effective_rank(emb)
        assert rank < 2.0


class TestComputeIsotropy:
    def test_range(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 8))
        iso = compute_isotropy(emb)
        assert -1.0 <= iso <= 1.0

    def test_isotropic_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((200, 50))
        iso = compute_isotropy(emb)
        assert abs(iso) < 0.3  # high-dim random → near-zero mean cosine
