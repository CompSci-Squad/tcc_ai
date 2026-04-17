"""Tests for baseline models: random, raw PCA, linear AE, windowed PCA."""

from __future__ import annotations

import numpy as np
import pytest

from tcc_itransformer.evaluation.baselines import (
    LinearAEBaseline,
    RandomEmbeddingBaseline,
    RawPCABaseline,
    WindowedPCABaseline,
    run_all_baselines,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def train_windows(rng: np.random.Generator) -> np.ndarray:
    """Synthetic (40, 12, 20) windows."""
    return rng.standard_normal((40, 12, 20))


@pytest.fixture()
def eval_windows(rng: np.random.Generator) -> np.ndarray:
    """Synthetic (20, 12, 20) windows."""
    return rng.standard_normal((20, 12, 20))


class TestRandomBaseline:
    def test_shape(self) -> None:
        b = RandomEmbeddingBaseline()
        emb = b.generate(50, 4)
        assert emb.shape == (50, 4)


class TestRawPCABaseline:
    def test_shape(self, rng: np.random.Generator) -> None:
        b = RawPCABaseline()
        train = rng.standard_normal((40, 20))
        emb_train = b.fit_transform(train, n_components=3)
        assert emb_train.shape == (40, 3)

        test = rng.standard_normal((10, 20))
        emb_test = b.transform(test)
        assert emb_test.shape == (10, 3)


class TestLinearAEBaseline:
    def test_trains_loss_decreases(self, rng: np.random.Generator) -> None:
        import torch
        import torch.nn as nn

        windows = rng.standard_normal((30, 12, 20))
        input_dim = 12 * 20
        b = LinearAEBaseline(input_dim, latent_dim=4, random_state=42)

        # Compute initial loss
        flat = windows.reshape(30, -1)
        x = torch.from_numpy(flat.astype(np.float32))
        with torch.no_grad():
            z = b.encoder(x)
            x_hat = b.decoder(z)
            loss_before = nn.MSELoss()(x_hat, x).item()

        b.fit(windows, n_epochs=50, lr=1e-3)

        with torch.no_grad():
            z = b.encoder(x)
            x_hat = b.decoder(z)
            loss_after = nn.MSELoss()(x_hat, x).item()

        assert loss_after < loss_before

    def test_transform_shape(self, rng: np.random.Generator) -> None:
        windows = rng.standard_normal((30, 12, 20))
        input_dim = 12 * 20
        b = LinearAEBaseline(input_dim, latent_dim=4)
        b.fit(windows, n_epochs=10)
        emb = b.transform(windows)
        assert emb.shape == (30, 4)


class TestWindowedPCABaseline:
    def test_shape(self, rng: np.random.Generator) -> None:
        windows = rng.standard_normal((40, 12, 20))
        b = WindowedPCABaseline()
        emb_train = b.fit_transform(windows, n_components=3)
        assert emb_train.shape == (40, 3)

        test = rng.standard_normal((10, 12, 20))
        emb_test = b.transform(test)
        assert emb_test.shape == (10, 3)


class TestRunAllBaselines:
    def test_keys(
        self,
        train_windows: np.ndarray,
        eval_windows: np.ndarray,
    ) -> None:
        results = run_all_baselines(
            train_windows, eval_windows, n_components=3, k=3
        )
        expected_keys = {"random", "raw_pca", "linear_ae", "windowed_pca"}
        assert set(results.keys()) == expected_keys
        for name, res in results.items():
            assert "embeddings" in res
            assert "labels" in res
            assert "silhouette" in res
            assert res["embeddings"].shape[0] == eval_windows.shape[0]
