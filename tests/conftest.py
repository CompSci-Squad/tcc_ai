"""Shared test fixtures for the iTransformer autoencoder test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tcc_itransformer.config import ExperimentConfig


@pytest.fixture()
def mock_panel_data() -> pd.DataFrame:
    """Random (100, 20) DataFrame simulating FRED-MD monthly data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    columns = [f"series_{i:02d}" for i in range(20)]
    return pd.DataFrame(rng.standard_normal((100, 20)), index=dates, columns=columns)


@pytest.fixture()
def mock_windows() -> torch.Tensor:
    """Random (30, 12, 20) tensor simulating windowed FRED-MD data."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(30, 12, 20, generator=gen)


@pytest.fixture()
def mock_embeddings() -> np.ndarray:
    """Random (30, 8) numpy array simulating latent embeddings."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 8))


@pytest.fixture()
def mock_labels() -> np.ndarray:
    """Random integer labels (30,) with 3 clusters."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 3, size=30)


@pytest.fixture()
def default_config() -> ExperimentConfig:
    """ExperimentConfig with all defaults."""
    return ExperimentConfig()


@pytest.fixture()
def tmp_results_dir(tmp_path: pytest.TempPathFactory) -> Path:
    """Temporary directory for MLflow results."""
    results = tmp_path / "results"
    results.mkdir()
    return results


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "slow: marks slow tests")
    config.addinivalue_line("markers", "quality: marks scientific quality gate tests")


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
