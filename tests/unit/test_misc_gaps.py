"""Misc unit tests covering remaining gaps:
- Config validators (n_heads must divide d_model, latent_dim <= d_model)
- compute_sha256 helper
- setup_mlflow + log_config (file:// backend)
- iTransformerAE backprop populates parameter gradients
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from tcc_itransformer.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_n_heads_must_divide_d_model(self) -> None:
        with pytest.raises(ValueError, match="must divide d_model"):
            ExperimentConfig(d_model=64, n_heads=5)

    def test_latent_dim_le_d_model(self) -> None:
        with pytest.raises(ValueError, match="latent_dim"):
            ExperimentConfig(d_model=32, n_heads=4, latent_dim=64)


# ---------------------------------------------------------------------------
# sha256 helper used by the data-download pipeline.
# ---------------------------------------------------------------------------

class TestComputeSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        from tcc_itransformer.pipelines.data_download import compute_sha256

        payload = b"hello FRED-MD\n"
        f = tmp_path / "x.csv"
        f.write_bytes(payload)
        assert compute_sha256(f) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# MLflow setup with file:// backend
# ---------------------------------------------------------------------------

class TestMlflowUtils:
    def test_setup_mlflow_creates_experiment(self, tmp_path: Path) -> None:
        import mlflow

        from tcc_itransformer.tracking.mlflow_utils import setup_mlflow

        uri = f"file:{tmp_path / 'mlruns'}"
        exp_id = setup_mlflow(uri, "unit_test_exp")
        assert exp_id
        exp = mlflow.get_experiment(exp_id)
        assert exp.name == "unit_test_exp"

    def test_log_config_inside_run(self, tmp_path: Path) -> None:
        import mlflow

        from tcc_itransformer.tracking.mlflow_utils import log_config, setup_mlflow

        uri = f"file:{tmp_path / 'mlruns'}"
        exp_id = setup_mlflow(uri, "log_cfg_exp")
        cfg = ExperimentConfig()
        with mlflow.start_run(experiment_id=exp_id):
            log_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# Model gradient flow
# ---------------------------------------------------------------------------

class TestModelGradients:
    def test_param_gradients_populated_after_backward(self) -> None:
        from tcc_itransformer.model.autoencoder import iTransformerAE

        model = iTransformerAE(
            n_series=5,
            window_size=6,
            d_model=16,
            n_heads=2,
            n_layers=1,
            latent_dim=4,
        )
        x = torch.randn(2, 6, 5)
        recon, _emb = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()

        named = list(model.named_parameters())
        assert named, "Model exposes no parameters"
        n_with_grad = sum(1 for _n, p in named if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_with_grad == len(named), (
            f"Only {n_with_grad}/{len(named)} parameters received non-zero grads"
        )
