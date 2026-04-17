"""Scientific quality gate tests for embedding quality.

These tests verify that the trained model meets minimum embedding quality
thresholds required for the thesis methodology to be valid.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    reconstruction_mse,
)
from tcc_itransformer.model.losses import naive_baseline_loss, reconstruction_loss


@pytest.mark.quality
class TestReconstructionBeatsBaseline:
    def test_reconstruction_beats_baseline(self, trained_model_and_data: dict) -> None:
        """Model reconstruction MSE must be lower than the naive mean baseline."""
        model = trained_model_and_data["model"]
        test_loader = trained_model_and_data["test_loader"]
        train_mean = trained_model_and_data["train_mean"]
        device = trained_model_and_data["device"]

        model.eval()
        model_mses: list[float] = []
        baseline_mses: list[float] = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                x_hat, _ = model(x)
                model_mses.append(reconstruction_loss(x, x_hat).item())
                baseline_mses.append(naive_baseline_loss(x, train_mean).item())

        model_mse = float(np.mean(model_mses))
        baseline_mse = float(np.mean(baseline_mses))

        assert model_mse < baseline_mse, (
            f"Model MSE ({model_mse:.6f}) not better than naive baseline ({baseline_mse:.6f})"
        )


@pytest.mark.quality
class TestNoEmbeddingCollapse:
    def test_no_embedding_collapse(self, trained_model_and_data: dict) -> None:
        """All embedding dimensions must have variance > 1e-4 (no collapse)."""
        train_emb = trained_model_and_data["train_emb"]
        collapse_info = check_embedding_collapse(train_emb, threshold=1e-4)

        assert not collapse_info["is_collapsed"], (
            f"Embedding collapse detected in dims: {collapse_info['collapsed_dims']}. "
            f"Per-dim variances: {collapse_info['per_dim_variance']}"
        )


@pytest.mark.quality
class TestEffectiveRankAbove2:
    def test_effective_rank_above_2(self, trained_model_and_data: dict) -> None:
        """Effective rank of embeddings must be > 2.0 to avoid trivial solutions."""
        train_emb = trained_model_and_data["train_emb"]
        eff_rank = compute_effective_rank(train_emb)

        assert eff_rank > 2.0, (
            f"Effective rank ({eff_rank:.2f}) is too low. "
            "Embeddings may be nearly degenerate."
        )


@pytest.mark.quality
class TestPCAVarianceExplained:
    def test_pca_variance_explained(self, trained_model_and_data: dict) -> None:
        """Adaptive PCA on train embeddings must explain > 90% variance."""
        from tcc_itransformer.evaluation.clustering import fit_adaptive_pca
        from tcc_itransformer.evaluation.effective_sample_size import (
            extract_non_overlapping_indices,
        )

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
        var_explained = float(np.sum(pca.explained_variance_ratio_))

        assert var_explained >= 0.9, (
            f"PCA explains only {var_explained:.1%} variance with {n_pca} components. "
            "Expected >= 90%."
        )
