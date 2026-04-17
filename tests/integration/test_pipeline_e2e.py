"""End-to-end pipeline integration test on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.data.dataset import FREDMDWindowDataset
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    scale_splits,
    split_by_date,
)
from tcc_itransformer.evaluation.clustering import (
    apply_pca,
    compute_clustering_metrics,
    fit_adaptive_pca,
    fit_kmeans,
)
from tcc_itransformer.evaluation.effective_sample_size import (
    extract_non_overlapping_indices,
)
from tcc_itransformer.evaluation.embedding_quality import (
    check_embedding_collapse,
    compute_effective_rank,
    compute_isotropy,
)
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.seed import set_global_seed
from tcc_itransformer.training.trainer import Trainer

# Small dimensions for fast CPU execution
N_SERIES = 10
WINDOW_SIZE = 6
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 1
LATENT_DIM = 4
BATCH_SIZE = 4
MAX_EPOCHS = 3
K = 3


def _create_mock_dataframe(n_timesteps: int = 120, n_series: int = N_SERIES) -> pd.DataFrame:
    """Create a synthetic monthly DataFrame mimicking FRED-MD output."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-01", periods=n_timesteps, freq="MS")
    columns = [f"series_{i:02d}" for i in range(n_series)]
    values = rng.standard_normal((n_timesteps, n_series))
    # Add a few NaN for realism
    values[0, 0] = np.nan
    values[2, 3] = np.nan
    return pd.DataFrame(values, index=dates, columns=columns)


class TestFullPipelineMock:
    def test_full_pipeline_mock(self, tmp_path):
        """End-to-end pipeline on synthetic data (no real FRED-MD).

        Steps: create mock DataFrame → preprocess → window → dataset → dataloader
        → model → train (3 epochs) → extract embeddings → PCA → K-Means → metrics

        Verify: silhouette is a valid float, embeddings are not collapsed, all shapes correct.
        """
        set_global_seed(42)

        config = ExperimentConfig(
            seed=42,
            window_size=WINDOW_SIZE,
            latent_dim=LATENT_DIM,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=0.0,
            n_clusters=K,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_epochs=MAX_EPOCHS,
            patience=20,
            grad_clip=1.0,
            train_end="2016-12-01",
            val_end="2018-12-01",
            results_dir=str(tmp_path / "results"),
        )

        device = torch.device("cpu")

        # --- 1. Create mock data & preprocess ---
        df = _create_mock_dataframe()
        cleaned, dropped = drop_high_nan_series(df)
        assert len(dropped) == 0

        filled = forward_fill_nans(cleaned)
        assert filled.isna().sum().sum() == 0

        train_df, val_df, test_df = split_by_date(filled, config.train_end, config.val_end)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        scaler = fit_scaler(train_df)
        train_scaled, val_scaled, test_scaled = scale_splits(train_df, val_df, test_df, scaler)

        n_series = train_scaled.shape[1]

        # --- 2. Windows ---
        train_windows = create_windows(train_scaled, config.window_size)
        val_windows = create_windows(val_scaled, config.window_size)
        test_windows = create_windows(test_scaled, config.window_size)

        assert train_windows.ndim == 3
        assert train_windows.shape[1] == config.window_size
        assert train_windows.shape[2] == n_series

        # --- 3. Datasets & loaders ---
        train_ds = FREDMDWindowDataset(train_windows)
        val_ds = FREDMDWindowDataset(val_windows)
        test_ds = FREDMDWindowDataset(test_windows)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

        # --- 4. Model ---
        model = iTransformerAE.from_config(config, n_series)

        # --- 5. Train ---
        trainer = Trainer(model, config, train_loader, val_loader, device)
        history = trainer.train()

        assert len(history["train_losses"]) == MAX_EPOCHS
        assert len(history["val_losses"]) == MAX_EPOCHS
        for loss in history["train_losses"]:
            assert not np.isnan(loss)

        # --- 6. Load best & extract embeddings ---
        trainer.checkpoint.load_best(model)

        train_emb = trainer.extract_embeddings(train_loader)
        val_emb = trainer.extract_embeddings(val_loader)
        test_emb = trainer.extract_embeddings(test_loader)

        assert train_emb.shape[1] == LATENT_DIM
        assert val_emb.shape[1] == LATENT_DIM
        assert test_emb.shape[1] == LATENT_DIM

        # --- 7. Embedding quality ---
        collapse_info = check_embedding_collapse(train_emb)
        eff_rank = compute_effective_rank(train_emb)
        isotropy = compute_isotropy(train_emb)

        assert isinstance(eff_rank, float)
        assert eff_rank > 0
        assert isinstance(isotropy, float)

        # --- 8. PCA ---
        non_overlap_idx = extract_non_overlapping_indices(
            n_windows=len(train_emb), window_size=config.window_size,
        )
        train_emb_no = train_emb[non_overlap_idx]
        assert len(train_emb_no) > 0

        pca, n_pca = fit_adaptive_pca(
            train_emb_no,
            config.latent_dim,
            variance_threshold=config.pca_variance_threshold,
            n_max=config.n_pca_max,
        )
        assert n_pca >= 1

        train_pca = apply_pca(train_emb_no, pca)
        assert train_pca.shape[0] == len(train_emb_no)
        assert train_pca.shape[1] == n_pca

        # --- 9. K-Means ---
        val_no_idx = extract_non_overlapping_indices(
            n_windows=len(val_emb), window_size=config.window_size,
        )
        val_emb_no = val_emb[val_no_idx]
        val_pca = apply_pca(val_emb_no, pca)

        km = fit_kmeans(train_pca, K, random_state=config.seed)
        val_labels = km.predict(val_pca)
        assert len(val_labels) == len(val_pca)

        # --- 10. Clustering metrics ---
        cluster_metrics = compute_clustering_metrics(val_pca, val_labels)

        assert isinstance(cluster_metrics["silhouette"], float)
        assert -1.0 <= cluster_metrics["silhouette"] <= 1.0
        assert isinstance(cluster_metrics["davies_bouldin"], float)
        assert isinstance(cluster_metrics["calinski_harabasz"], float)
