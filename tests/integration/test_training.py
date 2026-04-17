"""Integration tests for the training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.training.callbacks import EarlyStopping, ModelCheckpoint
from tcc_itransformer.training.trainer import Trainer

# Small dimensions for fast tests
N_SERIES = 10
WINDOW_SIZE = 6
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
LATENT_DIM = 6
BATCH_SIZE = 8
MAX_EPOCHS = 5


def _make_config(tmp_path) -> ExperimentConfig:
    """Create a small ExperimentConfig for testing."""
    return ExperimentConfig(
        seed=42,
        window_size=WINDOW_SIZE,
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        n_clusters=3,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=MAX_EPOCHS,
        patience=20,
        grad_clip=1.0,
        results_dir=str(tmp_path / "results"),
    )


def _make_dataloaders(n_train: int = 32, n_val: int = 16):
    """Create synthetic train and val DataLoaders."""
    gen = torch.Generator().manual_seed(42)
    train_x = torch.randn(n_train, WINDOW_SIZE, N_SERIES, generator=gen)
    val_x = torch.randn(n_val, WINDOW_SIZE, N_SERIES, generator=gen)

    train_ds = TensorDataset(train_x, torch.arange(n_train))
    val_ds = TensorDataset(val_x, torch.arange(n_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


class TestTrainingLoop:
    def test_loss_decreases(self, tmp_path):
        """Train for 5 epochs on small mock data, verify epoch 5 loss < epoch 1."""
        config = _make_config(tmp_path)
        train_loader, val_loader = _make_dataloaders()
        model = iTransformerAE.from_config(config, N_SERIES)
        device = torch.device("cpu")

        trainer = Trainer(model, config, train_loader, val_loader, device)
        history = trainer.train()

        assert len(history["train_losses"]) == MAX_EPOCHS
        assert history["train_losses"][-1] < history["train_losses"][0], (
            "Training loss should decrease over epochs"
        )

    def test_no_nan_loss(self, tmp_path):
        """Train for 3 epochs, no NaN in train or val loss."""
        config = _make_config(tmp_path)
        config = config.model_copy(update={"max_epochs": 3})
        train_loader, val_loader = _make_dataloaders()
        model = iTransformerAE.from_config(config, N_SERIES)
        device = torch.device("cpu")

        trainer = Trainer(model, config, train_loader, val_loader, device)
        history = trainer.train()

        for tl in history["train_losses"]:
            assert not np.isnan(tl), "Train loss contains NaN"
        for vl in history["val_losses"]:
            assert not np.isnan(vl), "Val loss contains NaN"

    def test_early_stopping_triggers(self, tmp_path):
        """Mock constant val_loss, verify training stops at patience."""
        patience = 3
        config = _make_config(tmp_path)
        config = config.model_copy(update={"patience": patience, "max_epochs": 50})
        train_loader, val_loader = _make_dataloaders()
        model = iTransformerAE.from_config(config, N_SERIES)
        device = torch.device("cpu")

        # Use a very low learning rate so loss doesn't improve
        trainer = Trainer(model, config, train_loader, val_loader, device)
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-10, weight_decay=0.0)

        history = trainer.train()

        # Should stop well before max_epochs
        actual_epochs = len(history["train_losses"])
        assert actual_epochs < 50, (
            f"Expected early stopping before 50 epochs, ran {actual_epochs}"
        )

    def test_checkpoint_save_load(self, tmp_path):
        """Train, save, load — embeddings from loaded model match original."""
        config = _make_config(tmp_path)
        train_loader, val_loader = _make_dataloaders()
        model = iTransformerAE.from_config(config, N_SERIES)
        device = torch.device("cpu")

        trainer = Trainer(model, config, train_loader, val_loader, device)
        trainer.train()

        # Extract embeddings before reload
        emb_before = trainer.extract_embeddings(val_loader)

        # Create fresh model and load checkpoint
        model_new = iTransformerAE.from_config(config, N_SERIES)
        trainer.checkpoint.load_best(model_new)
        model_new.to(device)

        # Extract embeddings after reload
        trainer.model = model_new
        emb_after = trainer.extract_embeddings(val_loader)

        np.testing.assert_allclose(emb_before, emb_after, atol=1e-6)

    def test_extract_embeddings_shape(self, tmp_path):
        """Extract embeddings, verify shape (n_samples, latent_dim)."""
        config = _make_config(tmp_path)
        n_val = 16
        train_loader, val_loader = _make_dataloaders(n_val=n_val)
        model = iTransformerAE.from_config(config, N_SERIES)
        device = torch.device("cpu")

        trainer = Trainer(model, config, train_loader, val_loader, device)
        trainer.train()

        emb = trainer.extract_embeddings(val_loader)
        assert emb.shape == (n_val, LATENT_DIM), (
            f"Expected ({n_val}, {LATENT_DIM}), got {emb.shape}"
        )
