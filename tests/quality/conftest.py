"""Fixtures for scientific quality gate tests.

These tests verify that a trained model meets minimum quality thresholds.
Fixtures are session-scoped: train once, reuse across all quality tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.training.trainer import Trainer

N_SERIES = 10
WINDOW_SIZE = 6
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
LATENT_DIM = 6
BATCH_SIZE = 8
MAX_EPOCHS = 50
K = 3


def _make_loaders(
    n_train: int = 60,
    n_val: int = 20,
    n_test: int = 20,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    gen = torch.Generator().manual_seed(42)
    train_x = torch.randn(n_train, WINDOW_SIZE, N_SERIES, generator=gen)
    val_x = torch.randn(n_val, WINDOW_SIZE, N_SERIES, generator=gen)
    test_x = torch.randn(n_test, WINDOW_SIZE, N_SERIES, generator=gen)
    train_ds = TensorDataset(train_x, torch.arange(n_train))
    val_ds = TensorDataset(val_x, torch.arange(n_val))
    test_ds = TensorDataset(test_x, torch.arange(n_test))
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


@pytest.fixture(scope="session")
def quality_config(tmp_path_factory: pytest.TempPathFactory) -> ExperimentConfig:
    tmp = tmp_path_factory.mktemp("quality")
    return ExperimentConfig(
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
        patience=50,
        grad_clip=1.0,
        results_dir=str(tmp / "results"),
    )


@pytest.fixture(scope="session")
def trained_model_and_data(
    quality_config: ExperimentConfig,
) -> dict:
    """Train a model and return model + embeddings + loaders."""
    torch.manual_seed(quality_config.seed)
    np.random.seed(quality_config.seed)

    train_loader, val_loader, test_loader = _make_loaders()
    device = torch.device("cpu")

    model = iTransformerAE.from_config(quality_config, N_SERIES)
    trainer = Trainer(model, quality_config, train_loader, val_loader, device)
    history = trainer.train()
    trainer.checkpoint.load_best(model)

    train_emb = trainer.extract_embeddings(train_loader)
    val_emb = trainer.extract_embeddings(val_loader)
    test_emb = trainer.extract_embeddings(test_loader)
    train_mean = trainer.compute_train_mean(train_loader)

    return {
        "model": model,
        "trainer": trainer,
        "config": quality_config,
        "history": history,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_emb": train_emb,
        "val_emb": val_emb,
        "test_emb": test_emb,
        "train_mean": train_mean,
        "device": device,
    }
