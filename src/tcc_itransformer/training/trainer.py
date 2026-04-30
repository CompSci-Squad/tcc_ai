"""Training orchestrator for the iTransformer autoencoder."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.model.losses import (
    masked_reconstruction_loss,
    reconstruction_loss,
)
from tcc_itransformer.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for iTransformerAE.

    Handles the full training loop with AdamW optimizer, cosine LR schedule,
    early stopping, and model checkpointing.

    Args:
        model: The iTransformer autoencoder model.
        config: Experiment configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model: iTransformerAE,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
        )
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.checkpoint = ModelCheckpoint(
            save_dir=Path(config.results_dir) / "checkpoints",
        )

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x = batch[0].to(self.device)
            mask = batch[1].to(self.device) if len(batch) == 3 else None
            self.optimizer.zero_grad()
            x_hat, _z = self.model(x)
            loss = (
                masked_reconstruction_loss(x, x_hat, mask)
                if mask is not None
                else reconstruction_loss(x, x_hat)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip,
            )
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        """Run one validation epoch. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            x = batch[0].to(self.device)
            mask = batch[1].to(self.device) if len(batch) == 3 else None
            x_hat, _z = self.model(x)
            loss = (
                masked_reconstruction_loss(x, x_hat, mask)
                if mask is not None
                else reconstruction_loss(x, x_hat)
            )
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self) -> dict:
        """Full training loop with early stopping and checkpointing.

        Returns:
            History dictionary with keys:
                train_losses, val_losses, best_epoch, stopped_epoch.
        """
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = 0

        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            self.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.checkpoint(self.model, val_loss):
                best_epoch = epoch

            logger.info(
                "Epoch %03d  train=%.6f  val=%.6f  lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            if self.early_stopping(val_loss):
                break

        stopped_epoch = len(train_losses) - 1

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
        }

    @torch.no_grad()
    def extract_embeddings(self, dataloader: DataLoader) -> np.ndarray:
        """Extract latent embeddings from all windows in a dataloader.

        Args:
            dataloader: DataLoader yielding (x, idx) batches.

        Returns:
            Array of shape (n_samples, latent_dim). When the dataloader is
            empty (e.g. D7.a dropped every window of a split), returns a
            zero-row array of shape (0, latent_dim) and logs a warning rather
            than raising — downstream code should handle empty splits.
        """
        self.model.eval()
        all_z: list[np.ndarray] = []

        for batch in dataloader:
            x = batch[0].to(self.device)
            z = self.model.encode(x)
            all_z.append(z.cpu().numpy())

        if not all_z:
            latent_dim = int(self.config.latent_dim)
            logger.warning(
                "extract_embeddings: dataloader is empty; returning (0, %d) array",
                latent_dim,
            )
            return np.zeros((0, latent_dim), dtype=np.float32)

        return np.concatenate(all_z, axis=0)

    @torch.no_grad()
    def compute_train_mean(self, dataloader: DataLoader) -> torch.Tensor:
        """Compute mean of all training windows for naive baseline.

        Args:
            dataloader: Training DataLoader.

        Returns:
            Tensor of shape (W, N) — mean window across training set.
        """
        total = None
        count = 0

        for batch in dataloader:
            x = batch[0]
            if total is None:
                total = torch.zeros_like(x[0])
            total += x.sum(dim=0)
            count += x.shape[0]

        if total is None:
            msg = "Empty dataloader"
            raise ValueError(msg)

        return total / count
