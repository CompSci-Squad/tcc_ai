"""Training callbacks: early stopping and model checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation loss.

    Monitors validation loss and signals when training should stop
    if no improvement is observed for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter: int = 0
        self._should_stop: bool = False

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def __call__(self, val_loss: float) -> bool:
        """Check whether training should stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self._patience,
                )

        return self._should_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self._best_loss = float("inf")
        self._counter = 0
        self._should_stop = False


class ModelCheckpoint:
    """Save best model weights based on validation loss.

    Persists the model state dict whenever a new lowest validation loss
    is observed.

    Args:
        save_dir: Directory where checkpoints are stored.
        filename: Name of the checkpoint file.
    """

    def __init__(self, save_dir: Path, filename: str = "best_model.pt") -> None:
        self._save_dir = Path(save_dir)
        self._filename = filename
        self._best_loss: float = float("inf")
        self._save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._save_dir / self._filename

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """Save model if validation loss improved.

        Args:
            model: The model whose state dict to save.
            val_loss: Current epoch's validation loss.

        Returns:
            True if model was saved (new best), False otherwise.
        """
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            logger.debug("Checkpoint saved at epoch loss=%.6f", val_loss)
            return True
        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best checkpoint weights into the given model.

        Args:
            model: Model to load weights into.

        Returns:
            Model with loaded weights.
        """
        state_dict = torch.load(self.path, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded best checkpoint from %s", self.path)
        return model
