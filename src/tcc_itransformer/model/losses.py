"""Loss functions for the iTransformer autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """MSE reconstruction loss averaged over all elements.

    Args:
        x: (B, W, N) original windows.
        x_hat: (B, W, N) reconstructed windows.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(x_hat, x, reduction="mean")


def masked_reconstruction_loss(
    x: Tensor, x_hat: Tensor, imputed_mask: Tensor
) -> Tensor:
    """MSE reconstruction loss computed only over observed (non-imputed) cells.

    D7.c policy: cells flagged as imputed by the ETL EM-PCA step are excluded
    from the loss, so the autoencoder is never graded on synthetic targets.

    Args:
        x: (B, W, N) original windows.
        x_hat: (B, W, N) reconstructed windows.
        imputed_mask: (B, W, N) Boolean tensor; ``True`` marks cells imputed
            by the ETL pipeline.

    Returns:
        Scalar MSE over observed cells. If a batch happens to be 100% imputed
        (no observed cells), returns 0.0 to avoid NaN gradients.
    """
    observed = (~imputed_mask).float()
    n_observed = observed.sum()
    if n_observed.item() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    sq = (x_hat - x) ** 2
    return (sq * observed).sum() / n_observed


def naive_baseline_loss(x: Tensor, train_mean: Tensor) -> Tensor:
    """MSE of predicting the training-set mean for every window.

    Provides a sanity-check baseline: the autoencoder should beat this.

    Args:
        x: (B, W, N) target windows.
        train_mean: (1, W, N) or (W, N) — mean computed from training set,
            broadcast to match x shape.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(train_mean.expand_as(x), x, reduction="mean")
