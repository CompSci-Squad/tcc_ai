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
