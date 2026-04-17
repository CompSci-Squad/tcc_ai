"""Tests for autoencoder loss functions."""

from __future__ import annotations

import torch
import pytest

from tcc_itransformer.model.losses import naive_baseline_loss, reconstruction_loss


def test_reconstruction_loss_zero() -> None:
    x = torch.randn(4, 12, 20)
    loss = reconstruction_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_reconstruction_loss_positive() -> None:
    x = torch.randn(4, 12, 20)
    x_hat = torch.randn(4, 12, 20)
    loss = reconstruction_loss(x, x_hat)
    assert loss.item() > 0.0


def test_reconstruction_loss_is_scalar() -> None:
    x = torch.randn(4, 12, 20)
    x_hat = torch.randn(4, 12, 20)
    loss = reconstruction_loss(x, x_hat)
    assert loss.dim() == 0


def test_naive_baseline_loss() -> None:
    x = torch.ones(4, 12, 20) * 3.0
    train_mean = torch.ones(1, 12, 20) * 1.0
    loss = naive_baseline_loss(x, train_mean)
    # Expected: MSE between 1.0 and 3.0 = (3.0 - 1.0)^2 = 4.0
    assert loss.item() == pytest.approx(4.0, abs=1e-6)


def test_naive_baseline_loss_2d_mean() -> None:
    x = torch.ones(4, 12, 20) * 2.0
    train_mean = torch.zeros(12, 20)  # (W, N) — broadcasts to (B, W, N)
    loss = naive_baseline_loss(x, train_mean)
    assert loss.item() == pytest.approx(4.0, abs=1e-6)


def test_naive_baseline_vs_model() -> None:
    x = torch.randn(4, 12, 20)
    x_hat = torch.randn(4, 12, 20)
    train_mean = x.mean(dim=0, keepdim=True)

    model_loss = reconstruction_loss(x, x_hat)
    baseline_loss = naive_baseline_loss(x, train_mean)

    assert model_loss.dim() == 0
    assert baseline_loss.dim() == 0
