"""Tests for autoencoder loss functions."""

from __future__ import annotations

import torch
import pytest

from tcc_itransformer.model.losses import (
    masked_reconstruction_loss,
    naive_baseline_loss,
    reconstruction_loss,
)


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


def test_masked_reconstruction_equals_unmasked_when_no_imputation() -> None:
    x = torch.randn(4, 12, 20)
    x_hat = torch.randn(4, 12, 20)
    mask = torch.zeros_like(x, dtype=torch.bool)
    masked = masked_reconstruction_loss(x, x_hat, mask)
    unmasked = reconstruction_loss(x, x_hat)
    assert masked.item() == pytest.approx(unmasked.item(), rel=1e-6)


def test_masked_reconstruction_ignores_imputed_cells() -> None:
    # Perfect reconstruction on observed cells; bogus values on imputed cells.
    x = torch.zeros(2, 3, 4)
    x_hat = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 0, 0] = True  # mark as imputed
    x_hat[0, 0, 0] = 100.0  # huge error on imputed cell
    loss = masked_reconstruction_loss(x, x_hat, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_masked_reconstruction_all_imputed_returns_zero() -> None:
    x = torch.randn(2, 3, 4)
    x_hat = torch.randn(2, 3, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    loss = masked_reconstruction_loss(x, x_hat, mask)
    assert loss.item() == 0.0


def test_masked_reconstruction_partial_mask_matches_manual() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    x_hat = torch.randn(2, 3, 4)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 0, :] = True  # row of imputed cells
    expected_sq = ((x_hat - x) ** 2)[~mask]
    expected = expected_sq.mean()
    loss = masked_reconstruction_loss(x, x_hat, mask)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
