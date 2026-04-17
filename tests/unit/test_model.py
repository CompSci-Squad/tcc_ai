"""Comprehensive tests for the iTransformer autoencoder model architecture."""

from __future__ import annotations

import torch
import pytest

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.model.decoder import iTransformerDecoder
from tcc_itransformer.model.encoder import iTransformerEncoder
from tcc_itransformer.model.layers import (
    DecoderFFNBlock,
    TransformerEncoderBlock,
    VariateEmbedding,
)

# Small dims for fast tests
B = 4
N = 20
W = 12
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 1
LATENT_DIM = 4
DROPOUT = 0.0  # deterministic tests


@pytest.fixture()
def sample_input() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(B, W, N, generator=gen)


@pytest.fixture()
def model() -> iTransformerAE:
    return iTransformerAE(
        n_series=N,
        window_size=W,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    )


# --- VariateEmbedding ---


def test_variate_embedding_shape() -> None:
    emb = VariateEmbedding(window_size=W, d_model=D_MODEL, dropout=DROPOUT)
    x = torch.randn(B, N, W)
    out = emb(x)
    assert out.shape == (B, N, D_MODEL)


def test_variate_embedding_large() -> None:
    emb = VariateEmbedding(window_size=12, d_model=64, dropout=0.0)
    x = torch.randn(4, 128, 12)
    out = emb(x)
    assert out.shape == (4, 128, 64)


# --- TransformerEncoderBlock ---


def test_transformer_block_shape() -> None:
    block = TransformerEncoderBlock(d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT)
    x = torch.randn(B, N, D_MODEL)
    out = block(x)
    assert out.shape == (B, N, D_MODEL)


def test_transformer_block_large() -> None:
    block = TransformerEncoderBlock(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(4, 128, 64)
    out = block(x)
    assert out.shape == (4, 128, 64)


# --- Encoder ---


def test_encoder_output_shape() -> None:
    encoder = iTransformerEncoder(
        n_series=N, window_size=W, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, latent_dim=LATENT_DIM, dropout=DROPOUT,
    )
    x = torch.randn(B, W, N)
    z = encoder(x)
    assert z.shape == (B, LATENT_DIM)


# --- Decoder ---


def test_decoder_output_shape() -> None:
    decoder = iTransformerDecoder(
        n_series=N, window_size=W, d_model=D_MODEL,
        n_layers=N_LAYERS, latent_dim=LATENT_DIM, dropout=DROPOUT,
    )
    z = torch.randn(B, LATENT_DIM)
    x_hat = decoder(z)
    assert x_hat.shape == (B, W, N)


# --- Autoencoder ---


def test_autoencoder_roundtrip(sample_input: torch.Tensor, model: iTransformerAE) -> None:
    x_hat, z = model(sample_input)
    assert x_hat.shape == sample_input.shape
    assert z.shape == (B, LATENT_DIM)


def test_encode_method(sample_input: torch.Tensor, model: iTransformerAE) -> None:
    z = model.encode(sample_input)
    assert z.shape == (B, LATENT_DIM)


def test_no_nan_forward(sample_input: torch.Tensor, model: iTransformerAE) -> None:
    x_hat, z = model(sample_input)
    assert not torch.isnan(x_hat).any(), "NaN detected in x_hat"
    assert not torch.isnan(z).any(), "NaN detected in z"


def test_gradient_flow(sample_input: torch.Tensor, model: iTransformerAE) -> None:
    x_hat, z = model(sample_input)
    loss = x_hat.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_param_count_small() -> None:
    small_model = iTransformerAE(
        n_series=20, window_size=12, d_model=32,
        n_heads=4, n_layers=1, latent_dim=4, dropout=0.0,
    )
    total = sum(p.numel() for p in small_model.parameters())
    assert total < 500_000, f"Parameter count {total} exceeds 500k"


def test_from_config() -> None:
    config = ExperimentConfig(d_model=32, n_heads=4, n_layers=1, latent_dim=4)
    model = iTransformerAE.from_config(config, n_series=20)
    x = torch.randn(2, config.window_size, 20)
    x_hat, z = model(x)
    assert x_hat.shape == x.shape
    assert z.shape == (2, config.latent_dim)
