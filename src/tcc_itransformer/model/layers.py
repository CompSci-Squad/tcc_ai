"""Reusable building blocks for the iTransformer autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VariateEmbedding(nn.Module):
    """Inverted embedding: projects each variate's W time steps to d_model.

    Follows the iTransformer variate-as-token paradigm where each series is
    treated as a single token whose raw features are the W time-step values.

    Input:  (B, N, W) — N variates, W time steps per variate
    Output: (B, N, d_model)
    """

    def __init__(self, window_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(window_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Project each variate from W time steps to d_model dimensions."""
        x = self.projection(x)  # (B, N, W) -> (B, N, d_model)
        x = self.norm(x)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with self-attention across variates.

    Self-attention operates across the N variate tokens so the model learns
    inter-series dependencies — the core insight of iTransformer.

    Components (pre-norm style):
        LayerNorm → MultiheadAttention → Residual
        LayerNorm → FFN (Linear → GELU → Dropout → Linear → Dropout) → Residual

    Input / Output: (B, N, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(p=dropout)

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(d_model)
        ffn_dim = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-norm attention and FFN with residual connections."""
        # Self-attention block
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout1(attn_out)

        # Feed-forward block
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderFFNBlock(nn.Module):
    """Pre-norm FFN block for the decoder (no attention).

    Components:
        LayerNorm → FFN (Linear → GELU → Dropout → Linear → Dropout) → Residual

    Input / Output: (B, N, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        ffn_dim = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-norm FFN with residual connection."""
        return x + self.ffn(self.norm(x))
