"""iTransformer encoder with variate-as-token inversion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcc_itransformer.model.layers import TransformerEncoderBlock, VariateEmbedding


class iTransformerEncoder(nn.Module):
    """iTransformer encoder: inverts the time-series representation so each
    variate becomes a token, then applies Transformer blocks and mean-pools
    to a fixed-size latent vector.

    Forward pass:
        1. Transpose: (B, W, N) → (B, N, W)
        2. VariateEmbedding: (B, N, W) → (B, N, d_model)
        3. L × TransformerEncoderBlock: (B, N, d_model) → (B, N, d_model)
        4. Mean pool across N variates: (B, N, d_model) → (B, d_model)
        5. Linear projection: (B, d_model) → z: (B, latent_dim)

    Args:
        n_series: Number of variates (N).
        window_size: Number of time steps per window (W).
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        latent_dim: Bottleneck dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_series: int,
        window_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = VariateEmbedding(window_size, d_model, dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.projection = nn.Linear(d_model, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode input windows to latent vectors.

        Args:
            x: (B, W, N) input tensor.

        Returns:
            z: (B, latent_dim) latent representation.
        """
        # Step 1: Invert — variate-as-token
        x = x.permute(0, 2, 1)  # (B, W, N) -> (B, N, W)

        # Step 2: Embed each variate
        x = self.embedding(x)  # (B, N, d_model)

        # Step 3: Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 4: Mean pool across variates
        x = x.mean(dim=1)  # (B, d_model)

        # Step 5: Project to latent space
        return self.projection(x)  # (B, latent_dim)
