"""Mirror decoder for iTransformer autoencoder reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcc_itransformer.model.layers import DecoderFFNBlock


class iTransformerDecoder(nn.Module):
    """Mirror decoder that reconstructs the original (B, W, N) windows from
    a latent vector z.

    Forward pass:
        1. Linear: (B, latent_dim) → (B, d_model)
        2. Expand: (B, d_model) → (B, N, d_model) via unsqueeze + expand
        3. L × DecoderFFNBlock: (B, N, d_model) → (B, N, d_model)
        4. Linear projection: (B, N, d_model) → (B, N, W)
        5. Transpose: (B, N, W) → (B, W, N)

    Note: The decoder does NOT use attention — only per-variate FFN blocks.

    Args:
        n_series: Number of variates (N).
        window_size: Number of time steps per window (W).
        d_model: Embedding dimension.
        n_layers: Number of FFN blocks.
        latent_dim: Bottleneck dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_series: int,
        window_size: int,
        d_model: int,
        n_layers: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_series = n_series
        self.expand = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            [DecoderFFNBlock(d_model, dropout) for _ in range(n_layers)]
        )
        self.projection = nn.Linear(d_model, window_size)

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vectors back to reconstructed windows.

        Args:
            z: (B, latent_dim) latent representation.

        Returns:
            x_hat: (B, W, N) reconstructed windows.
        """
        # Step 1: Project from latent to d_model
        x = self.expand(z)  # (B, d_model)

        # Step 2: Expand to N variates
        x = x.unsqueeze(1).expand(-1, self.n_series, -1)  # (B, N, d_model)

        # Step 3: FFN blocks
        for block in self.blocks:
            x = block(x)

        # Step 4: Project each variate to W time steps
        x = self.projection(x)  # (B, N, W)

        # Step 5: Transpose back to (B, W, N)
        return x.permute(0, 2, 1)
