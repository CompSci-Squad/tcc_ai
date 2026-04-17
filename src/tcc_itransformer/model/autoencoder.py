"""Full iTransformer autoencoder: encoder + decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.model.decoder import iTransformerDecoder
from tcc_itransformer.model.encoder import iTransformerEncoder


class iTransformerAE(nn.Module):
    """iTransformer autoencoder that compresses multivariate time-series
    windows into a low-dimensional latent space and reconstructs them.

    The encoder uses the variate-as-token paradigm with Transformer blocks,
    while the decoder is a symmetric FFN-based network.

    Args:
        n_series: Number of variates (N).
        window_size: Number of time steps per window (W).
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer / FFN blocks.
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
        self.encoder = iTransformerEncoder(
            n_series=n_series,
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = iTransformerDecoder(
            n_series=n_series,
            window_size=window_size,
            d_model=d_model,
            n_layers=n_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode then decode, returning both reconstruction and latent.

        Args:
            x: (B, W, N) input windows.

        Returns:
            Tuple of (x_hat, z):
                x_hat: (B, W, N) reconstructed windows.
                z: (B, latent_dim) latent representations.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        """Encode input windows to latent vectors (inference / embedding extraction).

        Args:
            x: (B, W, N) input windows.

        Returns:
            z: (B, latent_dim) latent representations.
        """
        return self.encoder(x)

    @classmethod
    def from_config(cls, config: ExperimentConfig, n_series: int) -> iTransformerAE:
        """Construct an autoencoder from an ExperimentConfig.

        Args:
            config: Experiment configuration with architecture hyperparameters.
            n_series: Number of variates in the dataset.

        Returns:
            Configured iTransformerAE instance.
        """
        return cls(
            n_series=n_series,
            window_size=config.window_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
