"""TF-C (Time-Frequency Consistency) contrastive encoder adapter.

TF-C (Zhang et al., NeurIPS 2022) learns representations by enforcing
consistency between the time-domain and frequency-domain views of the same
series.  The key insight for macro time series is that business cycles have
dominant spectral signatures (NBER recessions at ~7-year periods), which
TF-C's frequency branch is designed to capture.

Architecture:
- Time encoder: 1-D causal CNN over the raw sequence.
- Frequency encoder: 1-D CNN over the FFT magnitude spectrum.
- Projection heads map both branches to the same latent space.
- Cross-view contrastive loss (NT-Xent) pulls matched (time, freq) pairs
  of the same sample together and pushes unmatched pairs apart.

No external dependency beyond PyTorch.

Paper: https://arxiv.org/abs/2206.08496
Code:  https://github.com/mims-harvard/TFC-pretraining
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

_TFC_D_LATENT = 64
_TFC_D_PROJECT = 64
_TFC_N_EPOCHS = 40
_TFC_LR = 3e-4
_TFC_BATCH_SIZE = 32
_TFC_TEMPERATURE = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# TF-C network components
# ──────────────────────────────────────────────────────────────────────────────

class _TEncoder(nn.Module):
    """Time-domain 1-D CNN encoder.  Input: [B, N, T] → output: [B, d]."""

    def __init__(self, n_features: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # temporal mean pool → [B, 64, 1]
        )
        self.proj = nn.Linear(64, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] → permute → [B, N, T]
        h = self.net(x.permute(0, 2, 1)).squeeze(-1)  # [B, 64]
        return self.proj(h)  # [B, d_out]


class _FEncoder(nn.Module):
    """Frequency-domain 1-D CNN encoder.  Input: FFT magnitude [B, N, F] → [B, d]."""

    def __init__(self, n_features: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] → permute → [B, N, T]
        h = self.net(x.permute(0, 2, 1)).squeeze(-1)
        return self.proj(h)


class _TFCModel(nn.Module):
    """Full TF-C model with projection heads."""

    def __init__(self, n_features: int, d_latent: int, d_proj: int) -> None:
        super().__init__()
        self.t_enc = _TEncoder(n_features, d_latent)
        self.f_enc = _FEncoder(n_features, d_latent)
        # Projection heads (used only for contrastive loss, not for encoding)
        self.t_proj = nn.Sequential(nn.Linear(d_latent, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))
        self.f_proj = nn.Sequential(nn.Linear(d_latent, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj))

    def forward(self, x_t: torch.Tensor, x_f: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z_t = self.t_enc(x_t)
        z_f = self.f_enc(x_f)
        p_t = F.normalize(self.t_proj(z_t), dim=-1)
        p_f = F.normalize(self.f_proj(z_f), dim=-1)
        return z_t, z_f, p_t, p_f

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Average time + frequency representations for the final embedding."""
        z_t = self.t_enc(x)
        # Build frequency representation from FFT magnitude
        x_f = torch.fft.rfft(x, dim=1).abs()  # [B, T//2+1, N]
        z_f = self.f_enc(x_f)
        return (z_t + z_f) / 2.0  # [B, d_latent]


def _nt_xent_loss(p_t: torch.Tensor, p_f: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent cross-view contrastive loss between time and frequency projections."""
    B = p_t.shape[0]
    # Concatenate both views: [2B, d]
    z = torch.cat([p_t, p_f], dim=0)
    sim = torch.mm(z, z.T) / temperature  # [2B, 2B]
    # Mask self-similarity
    mask = torch.eye(2 * B, device=p_t.device).bool()
    sim.masked_fill_(mask, float("-inf"))
    # Positive pairs are (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(p_t.device)
    return F.cross_entropy(sim, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Adapter class
# ──────────────────────────────────────────────────────────────────────────────

class TFCEncoder(AltEncoder):
    """TF-C contrastive encoder (time + frequency consistency).

    Trains with NT-Xent cross-view loss between time-domain and
    frequency-domain representations of the same window.
    """

    name: ClassVar[str] = "tfc"
    tier: ClassVar[str] = "trainable"
    d_out: ClassVar[int] = _TFC_D_LATENT

    def __init__(
        self,
        d_latent: int = _TFC_D_LATENT,
        d_proj: int = _TFC_D_PROJECT,
        n_epochs: int = _TFC_N_EPOCHS,
        lr: float = _TFC_LR,
        batch_size: int = _TFC_BATCH_SIZE,
        temperature: float = _TFC_TEMPERATURE,
    ) -> None:
        self._d_latent = d_latent
        self._d_proj = d_proj
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._temperature = temperature
        self._model: _TFCModel | None = None

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Train TF-C with NT-Xent cross-view contrastive loss."""
        torch.manual_seed(seed)
        n, T, N = windows.shape
        logger.info(
            "Training TF-C: n=%d T=%d N=%d d_latent=%d epochs=%d",
            n, T, N, self._d_latent, self._n_epochs,
        )
        model = _TFCModel(N, self._d_latent, self._d_proj)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)
        X = torch.from_numpy(windows.astype(np.float32))  # [n, T, N]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
        )
        model.train()
        for epoch in range(self._n_epochs):
            total = 0.0
            for (xb,) in loader:
                # Frequency view: FFT magnitude
                xb_f = torch.fft.rfft(xb, dim=1).abs()  # [B, T//2+1, N]
                optimizer.zero_grad()
                _, _, p_t, p_f = model(xb, xb_f)
                loss = _nt_xent_loss(p_t, p_f, self._temperature)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("  epoch %d/%d loss=%.4f", epoch + 1, self._n_epochs, total)
        self._model = model
        self._model.eval()
        logger.info("TF-C training complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode windows → [n, d_latent] embeddings."""
        if self._model is None:
            msg = "TFCEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        all_Z: list[np.ndarray] = []
        n = windows.shape[0]
        for start in range(0, n, self._batch_size):
            xb = torch.from_numpy(windows[start : start + self._batch_size].astype(np.float32))
            with torch.no_grad():
                Z = self._model.encode(xb).cpu().numpy()
            all_Z.append(Z)
        return np.concatenate(all_Z, axis=0).astype(np.float32)
