"""TimesNet trainable encoder adapter.

TimesNet (Wu et al., ICLR 2023) converts 1-D time series into 2-D
representations by detecting dominant periods via FFT, reshaping each
variate's series into a (F × T/F) 2-D tensor, and applying 2-D
inception-style convolutions.  This allows the model to learn both
intra-period and inter-period temporal patterns.

We implement the core architecture directly (no external dep beyond PyTorch)
and train it as an autoencoder on the windowed FRED-MD panel.

Paper: https://arxiv.org/abs/2210.02186
Code:  https://github.com/thuml/Time-Series-Library
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

_TIMESNET_D_MODEL = 64
_TIMESNET_D_FF = 128
_TIMESNET_N_KERNELS = 6  # number of top-k FFT periods
_TIMESNET_N_EPOCHS = 40
_TIMESNET_LR = 1e-3
_TIMESNET_BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────────────────────
# TimesNet core blocks (no external dep)
# ──────────────────────────────────────────────────────────────────────────────

class _InceptionBlock(nn.Module):
    """Multi-scale 2-D conv block (Inception style) used in TimesNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        r = F.relu(self.bn(self.conv1(x) + self.conv3(x) + self.conv5(x) + self.proj(x)))
        return r


class _TimesBlock(nn.Module):
    """One TimesNet block: FFT period detection → 2-D conv → flatten."""

    def __init__(self, seq_len: int, d_model: int, d_ff: int, n_periods: int) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._n_periods = n_periods
        self.inception = _InceptionBlock(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _detect_periods(self, x: torch.Tensor) -> list[int]:
        """Return top-k dominant periods via FFT magnitude."""
        # x: [B, d, T]
        freq = torch.fft.rfft(x, dim=-1).abs()  # [B, d, T//2+1]
        freq_mean = freq.mean(dim=(0, 1))[1:]  # skip DC
        k = min(self._n_periods, freq_mean.shape[0])
        _, idx = torch.topk(freq_mean, k)
        periods = (self._seq_len / (idx + 1).float()).round().int().tolist()
        return [max(2, min(p, self._seq_len)) for p in periods]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]  → transpose → [B, d, T]
        B, T, d = x.shape
        xp = x.permute(0, 2, 1)  # [B, d, T]
        periods = self._detect_periods(xp)
        outs: list[torch.Tensor] = []
        for p in periods:
            # Pad to nearest multiple of period
            pad_len = (p - T % p) % p
            xpad = F.pad(xp, (0, pad_len))  # [B, d, T+pad]
            T2 = (T + pad_len) // p
            # Reshape to [B, d, T2, p] then Conv2d
            xr = xpad.reshape(B, d, T2, p)
            h = self.inception(xr)  # [B, d, T2, p]
            # Flatten back to [B, d, T+pad] → trim → [B, d, T]
            h = h.reshape(B, d, T2 * p)[:, :, :T]
            outs.append(h)
        agg = torch.stack(outs, dim=1).mean(dim=1)  # [B, d, T]
        agg = agg.permute(0, 2, 1)  # [B, T, d]
        return self.norm(x + self.ff(agg))


class _TimesNetAE(nn.Module):
    """Lightweight TimesNet autoencoder for self-supervised training."""

    def __init__(self, n_features: int, seq_len: int, d_model: int, d_ff: int, n_periods: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.block = _TimesBlock(seq_len, d_model, d_ff, n_periods)
        self.output_proj = nn.Linear(d_model, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] → [B, T, d] → mean over T → [B, d]
        h = self.block(self.input_proj(x))
        return h.mean(dim=1)  # [B, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder then decode (for reconstruction loss)
        h = self.block(self.input_proj(x))     # [B, T, d]
        return self.output_proj(h)              # [B, T, N]


# ──────────────────────────────────────────────────────────────────────────────
# Adapter class
# ──────────────────────────────────────────────────────────────────────────────

class TimesNetEncoder(AltEncoder):
    """TimesNet encoder trained with reconstruction objective.

    Architecture: Input projection → TimesBlock (FFT 2-D conv) → mean pool.
    No external dependency beyond PyTorch.
    """

    name: ClassVar[str] = "timesnet"
    tier: ClassVar[str] = "trainable"
    d_out: ClassVar[int] = _TIMESNET_D_MODEL

    def __init__(
        self,
        d_model: int = _TIMESNET_D_MODEL,
        d_ff: int = _TIMESNET_D_FF,
        n_periods: int = _TIMESNET_N_KERNELS,
        n_epochs: int = _TIMESNET_N_EPOCHS,
        lr: float = _TIMESNET_LR,
        batch_size: int = _TIMESNET_BATCH_SIZE,
    ) -> None:
        self._d_model = d_model
        self._d_ff = d_ff
        self._n_periods = n_periods
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._model: _TimesNetAE | None = None

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Train TimesNet autoencoder on training-split windows."""
        torch.manual_seed(seed)
        n, T, N = windows.shape
        logger.info(
            "Training TimesNet: n=%d T=%d N=%d d_model=%d epochs=%d",
            n, T, N, self._d_model, self._n_epochs,
        )
        model = _TimesNetAE(N, T, self._d_model, self._d_ff, self._n_periods)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)
        X = torch.from_numpy(windows.astype(np.float32))
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
                optimizer.zero_grad()
                recon = model(xb)
                loss = F.mse_loss(recon, xb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("  epoch %d/%d loss=%.4f", epoch + 1, self._n_epochs, total)
        self._model = model
        self._model.eval()
        logger.info("TimesNet training complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode windows → [n, d_model] embeddings."""
        if self._model is None:
            msg = "TimesNetEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        all_Z: list[np.ndarray] = []
        n = windows.shape[0]
        for start in range(0, n, self._batch_size):
            xb = torch.from_numpy(windows[start : start + self._batch_size].astype(np.float32))
            with torch.no_grad():
                Z = self._model.encode(xb).cpu().numpy()
            all_Z.append(Z)
        return np.concatenate(all_Z, axis=0).astype(np.float32)
