"""MOIRAI-1.0-R-Small zero-shot encoder adapter.

MOIRAI (Woo et al., ICLR 2024 / Salesforce) is an encoder-based universal
time-series foundation model with any-variate attention.  It natively handles
multivariate input without channel-independent assumptions.

We extract embeddings by forwarding windows through the patch embedding +
transformer encoder layers and mean-pooling the output patch tokens.

Install (not on PyPI — install from GitHub):
    pip install "git+https://github.com/SalesforceAIResearch/uni2ts.git"

Paper: https://arxiv.org/abs/2402.02592
HF:   https://huggingface.co/Salesforce/moirai-1.0-R-small
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

_MOIRAI_CHECKPOINT = "Salesforce/moirai-1.0-R-small"


class MoiraiEncoder(AltEncoder):
    """Zero-shot encoder using MOIRAI-1.0-R-Small (14M params).

    Strategy
    --------
    MOIRAI's transformer operates on patch tokens from each variate.
    We forward windows through the patch-embedding + transformer encoder
    and mean-pool the resulting patch token hidden states across all
    variates and patches to obtain a single [d_model]-dimensional vector
    per window.

    d_model ≈ 384 for MOIRAI-Small.
    """

    name: ClassVar[str] = "moirai"
    tier: ClassVar[str] = "zero-shot"
    d_out: ClassVar[int] = 384  # MOIRAI-Small d_model

    def __init__(self, checkpoint: str = _MOIRAI_CHECKPOINT, batch_size: int = 32) -> None:
        self._checkpoint = checkpoint
        self._batch_size = batch_size
        self._module = None

    def _load(self) -> None:
        if self._module is not None:
            return
        try:
            from uni2ts.model.moirai import MoiraiModule  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "MOIRAI requires uni2ts (not on PyPI):\n"
                "  pip install 'git+https://github.com/SalesforceAIResearch/uni2ts.git'\n"
                "Skip with: --encoders moment,ts2vec,patchtst,timesnet"
            )
            raise ImportError(msg) from exc

        import torch  # noqa: PLC0415

        logger.info("Loading %s (zero-shot, CPU)…", self._checkpoint)
        module = MoiraiModule.from_pretrained(self._checkpoint)
        module.eval()
        module = module.to(torch.device("cpu"))
        self._module = module
        self._torch = torch

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:  # noqa: ARG002
        self._load()

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode ``windows`` of shape ``[n, T, N]`` → ``[n, d_model]``."""
        self._load()
        import torch  # noqa: PLC0415

        n, T, N = windows.shape
        all_Z: list[np.ndarray] = []

        for start in range(0, n, self._batch_size):
            batch = windows[start : start + self._batch_size]  # [B, T, N]
            B = batch.shape[0]

            # MOIRAI patch-embedding expects [B, N, T] (variates, timesteps).
            x = torch.from_numpy(batch.transpose(0, 2, 1)).float()  # [B, N, T]

            with torch.no_grad():
                # Access the encoder's patch embedding + transformer directly.
                # MOIRAI stores its encoder transformer in module.model.
                # We use the raw patch_embed + blocks to get hidden states.
                try:
                    # Flatten variates into batch dim for patch embedding
                    x_flat = x.reshape(B * N, 1, T)  # [B*N, 1, T]
                    patch_emb = self._module.patch_embed(x_flat)  # [B*N, n_patches, d_model]
                    # Run through transformer encoder
                    h = self._module.encoder(patch_emb)  # [B*N, n_patches, d_model]
                    # Mean pool over patches and variates
                    h = h.mean(dim=1)  # [B*N, d_model]
                    h = h.reshape(B, N, -1).mean(dim=1)  # [B, d_model]
                except AttributeError:
                    # Fallback: try accessing model.encoder differently
                    logger.warning(
                        "MOIRAI internal API changed; attempting fallback via forward()."
                    )
                    # Pass through as a "forecasting" with prediction_length=1
                    # and extract the encoder last hidden state.
                    h = self._module(x, observed_mask=torch.ones_like(x).bool())
                    if hasattr(h, "last_hidden_state"):
                        h = h.last_hidden_state.mean(dim=(1, 2))
                    else:
                        h = h.mean(dim=(1, 2)) if h.ndim == 4 else h.mean(dim=1)

            all_Z.append(h.cpu().numpy())

        return np.concatenate(all_Z, axis=0).astype(np.float32)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from uni2ts.model.moirai import MoiraiModule  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False
