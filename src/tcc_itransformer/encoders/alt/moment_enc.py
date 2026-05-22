"""MOMENT-1 zero-shot encoder adapter.

MOMENT (Goswami et al., ICML 2024) is a family of open-source time-series
foundation models pre-trained on the Time-series Pile.  We use the
``"embedding"`` task to extract per-window representations without any
fine-tuning.

Install:
    uv add --optional phase_e momentfm

Paper: https://arxiv.org/abs/2402.03885
HF:   https://huggingface.co/AutonLab/MOMENT-1-small
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

# MOMENT minimum sequence length is patch_size = 8.  Our window W=6, so we
# right-pad with edge values to reach 8 before passing to the model.
_MOMENT_MIN_LEN = 8
# Use the smallest MOMENT variant to keep memory + latency low on CPU.
_MOMENT_CHECKPOINT = "AutonLab/MOMENT-1-small"


class MOMENTEncoder(AltEncoder):
    """Zero-shot encoder using MOMENT-1-small (37M params).

    Strategy
    --------
    For each window ``x`` of shape ``[T, N]``:
    1. Pad temporal dim to ≥ 8 (MOMENT patch_size requirement).
    2. Reshape to ``[1, N, T_padded]`` (batch=1, n_channels=N, seq_len).
    3. Forward through MOMENT encoder → ``embeddings: [1, N, d_model]``.
    4. Mean-pool over the channel dimension → ``[1, d_model]``.

    The channel mean-pool treats each of the 122 macro series as an
    independent token; their average captures global panel-level state.
    d_model = 256 for MOMENT-1-small.
    """

    name: ClassVar[str] = "moment"
    tier: ClassVar[str] = "zero-shot"
    d_out: ClassVar[int] = 512  # MOMENT-1-small d_model (T5-small backbone)

    def __init__(self, checkpoint: str = _MOMENT_CHECKPOINT, batch_size: int = 64) -> None:
        self._checkpoint = checkpoint
        self._batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from momentfm import MOMENTPipeline  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "MOMENT requires momentfm: "
                "uv add --optional phase_e momentfm"
            )
            raise ImportError(msg) from exc

        import torch  # noqa: PLC0415

        logger.info("Loading %s (zero-shot, CPU)…", self._checkpoint)
        model = MOMENTPipeline.from_pretrained(
            self._checkpoint,
            model_kwargs={"task_name": "embedding"},
        )
        model.init()
        model.eval()
        # Force CPU; MOMENT-1-small fits in ~500 MB RAM.
        model = model.to(torch.device("cpu"))
        self._model = model
        self._torch = torch

    # fit is a no-op: MOMENT is zero-shot
    def fit(self, windows: np.ndarray, seed: int = 42) -> None:  # noqa: ARG002
        self._load()

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode ``windows`` of shape ``[n, T, N]`` → ``[n, d_model]``."""
        self._load()
        import torch  # noqa: PLC0415

        n, T, N = windows.shape
        # Pad temporal dim to MOMENT's minimum length if needed.
        if T < _MOMENT_MIN_LEN:
            pad = np.repeat(windows[:, -1:, :], _MOMENT_MIN_LEN - T, axis=1)
            windows = np.concatenate([windows, pad], axis=1)
            T = _MOMENT_MIN_LEN

        all_Z: list[np.ndarray] = []
        for start in range(0, n, self._batch_size):
            batch = windows[start : start + self._batch_size]  # [B, T, N]
            # MOMENT expects [B, N, T] (channels second)
            x = torch.from_numpy(batch.transpose(0, 2, 1)).float()  # [B, N, T]
            with torch.no_grad():
                out = self._model(x_enc=x)
            # momentfm returns embeddings already pooled: [B, d_model]
            # (channels processed internally as B*N sequences, then aggregated)
            Z = out.embeddings.cpu().numpy()  # [B, d_model]
            all_Z.append(Z)

        return np.concatenate(all_Z, axis=0).astype(np.float32)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import momentfm  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False
