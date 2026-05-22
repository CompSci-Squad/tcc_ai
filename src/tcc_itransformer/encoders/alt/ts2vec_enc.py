"""TS2Vec trainable contrastive encoder adapter.

TS2Vec (Yue et al., AAAI 2022) learns universal representations via
hierarchical contrastive learning on augmented context views.  Unlike
reconstruction-based AEs, it uses timestamp-level contrastive objectives
which are invariant to temporal augmentations.

Install:
    uv add --optional phase_e ts2vec

Paper: https://arxiv.org/abs/2106.10466
Code:  https://github.com/zhihanyue/ts2vec
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

# Output dimensionality for the TS2Vec encoder.
# 64 balances expressiveness vs. clustering stability for our 7-dim downstream.
_TS2VEC_OUTPUT_DIMS = 64
_TS2VEC_HIDDEN_DIMS = 64
_TS2VEC_DEPTH = 10
_TS2VEC_LR = 0.001
_TS2VEC_BATCH_SIZE = 16
_TS2VEC_N_EPOCHS = 40  # ~10 minutes on CPU for W6, N=122


class TS2VecEncoder(AltEncoder):
    """Trainable TS2Vec encoder (contrastive, hierarchical).

    Trains on the training-split windows and encodes all splits by
    temporal mean-pooling the per-timestep representations.

    Parameters
    ----------
    output_dims : int
        Dimension of the per-timestep representation (default 64).
    n_epochs : int
        Number of training epochs (default 40).
    """

    name: ClassVar[str] = "ts2vec"
    tier: ClassVar[str] = "trainable"
    d_out: ClassVar[int] = _TS2VEC_OUTPUT_DIMS

    def __init__(
        self,
        output_dims: int = _TS2VEC_OUTPUT_DIMS,
        n_epochs: int = _TS2VEC_N_EPOCHS,
        lr: float = _TS2VEC_LR,
        batch_size: int = _TS2VEC_BATCH_SIZE,
    ) -> None:
        self._output_dims = output_dims
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._model = None

    def _check_import(self) -> None:
        try:
            import ts2vec  # noqa: F401, PLC0415
        except ImportError as exc:
            msg = "TS2Vec requires ts2vec: uv add --optional phase_e ts2vec"
            raise ImportError(msg) from exc

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Train TS2Vec on training-split windows.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]
        """
        self._check_import()
        from ts2vec import TS2Vec  # noqa: PLC0415

        n, T, N = windows.shape
        logger.info(
            "Training TS2Vec: n_windows=%d T=%d N=%d output_dims=%d epochs=%d",
            n, T, N, self._output_dims, self._n_epochs,
        )
        # TS2Vec expects [n_instances, T, input_dims] — same as our windows.
        self._model = TS2Vec(
            input_dims=N,
            output_dims=self._output_dims,
            hidden_dims=_TS2VEC_HIDDEN_DIMS,
            depth=_TS2VEC_DEPTH,
            device="cpu",
            lr=self._lr,
            batch_size=self._batch_size,
        )
        self._model.fit(
            windows.astype(np.float32),
            n_epochs=self._n_epochs,
            verbose=True,
        )
        logger.info("TS2Vec training complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode windows → temporal mean-pooled embeddings.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]

        Returns
        -------
        np.ndarray, shape [n_windows, output_dims]
        """
        if self._model is None:
            msg = "TS2VecEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        # TS2Vec.encode returns [n, T, d] per-timestep; average over T.
        z = self._model.encode(windows.astype(np.float32))  # [n, T, d]
        return z.mean(axis=1).astype(np.float32)  # [n, d]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ts2vec  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False
