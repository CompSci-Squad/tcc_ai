"""Abstract base class for Phase E alternative encoders.

All encoders in ``tcc_itransformer/encoders/alt/`` implement this interface.
The Phase E runner (``scripts/run_phase_e_encoders.py``) calls ``fit`` then
``encode`` on every encoder in the REGISTRY and saves ``Z_{split}.parquet``
files that are directly consumed by ``pipelines.clustering_ablation.run_ablation``.

Array conventions
-----------------
- ``windows`` : ``np.ndarray``, shape ``[n_windows, T, N]``
    ``T`` = window_size (months), ``N`` = number of features (122 for FRED-MD).
- Return value of ``encode`` : ``np.ndarray``, shape ``[n_windows, d]``
    ``d`` is encoder-specific (e.g., 32 for TS2Vec, 256 for MOMENT-Small).
    ``d`` should be ≤ 512 for efficient downstream clustering.

Zero-shot encoders (MOMENT, MOIRAI): ``fit`` is a no-op.
Trainable encoders (TS2Vec, PatchTST, …): ``fit`` trains on train-split windows.
Classical encoders (HMM, MS-VAR, BOCPD): produce soft cluster probabilities
    as the embedding vector (shape ``[n_windows, K]``), where K = n_regimes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class AltEncoder(ABC):
    """Base class for all alternative encoders in Phase E.

    Attributes
    ----------
    name : str
        Short identifier used in output paths and CSVs.
    tier : str
        One of ``"zero-shot"``, ``"trainable"``, ``"classical"``.
    d_out : int
        Embedding dimensionality returned by ``encode``.
    requires_gpu : bool
        If True, only run when a CUDA device is available.
    """

    name: ClassVar[str]
    tier: ClassVar[str]
    d_out: ClassVar[int]
    requires_gpu: ClassVar[bool] = False

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Train the encoder on the training-split windows.

        Default is a no-op for zero-shot encoders.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]
        seed : int
            Random seed for reproducibility.
        """

    @abstractmethod
    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Extract embeddings for a set of windows.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]

        Returns
        -------
        np.ndarray, shape [n_windows, d_out]
        """

    # ── helpers ──────────────────────────────────────────────────────────

    def save_split_parquets(
        self,
        Z_train: np.ndarray,
        Z_val: np.ndarray,
        Z_test: np.ndarray,
        train_dates: pd.DatetimeIndex,
        val_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        out_dir: Path,
    ) -> None:
        """Write ``Z_{train,val,test}.parquet`` in the format expected by
        ``pipelines.clustering_ablation.run_ablation``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for Z, dates, name in (
            (Z_train, train_dates, "train"),
            (Z_val, val_dates, "val"),
            (Z_test, test_dates, "test"),
        ):
            d = Z.shape[1]
            cols = [f"z_{i}" for i in range(d)]
            df = pd.DataFrame(Z.astype(np.float32), columns=cols)
            df.insert(0, "date", dates)
            df.to_parquet(out_dir / f"Z_{name}.parquet", index=False)

    @classmethod
    def is_available(cls) -> bool:
        """Return True when all optional dependencies are installed."""
        return True
