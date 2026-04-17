"""PyTorch Dataset for windowed FRED-MD data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FREDMDWindowDataset(Dataset):
    """PyTorch Dataset wrapping pre-computed windows.

    Each sample is a single window of multivariate time-series data returned
    as a float32 tensor suitable for the iTransformer autoencoder.

    Args:
        windows: Pre-computed windows of shape (n_windows, window_size, n_features).
    """

    def __init__(self, windows: np.ndarray) -> None:
        self._windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a single window and its index.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Tuple of (x, idx) where x has shape (window_size, n_features)
            and dtype float32.
        """
        return self._windows[idx], idx
