"""PyTorch Dataset for windowed FRED-MD data."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FREDMDWindowDataset(Dataset):
    """PyTorch Dataset wrapping pre-computed windows.

    Each sample is a single window of multivariate time-series data returned
    as a float32 tensor suitable for the iTransformer autoencoder.

    D7 imputation policies (controlled by ``drop_imputed`` and ``return_mask``):

    * ``drop_imputed=True`` (D7.a, evaluation/test policy): drop windows whose
      target row (last row) has *too many* imputed cells. The fraction
      tolerated is controlled by ``min_observed_fraction`` — by default
      ``0.95``, meaning a target row is rejected only when more than 5% of
      its cells are imputed. The default of 0.95 (instead of 1.0) prevents
      a single late-publishing series from invalidating otherwise
      well-observed test windows.
    * ``return_mask=True`` (D7.c, training policy): keep all windows and
      surface the per-cell imputation mask alongside each sample so the
      training loop can apply ``masked_reconstruction_loss`` and grade the
      autoencoder only on observed cells.

    The two flags are orthogonal. Typical setup:
        * train/val: ``drop_imputed=False, return_mask=True``
        * test:      ``drop_imputed=True,  return_mask=False, min_observed_fraction=0.95``

    Args:
        windows: Pre-computed windows of shape (n_windows, window_size, n_features).
        mask_windows: Optional Boolean mask of the same shape. ``True`` marks
            cells that were imputed by the ETL EM-PCA step.
        drop_imputed: D7.a policy. If True and ``mask_windows`` is given,
            drop windows whose target row has too many imputed cells.
        min_observed_fraction: Lower bound on the fraction of *observed*
            (non-imputed) cells in the target row. Windows with a smaller
            observed fraction are dropped. Use ``1.0`` to recover the
            strict "any imputed cell rejects the window" behaviour.
        return_mask: D7.c policy. If True, ``__getitem__`` returns
            ``(window, mask, idx)`` instead of ``(window, idx)``. Requires
            ``mask_windows`` to be provided.
    """

    def __init__(
        self,
        windows: np.ndarray,
        mask_windows: np.ndarray | None = None,
        *,
        drop_imputed: bool = True,
        min_observed_fraction: float = 0.95,
        return_mask: bool = False,
    ) -> None:
        if not 0.0 <= min_observed_fraction <= 1.0:
            raise ValueError(
                f"min_observed_fraction must be in [0, 1], got {min_observed_fraction}"
            )
        if return_mask and mask_windows is None:
            raise ValueError("return_mask=True requires mask_windows to be provided")

        if mask_windows is not None and mask_windows.shape != windows.shape:
            raise ValueError(
                f"mask_windows shape {mask_windows.shape} != windows shape {windows.shape}"
            )

        if mask_windows is not None and drop_imputed:
            n_features = mask_windows.shape[2]
            observed_frac = 1.0 - mask_windows[:, -1, :].mean(axis=1)
            keep = observed_frac >= min_observed_fraction
            n_dropped = int((~keep).sum())
            if n_dropped:
                logger.info(
                    "D7.a target-row filter: dropped %d/%d windows "
                    "(min_observed_fraction=%.3f, n_features=%d)",
                    n_dropped, len(windows), min_observed_fraction, n_features,
                )
            windows = windows[keep]
            mask_windows = mask_windows[keep]
            self._kept_indices = np.where(keep)[0]
        else:
            self._kept_indices = np.arange(len(windows))

        self._windows = torch.from_numpy(windows).float()
        self._return_mask = return_mask
        if return_mask:
            self._mask = torch.from_numpy(mask_windows.astype(bool))
            n_imputed = int(self._mask.sum().item())
            n_total = int(self._mask.numel())
            logger.info(
                "D7.c masked-loss policy: %d/%d cells (%.2f%%) flagged imputed across %d kept windows",
                n_imputed, n_total, 100.0 * n_imputed / max(n_total, 1), len(self._windows),
            )
        else:
            self._mask = None

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int):
        """Return one window and its index.

        If ``return_mask=True`` was set, also returns the per-cell imputation
        mask so the trainer can apply masked MSE.
        """
        if self._return_mask:
            return self._windows[idx], self._mask[idx], idx
        return self._windows[idx], idx

    @property
    def kept_indices(self) -> np.ndarray:
        """Indices into the original window array that survived the mask filter."""
        return self._kept_indices
