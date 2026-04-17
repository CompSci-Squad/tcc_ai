"""Data loading, transformation, and windowing modules."""

from __future__ import annotations

from tcc_itransformer.data.dataset import FREDMDWindowDataset
from tcc_itransformer.data.fred_md import (
    apply_tcode,
    load_fred_md,
    remove_outliers,
    transform_panel,
    verify_sha256,
)
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    scale_splits,
    split_by_date,
)

__all__ = [
    "FREDMDWindowDataset",
    "apply_tcode",
    "create_windows",
    "drop_high_nan_series",
    "fit_scaler",
    "forward_fill_nans",
    "load_fred_md",
    "remove_outliers",
    "scale_splits",
    "split_by_date",
    "transform_panel",
    "verify_sha256",
]
