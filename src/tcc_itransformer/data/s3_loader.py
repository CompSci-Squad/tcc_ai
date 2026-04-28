"""S3 panel loader for SageMaker training jobs.

Reads parquet panels written by tcc_etl into:

    s3://<bucket>/fred_md/transformed/year=YYYY/month=MM/*.parquet

In a SageMaker Training Job, the SDK auto-mounts the S3 input channel
under SM_CHANNEL_TRAINING (typically /opt/ml/input/data/training), so
during training we read parquet files from that local path; during
local development we can read directly from S3 via fsspec/s3fs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_S3_PREFIX = "fred_md/transformed"


def _read_parquet_glob(root: str | Path) -> pd.DataFrame:
    """Read every *.parquet under root and concat (s3:// or local)."""
    root_str = str(root)
    if root_str.startswith("s3://"):
        # fsspec/s3fs handles s3 globbing transparently for pandas+pyarrow.
        try:
            import s3fs  # noqa: F401  # ensure backend is importable
        except ImportError as exc:  # pragma: no cover
            raise ImportError("s3fs required to read s3://; pip install s3fs") from exc
        # Use pyarrow dataset via pandas
        return pd.read_parquet(root_str)
    p = Path(root_str)
    files = sorted(p.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {p}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_panel_from_s3(
    bucket: str,
    *,
    prefix: str = DEFAULT_S3_PREFIX,
    date_column: str = "date",
) -> pd.DataFrame:
    """Load the full transformed panel from S3 (or SageMaker channel mount).

    Resolution order:
      1. If env var SM_CHANNEL_TRAINING is set, read from that local path
         (we are inside a SageMaker Training Job).
      2. Otherwise read from `s3://{bucket}/{prefix}/`.

    Args:
        bucket: S3 bucket name (ignored when SM_CHANNEL_TRAINING is set).
        prefix: Key prefix inside the bucket.
        date_column: Column to set as DatetimeIndex.

    Returns:
        Wide-format DataFrame indexed by date, one column per series.
    """
    sm_channel = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_channel:
        logger.info("Reading panel from SageMaker channel %s", sm_channel)
        df = _read_parquet_glob(sm_channel)
    else:
        uri = f"s3://{bucket}/{prefix}"
        logger.info("Reading panel from %s", uri)
        df = _read_parquet_glob(uri)

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
    return df


def resolve_output_dir() -> Path:
    """Return the directory where artifacts must be written.

    On SageMaker:
        - SM_MODEL_DIR (model.tar.gz contents) for model weights
        - SM_OUTPUT_DATA_DIR for auxiliary outputs (figures, jsons)
    Locally: ./results/<run>
    """
    return Path(os.environ.get("SM_MODEL_DIR", "results/local_run"))


def resolve_aux_output_dir() -> Path:
    return Path(os.environ.get("SM_OUTPUT_DATA_DIR", "results/local_run/aux"))
