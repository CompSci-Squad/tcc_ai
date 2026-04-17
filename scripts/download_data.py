"""Download FRED-MD data snapshot and compute SHA-256 hash.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --vintage 2026-04
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

FRED_MD_BASE_URL = "https://files.stlouisfed.org/files/htdocs/fred-md"
SNAPSHOT_DIR = Path("data/snapshots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FRED-MD data snapshot.")
    parser.add_argument(
        "--vintage",
        type=str,
        default="2026-04",
        help="FRED-MD vintage in YYYY-MM format (default: 2026-04).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SNAPSHOT_DIR),
        help="Directory to save the snapshot.",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_fred_md(vintage: str, output_dir: Path) -> tuple[Path, Path]:
    """Download FRED-MD CSV and save SHA-256 hash.

    Args:
        vintage: Vintage string in YYYY-MM format.
        output_dir: Directory to save files.

    Returns:
        Tuple of (csv_path, hash_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    year, month = vintage.split("-")
    csv_name = f"fred_md_{year}_{month}.csv"
    csv_path = output_dir / csv_name
    hash_path = output_dir / f"{csv_name.replace('.csv', '.sha256')}"

    # FRED-MD URL pattern: monthly/YYYY-MM.csv
    url = f"{FRED_MD_BASE_URL}/monthly/{vintage}.csv"
    logger.info("Downloading FRED-MD vintage %s from %s", vintage, url)

    urlretrieve(url, csv_path)  # noqa: S310
    logger.info("Saved CSV to %s (%d bytes)", csv_path, csv_path.stat().st_size)

    # Compute and save SHA-256
    sha = compute_sha256(csv_path)
    hash_path.write_text(sha + "\n")
    logger.info("SHA-256: %s → %s", sha, hash_path)

    return csv_path, hash_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    csv_path, hash_path = download_fred_md(args.vintage, Path(args.output_dir))
    logger.info("Done. CSV: %s, Hash: %s", csv_path, hash_path)


if __name__ == "__main__":
    main()
