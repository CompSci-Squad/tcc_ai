"""Download FRED-MD CSV vintage and write a SHA-256 hash sidecar."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

FRED_MD_BASE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md"
DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")
DEFAULT_VINTAGE = "2026-03"


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_fred_md(vintage: str, output_dir: Path) -> tuple[Path, Path]:
    """Download a FRED-MD vintage CSV; write the file and a ``.sha256`` sidecar.

    The CDN rejects the default urllib User-Agent, hence httpx with an explicit
    ``User-Agent`` header.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    year, month = vintage.split("-")
    csv_name = f"fred_md_{year}_{month}.csv"
    csv_path = output_dir / csv_name
    hash_path = output_dir / f"{csv_name.replace('.csv', '.sha256')}"

    url = f"{FRED_MD_BASE_URL}/monthly/{vintage}-md.csv"
    logger.info("Downloading FRED-MD vintage %s from %s", vintage, url)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        resp = client.get(url, headers={"User-Agent": "tcc-itransformer/1.0"})
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)
    logger.info("Saved CSV to %s (%d bytes)", csv_path, csv_path.stat().st_size)

    sha = compute_sha256(csv_path)
    hash_path.write_text(sha + "\n")
    logger.info("SHA-256: %s -> %s", sha, hash_path)
    return csv_path, hash_path


def download_nber_usrec(output: Path) -> Path:
    """Download the NBER USREC indicator CSV from FRED and write a SHA-256 sidecar."""
    from urllib.request import urlretrieve

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading NBER USREC from %s", url)
    urlretrieve(url, output)  # noqa: S310 - trusted FRED endpoint
    digest = compute_sha256(output)
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.name}\n")
    logger.info("Saved %s (sha256=%s)", output, digest[:16])
    return output
