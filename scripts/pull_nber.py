"""Download NBER USREC recession indicator snapshot from FRED.

Pre_projeto §4.4 validation layer 2: NBER USREC overlap requires a
reproducible local snapshot of the recession indicator series.

The series is fetched from the FRED CSV endpoint (no API key needed).

Usage:
    python scripts/pull_nber.py
    python scripts/pull_nber.py --output data/snapshots/nber_usrec_2026-04.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

NBER_USREC_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
DEFAULT_OUTPUT = Path("data/snapshots/nber_usrec.csv")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_usrec(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading NBER USREC from %s", NBER_USREC_URL)
    urlretrieve(NBER_USREC_URL, output)  # noqa: S310 — trusted FRED endpoint
    digest = _sha256(output)
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.name}\n")
    logger.info("Saved %s (sha256=%s)", output, digest[:16])
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    download_usrec(args.output)


if __name__ == "__main__":
    main()
