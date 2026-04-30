"""A4 pre-registered tiebreak winner picker for stage-2.

Reads `results/stage2_summary.csv` (produced by `pick_stage1_winner.py` style
pipeline). Selects the winner by:

  1. Minimum `best_val_loss`.
  2. If multiple configs are within `--tol` of the minimum, apply pre-registered
     tiebreak (in order):
        a. lowest K (configs/.../W{W}_d{d}_K{K}.yaml -> K),
        b. fewest model parameters (estimated from window x d_lat),
        c. earliest `best_epoch`.

Writes:
  - `<winner-yaml>`  copy of the winning config + `_stage2_winner` block.
  - prints the ranking and tiebreak trace.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

CFG_PATTERN = re.compile(r"W(?P<W>\d+)_d(?P<d>\d+)_K(?P<K>\d+)")


def _parse_cfg_name(cfg_path: str) -> dict[str, int]:
    m = CFG_PATTERN.search(Path(cfg_path).stem)
    if not m:
        return {"W": -1, "d": -1, "K": -1}
    return {k: int(v) for k, v in m.groupdict().items()}


def _approx_params(W: int, d: int) -> int:
    """Crude size proxy: dominated by attention QKV + FFN of width d over W tokens."""
    return W * d * d * 4


def _load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "Completed" or not r.get("best_val_loss"):
                continue
            cfg_meta = _parse_cfg_name(r["config"])
            rows.append({
                "config": r["config"],
                "job": r["job"],
                "best_val_loss": float(r["best_val_loss"]),
                "best_epoch": int(r["best_epoch"]) if r.get("best_epoch") else 10**9,
                **cfg_meta,
                "approx_params": _approx_params(cfg_meta["W"], cfg_meta["d"]),
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", default="results/stage2_summary.csv")
    p.add_argument("--winner-yaml", default="configs/stage2_winner.yaml")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="best_val_loss tolerance for tiebreak band")
    args = p.parse_args()

    rows = _load_rows(Path(args.summary_csv))
    if not rows:
        print("no completed rows", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["best_val_loss"])
    best_loss = rows[0]["best_val_loss"]
    band = [r for r in rows if r["best_val_loss"] - best_loss <= args.tol]

    print(f"=== {len(rows)} completed configs ===")
    print(f"min best_val_loss = {best_loss:.6f}")
    print(f"tiebreak band (within {args.tol}): {len(band)} configs")
    for r in band:
        print(f"  {Path(r['config']).stem:<20} val={r['best_val_loss']:.6f} "
              f"K={r['K']} params~{r['approx_params']} best_epoch={r['best_epoch']}")

    # Tiebreak: K asc, params asc, best_epoch asc.
    band.sort(key=lambda r: (r["K"], r["approx_params"], r["best_epoch"]))
    winner = band[0]
    print(f"\n=== WINNER (pre-registered tiebreak): {Path(winner['config']).stem} ===")
    print(f"  best_val_loss={winner['best_val_loss']:.6f}  K={winner['K']}  "
          f"params~{winner['approx_params']}  best_epoch={winner['best_epoch']}")

    cfg_path = Path(winner["config"])
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["_stage2_winner"] = {
        "job": winner["job"],
        "best_val_loss": winner["best_val_loss"],
        "best_epoch": winner["best_epoch"],
        "tiebreak": "K_asc, params_asc, best_epoch_asc (pre-registered)",
        "source_config": winner["config"],
        "band_size": len(band),
        "band_tol": args.tol,
    }
    out = Path(args.winner_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"=== wrote {out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
