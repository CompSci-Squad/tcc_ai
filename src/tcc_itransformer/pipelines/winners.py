"""Pick the stage-1 and stage-2 winners from completed sweeps.

Stage 1: query SageMaker for completed jobs in ``.sm_sweep_jobs.txt``,
download ``output.tar.gz``, parse ``history.json``, rank by ``best_val_loss``,
write ``results/stage1_summary.csv`` + ``configs/stage1_winner.yaml``.

Stage 2: read ``results/stage2_summary.csv`` (produced upstream), apply the
pre-registered tiebreak (K asc, params asc, best_epoch asc) within a
``best_val_loss`` band, write ``configs/stage2_winner.yaml``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import tarfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CFG_PATTERN = re.compile(r"W(?P<W>\d+)_d(?P<d>\d+)_K(?P<K>\d+)")


# ---------- stage 1 ----------


def _load_jobs(path: Path) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        seen.setdefault(parts[0], parts[1])
    return list(seen.items())


def _fetch_status(sm, job_name: str) -> str:
    try:
        return sm.describe_training_job(
            TrainingJobName=job_name,
        )["TrainingJobStatus"]
    except Exception as exc:
        logger.warning("[%s] describe failed: %s", job_name, exc)
        return "Unknown"


def _download_metrics(s3, bucket: str, job_name: str, cache_dir: Path) -> dict | None:
    target_dir = cache_dir / job_name
    metrics_path = target_dir / "history.json"
    if not metrics_path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        # SageMaker output path: jobs/<JOB>/output/<JOB>/output/output.tar.gz
        key = f"jobs/{job_name}/output/{job_name}/output/output.tar.gz"
        tar_path = target_dir / "output.tar.gz"
        try:
            s3.download_file(bucket, key, str(tar_path))
        except Exception as exc:
            logger.warning("[%s] download failed: %s", job_name, exc)
            return None
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.name.endswith("history.json"):
                    tf.extract(member, target_dir)
                    extracted = target_dir / member.name
                    if extracted != metrics_path:
                        shutil.copy(extracted, metrics_path)
                    break
    if not metrics_path.exists():
        logger.warning("[%s] no history.json in archive", job_name)
        return None
    h = json.loads(metrics_path.read_text())
    val_losses = h.get("val_losses") or []
    train_losses = h.get("train_losses") or []
    if not val_losses:
        return None
    best_epoch = h.get(
        "best_epoch",
        int(min(range(len(val_losses)), key=lambda i: val_losses[i])),
    )
    return {
        "best_val_loss": float(min(val_losses)),
        "final_val_loss": float(val_losses[-1]),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "best_epoch": best_epoch,
        "stopped_epoch": h.get("stopped_epoch"),
        "n_epochs": len(val_losses),
    }


def pick_stage1_winner(
    *,
    jobs_file: Path = Path(".sm_sweep_jobs.txt"),
    bucket: str = "tcc-regime-etl-sagemaker",
    region: str = "us-east-1",
    summary_csv: Path = Path("results/stage1_summary.csv"),
    winner_yaml: Path = Path("configs/stage1_winner.yaml"),
    cache_dir: Path = Path("results/sm_outputs"),
) -> int:
    import boto3

    if not jobs_file.exists():
        logger.error("missing %s", jobs_file)
        return 1
    jobs = _load_jobs(jobs_file)
    logger.info("=== %d unique jobs in %s ===", len(jobs), jobs_file)

    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    rows: list[dict] = []
    for job_name, cfg_path in jobs:
        status = _fetch_status(sm, job_name)
        row: dict = {"job": job_name, "config": cfg_path, "status": status}
        if status == "Completed":
            metrics = _download_metrics(s3, bucket, job_name, cache_dir)
            if metrics:
                row.update(**metrics)
        rows.append(row)
        logger.info(
            "  %s  %s  %s  best_val=%s",
            job_name, status, Path(cfg_path).stem,
            row.get("best_val_loss"),
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    field_order = [
        "config", "job", "status", "best_val_loss", "final_val_loss",
        "final_train_loss", "best_epoch", "stopped_epoch", "n_epochs",
    ]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("best_val_loss") or 1e9)):
            w.writerow(r)
    logger.info("=== wrote %s ===", summary_csv)

    completed = [r for r in rows if r.get("best_val_loss") is not None]
    if not completed:
        logger.error("no completed jobs with metrics")
        return 1
    winner = min(completed, key=lambda r: r["best_val_loss"])
    logger.info(
        "=== WINNER: %s (best_val=%.4f, best_epoch=%s) ===",
        Path(winner["config"]).stem, winner["best_val_loss"],
        winner.get("best_epoch"),
    )

    cfg = yaml.safe_load(Path(winner["config"]).read_text())
    cfg["_stage1_winner"] = {
        "job": winner["job"],
        "best_val_loss": winner["best_val_loss"],
        "final_val_loss": winner.get("final_val_loss"),
        "best_epoch": winner.get("best_epoch"),
        "source_config": winner["config"],
    }
    winner_yaml.parent.mkdir(parents=True, exist_ok=True)
    winner_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger.info("=== wrote %s ===", winner_yaml)
    return 0


# ---------- stage 2 ----------


def _parse_cfg_name(cfg_path: str) -> dict[str, int]:
    m = CFG_PATTERN.search(Path(cfg_path).stem)
    if not m:
        return {"W": -1, "d": -1, "K": -1}
    return {k: int(v) for k, v in m.groupdict().items()}


def _approx_params(W: int, d: int) -> int:
    """Crude size proxy: dominated by attention QKV + FFN of width d over W tokens."""
    return W * d * d * 4


def _load_stage2_rows(csv_path: Path) -> list[dict]:
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


def pick_stage2_winner(
    *,
    summary_csv: Path = Path("results/stage2_summary.csv"),
    winner_yaml: Path = Path("configs/stage2_winner.yaml"),
    tol: float = 1e-4,
) -> int:
    rows = _load_stage2_rows(summary_csv)
    if not rows:
        logger.error("no completed rows")
        return 1

    rows.sort(key=lambda r: r["best_val_loss"])
    best_loss = rows[0]["best_val_loss"]
    band = [r for r in rows if r["best_val_loss"] - best_loss <= tol]

    print(f"=== {len(rows)} completed configs ===")
    print(f"min best_val_loss = {best_loss:.6f}")
    print(f"tiebreak band (within {tol}): {len(band)} configs")
    for r in band:
        print(
            f"  {Path(r['config']).stem:<20} val={r['best_val_loss']:.6f} "
            f"K={r['K']} params~{r['approx_params']} best_epoch={r['best_epoch']}"
        )

    band.sort(key=lambda r: (r["K"], r["approx_params"], r["best_epoch"]))
    winner = band[0]
    print(
        f"\n=== WINNER (pre-registered tiebreak): "
        f"{Path(winner['config']).stem} ==="
    )
    print(
        f"  best_val_loss={winner['best_val_loss']:.6f}  K={winner['K']}  "
        f"params~{winner['approx_params']}  best_epoch={winner['best_epoch']}"
    )

    cfg_path = Path(winner["config"])
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["_stage2_winner"] = {
        "job": winner["job"],
        "best_val_loss": winner["best_val_loss"],
        "best_epoch": winner["best_epoch"],
        "tiebreak": "K_asc, params_asc, best_epoch_asc (pre-registered)",
        "source_config": winner["config"],
        "band_size": len(band),
        "band_tol": tol,
    }
    winner_yaml.parent.mkdir(parents=True, exist_ok=True)
    winner_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"=== wrote {winner_yaml} ===")
    return 0
