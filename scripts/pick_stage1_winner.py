"""Pick the stage-1 winner from completed SageMaker training jobs.

Reads .sm_sweep_jobs.txt (job_name  config_path), queries SageMaker for status,
downloads output.tar.gz for completed jobs, parses final_metrics.json, ranks by
best_val_loss (lower=better), writes:
  - results/stage1_summary.csv   (full ranking table)
  - configs/stage1_winner.yaml   (copy of winner's config + frozen flag)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-file", default=".sm_sweep_jobs.txt")
    p.add_argument("--bucket", default="tcc-regime-etl-sagemaker")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--summary-csv", default="results/stage1_summary.csv")
    p.add_argument("--winner-yaml", default="configs/stage1_winner.yaml")
    p.add_argument("--cache-dir", default="results/sm_outputs")
    return p.parse_args()


def load_jobs(path: Path) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        job, cfg = parts[0], parts[1]
        seen.setdefault(job, cfg)
    return list(seen.items())


def fetch_status(sm, job_name: str) -> str:
    try:
        return sm.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
    except Exception as exc:
        print(f"  [{job_name}] describe failed: {exc}", file=sys.stderr)
        return "Unknown"


def download_metrics(s3, bucket: str, job_name: str, cache_dir: Path) -> dict | None:
    """Pull output.tar.gz, extract history.json, return summarized metrics."""
    target_dir = cache_dir / job_name
    metrics_path = target_dir / "history.json"
    if not metrics_path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        # Path nesting: jobs/<JOB>/output/<JOB>/output/output.tar.gz
        key = f"jobs/{job_name}/output/{job_name}/output/output.tar.gz"
        tar_path = target_dir / "output.tar.gz"
        try:
            s3.download_file(bucket, key, str(tar_path))
        except Exception as exc:
            print(f"  [{job_name}] download failed: {exc}", file=sys.stderr)
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
        print(f"  [{job_name}] no history.json in archive", file=sys.stderr)
        return None
    h = json.loads(metrics_path.read_text())
    val_losses = h.get("val_losses") or []
    train_losses = h.get("train_losses") or []
    if not val_losses:
        return None
    best_epoch = h.get("best_epoch", int(min(range(len(val_losses)), key=lambda i: val_losses[i])))
    return {
        "best_val_loss": float(min(val_losses)),
        "final_val_loss": float(val_losses[-1]),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "best_epoch": best_epoch,
        "stopped_epoch": h.get("stopped_epoch"),
        "n_epochs": len(val_losses),
    }


def main() -> int:
    args = parse_args()
    jobs_file = Path(args.jobs_file)
    if not jobs_file.exists():
        print(f"missing {jobs_file}", file=sys.stderr)
        return 1
    jobs = load_jobs(jobs_file)
    print(f"=== {len(jobs)} unique jobs in {jobs_file} ===")

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)
    cache_dir = Path(args.cache_dir)

    rows: list[dict] = []
    for job_name, cfg_path in jobs:
        status = fetch_status(sm, job_name)
        row: dict = {"job": job_name, "config": cfg_path, "status": status}
        if status == "Completed":
            metrics = download_metrics(s3, args.bucket, job_name, cache_dir)
            if metrics:
                row.update(
                    best_val_loss=metrics["best_val_loss"],
                    final_val_loss=metrics["final_val_loss"],
                    final_train_loss=metrics["final_train_loss"],
                    best_epoch=metrics["best_epoch"],
                    stopped_epoch=metrics["stopped_epoch"],
                    n_epochs=metrics["n_epochs"],
                )
        rows.append(row)
        print(f"  {job_name}  {status}  {Path(cfg_path).stem}  "
              f"best_val={row.get('best_val_loss')}")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    field_order = ["config", "job", "status", "best_val_loss", "final_val_loss",
                   "final_train_loss", "best_epoch", "stopped_epoch", "n_epochs"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("best_val_loss") or 1e9)):
            w.writerow(r)
    print(f"\n=== wrote {summary_path} ===")

    completed = [r for r in rows if r.get("best_val_loss") is not None]
    if not completed:
        print("no completed jobs with metrics", file=sys.stderr)
        return 1
    winner = min(completed, key=lambda r: r["best_val_loss"])
    print(f"\n=== WINNER: {Path(winner['config']).stem} "
          f"(best_val={winner['best_val_loss']:.4f}, best_epoch={winner.get('best_epoch')}) ===")

    cfg = yaml.safe_load(Path(winner["config"]).read_text())
    cfg["_stage1_winner"] = {
        "job": winner["job"],
        "best_val_loss": winner["best_val_loss"],
        "final_val_loss": winner.get("final_val_loss"),
        "best_epoch": winner.get("best_epoch"),
        "source_config": winner["config"],
    }
    out = Path(args.winner_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"=== wrote {out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
