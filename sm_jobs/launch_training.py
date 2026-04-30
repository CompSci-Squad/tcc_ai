"""Launch a SageMaker Training Job for one experiment configuration.

Usage:
    python sagemaker/launch_training.py \
        --config configs/default.yaml \
        --bucket tcc-regime-etl-sagemaker \
        --role arn:aws:iam::ACCT:role/SageMakerExecRole \
        --instance-type ml.g4dn.xlarge

The Estimator ships the entire `tcc_ai/` directory as the source bundle so
`scripts/run_single.py` and `src/tcc_itransformer/` are available inside
the container.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_image_digest(image_uri: str, region: str) -> str:
    """Resolve a mutable ECR tag (e.g. ``:latest``) to an immutable
    ``@sha256:...`` reference so the training job is reproducible.

    Falls back to the original URI on any error (best-effort; logged).
    """
    if "@sha256:" in image_uri or ".dkr.ecr." not in image_uri:
        return image_uri
    try:
        import boto3  # local import so the script still loads without boto3
        repo_part, _, tag = image_uri.partition(":")
        registry, _, repo = repo_part.partition("/")
        ecr = boto3.client("ecr", region_name=region)
        resp = ecr.describe_images(
            repositoryName=repo,
            imageIds=[{"imageTag": tag or "latest"}],
        )
        digest = resp["imageDetails"][0]["imageDigest"]
        pinned = f"{registry}/{repo}@{digest}"
        logger.info("Pinned image %s -> %s", image_uri, pinned)
        return pinned
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not resolve digest for %s (%s); using mutable tag.", image_uri, exc)
        return image_uri


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--bucket", required=True, help="Artifacts bucket (jobs/, model.tar.gz)")
    p.add_argument(
        "--data-bucket",
        default="",
        help="Bucket holding the ETL-v2 panel. Defaults to --bucket (legacy).",
    )
    p.add_argument("--role", required=True, help="SageMaker execution IAM role ARN")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--instance-type", default="ml.m7i.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--max-run-seconds", type=int, default=4 * 60 * 60)
    p.add_argument("--data-prefix", default="fred_md/transformed/year=2026/month=04")
    p.add_argument("--usrec-prefix", default="snapshots/nber_usrec.csv")
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    p.add_argument("--experiment-name", default="itransformer-sagemaker")
    p.add_argument("--image-uri", default="", help="Custom training image URI (optional)")
    p.add_argument(
        "--keep-alive-seconds",
        type=int,
        default=1800,
        help="SageMaker warm pool TTL (0 disables).",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and return immediately (for parallel sweeps). Prints JOB_NAME=<name> to stdout.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import sagemaker
        from sagemaker.inputs import TrainingInput
        from sagemaker.pytorch import PyTorch
    except ImportError as exc:
        raise SystemExit(
            "sagemaker SDK required: pip install sagemaker"
        ) from exc

    boto_session = None
    sm_session = sagemaker.Session(default_bucket=args.bucket)

    job_name = f"itransformer-{int(time.time())}-{secrets.token_hex(2)}"
    output_path = f"s3://{args.bucket}/jobs/{job_name}/output"

    # Stage only the files SageMaker needs into a tempdir so we don't upload
    # .venv/, results/, data/, mlruns/, notebooks/, .git/, __pycache__, etc.
    repo_root = Path(__file__).resolve().parent.parent
    staging = Path(tempfile.mkdtemp(prefix="sm-source-"))
    include_dirs = ["sm_jobs", "src", "scripts", "configs"]
    include_files: list[str] = []  # NOTE: do NOT include pyproject.toml — SM toolkit will pip-install it and shadow the entrypoint
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
    for d in include_dirs:
        src = repo_root / d
        if src.exists():
            shutil.copytree(src, staging / d, ignore=ignore)
    for f in include_files:
        src = repo_root / f
        if src.exists():
            shutil.copy2(src, staging / f)
    # Promote entrypoint to staging root so SM's cwd matches configs/ path.
    shutil.copy2(repo_root / "sm_jobs" / "train_entrypoint.py", staging / "train_entrypoint.py")
    staged_size_mb = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()) / 1e6
    logger.info("Staged source bundle: %s (%.1f MB)", staging, staged_size_mb)

    estimator_kwargs: dict[str, object] = dict(
        entry_point="train_entrypoint.py",
        source_dir=str(staging),
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        max_run=args.max_run_seconds,
        output_path=output_path,
        sagemaker_session=sm_session,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {"Name": "train_loss", "Regex": r"train_loss[=:\s]+([0-9.]+)"},
            {"Name": "val_loss",   "Regex": r"val_loss[=:\s]+([0-9.]+)"},
            {"Name": "test_loss",  "Regex": r"test_loss[=:\s]+([0-9.]+)"},
            {"Name": "dbcv",       "Regex": r"dbcv[=:\s]+(-?[0-9.]+)"},
            {"Name": "nber_f1",    "Regex": r"nber_f1[=:\s]+([0-9.]+)"},
        ],
        tags=[
            {"Key": "Project",     "Value": "tcc-regime-etl"},
            {"Key": "Component",   "Value": "itransformer-ae"},
            {"Key": "ConfigPath",  "Value": args.config},
            {"Key": "DataPrefix",  "Value": args.data_prefix},
        ],
        hyperparameters={
            k: v for k, v in {
                "config": args.config,
                "mlflow-uri": args.mlflow_uri,
                "experiment-name": args.experiment_name,
            }.items() if v
        },
        environment={
            k: v for k, v in {
                "MLFLOW_TRACKING_URI": args.mlflow_uri,
                "MLFLOW_EXPERIMENT_NAME": args.experiment_name,
            }.items() if v
        },
    )
    if args.keep_alive_seconds and args.keep_alive_seconds > 0:
        estimator_kwargs["keep_alive_period_in_seconds"] = args.keep_alive_seconds
    if args.image_uri:
        estimator_kwargs["image_uri"] = resolve_image_digest(args.image_uri, args.region)
    else:
        estimator_kwargs["framework_version"] = "2.4.0"
        estimator_kwargs["py_version"] = "py311"

    estimator = PyTorch(**estimator_kwargs)

    data_bucket = args.data_bucket or args.bucket
    inputs = {
        "training": TrainingInput(
            s3_data=f"s3://{data_bucket}/{args.data_prefix}/",
            distribution="FullyReplicated",
            input_mode="File",
        ),
        "usrec": TrainingInput(
            s3_data=f"s3://{args.bucket}/{args.usrec_prefix}",
            input_mode="File",
        ),
    }

    logger.info("Launching SageMaker job %s on %s", job_name, args.instance_type)
    estimator.fit(inputs=inputs, job_name=job_name, wait=not args.no_wait)
    if args.no_wait:
        # Stable, machine-parseable line for orchestrators (xargs/awk).
        print(f"JOB_NAME={job_name}", flush=True)
        logger.info("Submitted (no-wait). Outputs will land at: %s", output_path)
    else:
        logger.info("Job complete. Outputs: %s", output_path)


if __name__ == "__main__":
    main()
