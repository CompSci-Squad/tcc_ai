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
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True, help="SageMaker execution IAM role ARN")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--max-run-seconds", type=int, default=4 * 60 * 60)
    p.add_argument("--data-prefix", default="fred_md/transformed")
    p.add_argument("--usrec-prefix", default="snapshots/nber_usrec.csv")
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    p.add_argument("--experiment-name", default="itransformer-sagemaker")
    p.add_argument("--image-uri", default="", help="Custom training image URI (optional)")
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

    job_name = f"itransformer-{int(time.time())}"
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
    if args.image_uri:
        estimator_kwargs["image_uri"] = args.image_uri
    else:
        estimator_kwargs["framework_version"] = "2.4.0"
        estimator_kwargs["py_version"] = "py311"

    estimator = PyTorch(**estimator_kwargs)

    inputs = {
        "training": TrainingInput(
            s3_data=f"s3://{args.bucket}/{args.data_prefix}/",
            distribution="FullyReplicated",
            input_mode="File",
        ),
        "usrec": TrainingInput(
            s3_data=f"s3://{args.bucket}/{args.usrec_prefix}",
            input_mode="File",
        ),
    }

    logger.info("Launching SageMaker job %s on %s", job_name, args.instance_type)
    estimator.fit(inputs=inputs, job_name=job_name, wait=True)
    logger.info("Job complete. Outputs: %s", output_path)


if __name__ == "__main__":
    main()
