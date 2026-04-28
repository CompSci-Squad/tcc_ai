"""SageMaker Training Job entrypoint.

The container is invoked by SageMaker with:
    python sagemaker/train_entrypoint.py --config <hyperparam>

Channels mounted automatically:
    SM_CHANNEL_TRAINING       -> /opt/ml/input/data/training      (parquet panel)
    SM_CHANNEL_USREC          -> /opt/ml/input/data/usrec         (NBER snapshot)

Outputs (written by sagemaker.pytorch.PyTorch Estimator conventions):
    SM_MODEL_DIR              -> /opt/ml/model         (model.tar.gz contents)
    SM_OUTPUT_DATA_DIR        -> /opt/ml/output/data   (output.tar.gz)

Hyperparameters supplied to the Estimator are passed as CLI flags by SM.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add /opt/ml/code to sys.path so `tcc_itransformer` resolves when the source
# tree is shipped under SAGEMAKER_SUBMIT_DIRECTORY.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow

from tcc_itransformer.config import ExperimentConfig
from tcc_itransformer.tracking.mlflow_utils import (
    log_config,
    log_evaluation_metrics,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=os.environ.get("SM_HP_CONFIG", "configs/default.yaml"))
    p.add_argument("--mlflow-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    p.add_argument("--experiment-name", type=str, default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "itransformer-sagemaker"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Inside SM, code dir is current working directory; try resolving there.
        alt = Path("/opt/ml/code") / args.config
        if alt.exists():
            config_path = alt
    cfg = ExperimentConfig.from_yaml(config_path)

    # Resolve data path: prefer SM channel mount.
    sm_training = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_training:
        # Prefer FRED-MD-format CSV; fall back to parquet panel.
        candidates = sorted(Path(sm_training).rglob("*.csv"))
        if candidates:
            cfg = cfg.model_copy(update={"data_path": str(candidates[0])})
        else:
            parquets = sorted(Path(sm_training).rglob("*.parquet"))
            if parquets:
                cfg = cfg.model_copy(update={"data_path": str(parquets[0].parent)})
        logger.info("Overriding data_path -> %s", cfg.data_path)

    # NBER channel
    sm_usrec = os.environ.get("SM_CHANNEL_USREC")
    if sm_usrec:
        usrec_files = list(Path(sm_usrec).rglob("*.csv"))
        if usrec_files:
            cfg = cfg.model_copy(update={"nber_usrec_path": str(usrec_files[0])})
            logger.info("Overriding nber_usrec_path -> %s", cfg.nber_usrec_path)

    # MLflow: prefer SageMaker-managed tracking server if URI provided.
    if args.mlflow_uri:
        experiment_id = setup_mlflow(args.mlflow_uri, args.experiment_name)
    else:
        experiment_id = setup_mlflow(f"file:./{cfg.results_dir}/mlruns", cfg.experiment_name)

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    aux_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    model_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"sm_W{cfg.window_size}_d{cfg.latent_dim}_K{cfg.n_clusters}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        log_config(cfg)
        # Local import avoids loading torch before sys.path is configured.
        from scripts.run_single import run_full_pipeline  # type: ignore[import-not-found]

        metrics = run_full_pipeline(cfg, model_dir=model_dir, aux_dir=aux_dir)
        log_evaluation_metrics(metrics)

    logger.info("SageMaker training job complete: %s", run_name)


if __name__ == "__main__":
    main()
