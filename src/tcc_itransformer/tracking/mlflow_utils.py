"""MLflow tracking utilities for experiment logging."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import mlflow

from tcc_itransformer.config import ExperimentConfig

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Set tracking URI and create/get experiment.

    Args:
        tracking_uri: MLflow tracking URI, e.g. ``'file:./results/mlruns'``.
        experiment_name: Name of the MLflow experiment.

    Returns:
        The experiment ID as a string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info(
        "MLflow experiment '%s' (id=%s) at %s",
        experiment_name,
        experiment_id,
        tracking_uri,
    )
    return experiment_id


def log_config(config: ExperimentConfig) -> None:
    """Log all config parameters and git commit to the active MLflow run.

    Args:
        config: Experiment configuration to log.
    """
    mlflow.log_params(config.model_dump_for_mlflow())
    mlflow.set_tag("git_commit", get_git_commit())


def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float) -> None:
    """Log train and validation loss at a given epoch step.

    Args:
        epoch: Current epoch number (used as step).
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch.
    """
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)


def log_evaluation_metrics(metrics: dict[str, float]) -> None:
    """Log a dictionary of evaluation metrics to the active run.

    Args:
        metrics: Mapping of metric name to float value.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifact_file(path: str | Path) -> None:
    """Log a file as an MLflow artifact.

    Args:
        path: Path to the file to log.
    """
    mlflow.log_artifact(str(path))
