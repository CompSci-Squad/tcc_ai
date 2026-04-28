"""Integration test for the SageMaker training entrypoint.

Verifies the entrypoint resolves SM_* environment variables, locates the
parquet panel mounted at SM_CHANNEL_TRAINING, and persists the expected
artifacts to SM_MODEL_DIR / SM_OUTPUT_DATA_DIR — without actually launching
a real SageMaker job (everything runs locally with fakes).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_synthetic_panel(out_path: Path, n_timesteps: int = 240, n_series: int = 8) -> None:
    """Write a FRED-MD-format CSV: header row, tcode row, then monthly data."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-01", periods=n_timesteps, freq="MS")
    cols = [f"s{i:02d}" for i in range(n_series)]
    data = rng.standard_normal((n_timesteps, n_series))
    # Header
    lines = ["sasdate," + ",".join(cols)]
    # tcode row (1 = no transform, the simplest)
    lines.append("Transform:," + ",".join(["1"] * n_series))
    # Data rows
    for d, row in zip(dates, data):
        lines.append(d.strftime("%m/%d/%Y") + "," + ",".join(f"{v:.6f}" for v in row))
    out_path.write_text("\n".join(lines) + "\n")


def _make_synthetic_usrec(out_path: Path) -> None:
    dates = pd.date_range("2005-01-01", periods=240, freq="MS")
    df = pd.DataFrame({"date": dates, "USREC": [0] * 240})
    df.to_csv(out_path, index=False)


@pytest.mark.integration
def test_sagemaker_entrypoint_runs_locally(tmp_path: Path) -> None:
    """Run sagemaker/train_entrypoint.py with fake SM_* env vars."""
    # --- arrange directory layout that mimics SageMaker container mounts ---
    training_dir = tmp_path / "input" / "data" / "training"
    usrec_dir = tmp_path / "input" / "data" / "usrec"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    for d in (training_dir, usrec_dir, model_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    _make_synthetic_panel(training_dir / "panel.csv")
    _make_synthetic_usrec(usrec_dir / "nber_usrec.csv")

    # --- write a minimal config tuned for fast CPU execution ---
    cfg = {
        "experiment_name": "sm_smoke",
        "results_dir": str(tmp_path / "results"),
        "data_path": str(training_dir),
        "nber_usrec_path": str(usrec_dir / "nber_usrec.csv"),
        "seed": 42,
        "window_size": 6,
        "horizon": 1,
        "stride": 1,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "latent_dim": 4,
        "n_clusters": 3,
        "batch_size": 8,
        "max_epochs": 2,
        "patience": 5,
        "n_pca_max": 4,
        "umap_n_components": 3,
        "umap_n_neighbors": 5,
        "umap_min_dist": 0.0,
        "hdbscan_min_cluster_sizes": [3, 5],
        "hdbscan_min_samples_grid": [None, 1],
        "hdbscan_max_noise_fraction": 0.9,
        "explain_top_k": 3,
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # --- env mimicking a SageMaker training container ---
    env = os.environ.copy()
    env.update({
        "SM_CHANNEL_TRAINING": str(training_dir),
        "SM_CHANNEL_USREC": str(usrec_dir),
        "SM_MODEL_DIR": str(model_dir),
        "SM_OUTPUT_DATA_DIR": str(output_dir),
        "MLFLOW_TRACKING_URI": f"file:{tmp_path / 'mlruns'}",
        "MLFLOW_EXPERIMENT_NAME": "sm_smoke",
        # Ensure repo `src/` is importable when running entrypoint as a script.
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + str(Path(__file__).resolve().parents[2]),
    })

    entrypoint = Path(__file__).resolve().parents[2] / "sagemaker" / "train_entrypoint.py"

    result = subprocess.run(
        [sys.executable, str(entrypoint), "--config", str(cfg_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Entrypoint failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # --- assert expected outputs were persisted ---
    metrics_path = model_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json not written to SM_MODEL_DIR"
    metrics = json.loads(metrics_path.read_text())
    assert "silhouette" in metrics or "kw_n_significant" in metrics

    assert (output_dir / "history.json").exists(), "history.json not written"
    assert (model_dir / "config.yaml").exists(), "config.yaml not written"
