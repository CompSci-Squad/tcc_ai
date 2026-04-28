"""Experiment configuration — Pydantic v2 models with YAML serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    """Complete experiment configuration for iTransformer autoencoder pipeline.

    All hyperparameters, data paths, and experiment metadata are defined here.
    This is the single source of truth — no magic numbers anywhere else.
    """

    # Reproducibility
    seed: int = Field(default=42, ge=0)

    # Architecture
    window_size: Literal[6, 12, 24] = 12
    latent_dim: int = Field(default=8, ge=4, le=12)
    d_model: Literal[32, 64] = 64
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=2, ge=1, le=4)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # Clustering
    n_clusters: int = Field(default=4, ge=2, le=10)

    # Training
    batch_size: int = Field(default=32, ge=8)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    max_epochs: int = Field(default=200, ge=1)
    patience: int = Field(default=10, ge=1)
    grad_clip: float = Field(default=1.0, ge=0)

    # PCA
    pca_variance_threshold: float = Field(default=0.9, gt=0, le=1)
    n_pca_max: int = Field(default=5, ge=1)

    # UMAP (principal — pre_projeto §4.3 Module 2)
    umap_n_components: int = Field(default=5, ge=2, le=20)
    umap_n_neighbors: int = Field(default=15, ge=2)
    umap_min_dist: float = Field(default=0.0, ge=0.0, le=1.0)

    # HDBSCAN (principal — pre_projeto §4.3 Module 3)
    hdbscan_min_cluster_sizes: list[int] = Field(default=[5, 8, 10, 15, 20])
    hdbscan_min_samples_grid: list[int | None] = Field(default=[None, 1, 5])
    hdbscan_max_noise_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    # Validation snapshots
    nber_usrec_path: str = "data/snapshots/nber_usrec.csv"
    explain_top_k: int = Field(default=5, ge=1, le=50)

    # Data splits
    train_end: str = "2018-12-01"
    val_end: str = "2021-12-01"

    # Paths
    data_path: str = "data/snapshots/fred_md_2026_04.csv"
    results_dir: str = "results"

    # MLflow
    experiment_name: str = "itransformer-autoencoder"

    @model_validator(mode="after")
    def check_heads_divide_d_model(self) -> Self:
        """Ensure n_heads divides d_model evenly for multi-head attention."""
        if self.d_model % self.n_heads != 0:
            msg = (
                f"n_heads ({self.n_heads}) must divide d_model ({self.d_model}) evenly. "
                f"Got d_model % n_heads = {self.d_model % self.n_heads}."
            )
            raise ValueError(msg)
        if self.latent_dim > self.d_model:
            msg = (
                f"latent_dim ({self.latent_dim}) must be <= d_model ({self.d_model})."
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ExperimentConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination path for the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def model_dump_for_mlflow(self) -> dict[str, str | int | float]:
        """Return a flat dictionary suitable for mlflow.log_params().

        All values are converted to primitive types for MLflow compatibility.
        """
        flat: dict[str, str | int | float] = {}
        for k, v in self.model_dump().items():
            if isinstance(v, (int, float)):
                flat[k] = v
            else:
                flat[k] = str(v)
        return flat
