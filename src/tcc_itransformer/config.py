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

    # Pipeline gates — when False, only train AE + extract embeddings + reconstruction MSE.
    # The downstream UMAP/HDBSCAN/NBER/explanations block is then skipped, which keeps
    # SageMaker training jobs cheap (GPU only does AE work). Run the clustering grid
    # afterwards via `tcc eval ablation` against the cached embeddings.
    run_clustering: bool = True

    # Validation snapshots
    nber_usrec_path: str = "data/snapshots/nber_usrec.csv"
    explain_top_k: int = Field(default=5, ge=1, le=50)

    # Data splits — re-locked 2026-04-30 (Option C, B1 in panel-remediation-plan).
    # Previous Option B had VAL=2018-19 (0 NBER recession months) -> NBER F1
    # structurally undefined on VAL and 0 on TEST (TEST=2020-06+ has 0 recessions
    # post-mapping window). New split puts both 2001 and 2008 recessions in VAL
    # so the frozen-NBER mapping has a real signal to fit, and TEST 2010-26
    # carries 2020 COVID + recovery for honest evaluation.
    train_end: str = "1999-12-01"
    val_end: str = "2009-12-01"

    # Paths
    data_path: str = "data/snapshots/fred_md_2026_04.csv"
    mask_path: str | None = None  # ETL v2: path to fred_md_mask_balanced_*.parquet
    results_dir: str = "results"

    # Data contract — selects loader.
    # - "fred_md_csv": legacy McCracken-Ng FRED-MD CSV (tcodes in row 2). Applies
    #   transform_panel + drop_high_nan + forward_fill internally.
    # - "etl_v2_parquet": already-transformed-and-imputed wide parquet from
    #   tcc_etl v2 (s3://tcc-regime-etl-panel-data/fred_md/transformed/...).
    #   Skips transform/dropna/ffill. Requires mask_path. Applies D7 window-end
    #   imputation filter when drop_imputed_windows=True.
    data_format: Literal["fred_md_csv", "etl_v2_parquet"] = "fred_md_csv"
    data_contract: str = "fred_md_csv_v1"  # MLflow tag for lineage
    data_sha256: str | None = None  # Optional content hash for reproducibility

    # D7 imputation policy (pre_analysis_plan.md addendum 2026-04-29).
    # The principal policy is (c)+(a):
    #   - loss_mask_imputed=True (D7.c): train/val use the full window set
    #     and apply masked MSE loss; the AE is graded only on observed cells.
    #   - eval_drop_imputed_target=True (D7.a): the test split additionally
    #     drops windows whose target (last) row has any imputed cell, so
    #     reported test metrics and downstream clustering use only windows
    #     anchored on a fully observed date.
    # Robustness appendix (D7.b): set both to False to relax to the legacy
    # behaviour where imputed cells are treated as observed.
    loss_mask_imputed: bool = True
    eval_drop_imputed_target: bool = True
    # D7.a tolerance: a test window's target row is dropped only when more
    # than (1 - min_observed_fraction) of its cells are imputed. Default 0.95
    # tolerates up to 5% imputation per row, which prevents a single
    # late-publishing series (e.g. CP3M) from invalidating an otherwise
    # well-observed regime label. Set to 1.0 to recover the strict policy
    # ("any imputed cell rejects the window"); set to 0.0 to disable filtering.
    eval_min_observed_fraction: float = 0.95
    # Deprecated shim — kept only for backward compat with older yamls.
    # Ignored when loss_mask_imputed/eval_drop_imputed_target are set explicitly.
    drop_imputed_windows: bool = True

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
