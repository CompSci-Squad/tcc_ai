"""``tcc`` command-line entry-point.

Subcommand groups:
    data        FRED-MD / NBER snapshots, environment log
    configs     Generate stage-1 / stage-2 sweep YAMLs
    train       Single-config or full sweep training + evaluation
    eval        Baselines, ablations, falsification, confound check
    winners     Pick stage-1 / stage-2 winners
    analysis    Export results to LaTeX
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="TCC iTransformer regime-detection toolkit.",
)


# ============================================================================
# data
# ============================================================================
data_app = typer.Typer(no_args_is_help=True, help="Snapshot management.")
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    vintage: str = typer.Option("2026-04", help="FRED-MD vintage YYYY-MM"),
    output_dir: Path = typer.Option(Path("data/snapshots")),
) -> None:
    """Download a FRED-MD vintage CSV + SHA256."""
    _setup_logging()
    from tcc_itransformer.pipelines.data_download import download_fred_md
    download_fred_md(vintage=vintage, output_dir=output_dir)


@data_app.command("pull-nber")
def data_pull_nber(
    output: Path = typer.Option(Path("data/snapshots/nber_usrec.csv")),
) -> None:
    """Download the NBER USREC monthly recession indicator."""
    _setup_logging()
    from tcc_itransformer.pipelines.data_download import download_nber_usrec
    download_nber_usrec(output=output)


@data_app.command("env-log")
def data_env_log(
    output: Path = typer.Option(Path("docs/environment.json")),
) -> None:
    """Snapshot interpreter, GPU, and package versions to a JSON file."""
    _setup_logging()
    from tcc_itransformer.pipelines.env_log import write_environment
    write_environment(output_path=output)


# ============================================================================
# configs
# ============================================================================
configs_app = typer.Typer(no_args_is_help=True, help="Sweep config generators.")
app.add_typer(configs_app, name="configs")


@configs_app.command("gen-stage1")
def configs_gen_stage1() -> None:
    """Generate stage-1 (LR x dropout) sweep YAMLs."""
    _setup_logging()
    from tcc_itransformer.pipelines.sweep_configs import generate_stage1
    generate_stage1()


@configs_app.command("gen-stage2")
def configs_gen_stage2(
    frozen_stage1: Path = typer.Option(
        Path("configs/stage1_winner.yaml"),
        help="Stage-1 winner YAML to inherit hyperparameters from",
    ),
    output_dir: Path = typer.Option(Path("configs/sweep")),
) -> None:
    """Generate the W x d x K stage-2 sweep YAMLs."""
    _setup_logging()
    from tcc_itransformer.pipelines.sweep_configs import (
        generate_stage2, load_stage1_winner,
    )
    frozen = load_stage1_winner(frozen_stage1)
    generate_stage2(frozen=frozen, output_dir=output_dir)


# ============================================================================
# train
# ============================================================================
train_app = typer.Typer(no_args_is_help=True, help="Training pipelines.")
app.add_typer(train_app, name="train")


@train_app.command("single")
def train_single(
    config: Path = typer.Option(..., help="Path to the YAML config"),
) -> None:
    """Train one config end-to-end with full evaluation under a new MLflow run."""
    _setup_logging()
    from tcc_itransformer.config import ExperimentConfig
    from tcc_itransformer.pipelines.single import run_single_with_mlflow
    cfg = ExperimentConfig.from_yaml(config)
    run_single_with_mlflow(cfg)


@train_app.command("sweep")
def train_sweep(
    config_dir: Path = typer.Option(Path("configs/sweep")),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Train one model per (W, d) and evaluate every K post-hoc."""
    _setup_logging()
    from tcc_itransformer.pipelines.sweep import run_sweep
    run_sweep(config_dir=config_dir, dry_run=dry_run)


# ============================================================================
# eval
# ============================================================================
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation pipelines.")
app.add_typer(eval_app, name="eval")


@eval_app.command("baselines")
def eval_baselines(
    config_dir: Path = typer.Option(Path("configs/baselines_op")),
) -> None:
    """Run the 4 baselines per config + locked panel CSV."""
    _setup_logging()
    from tcc_itransformer.pipelines.baselines import run_baselines
    run_baselines(config_dir=config_dir)


@eval_app.command("ablation")
def eval_ablation(
    embeddings_dir: Path = typer.Option(...),
    output_dir: Path = typer.Option(Path("results/clustering_ablation")),
    n_clusters: int = typer.Option(4),
    seed: int = typer.Option(42),
    usrec_csv: Path = typer.Option(Path("data/snapshots/nber_usrec.csv")),
    methods_dr: str = typer.Option(
        "pca,umap,tsne", help="Comma-separated DR methods",
    ),
    methods_cl: str = typer.Option(
        "kmeans,hdbscan", help="Comma-separated clusterers",
    ),
    n_perm: int = typer.Option(1000),
    n_boot: int = typer.Option(1000),
) -> None:
    """{PCA,UMAP,t-SNE} x {KMeans,HDBSCAN} ablation on cached embeddings."""
    _setup_logging()
    from tcc_itransformer.pipelines.clustering_ablation import run_ablation
    run_ablation(
        embeddings_dir=embeddings_dir, output_dir=output_dir,
        n_clusters=n_clusters, seed=seed,
        usrec_csv=usrec_csv if usrec_csv.exists() else None,
        methods_dr=tuple(s.strip() for s in methods_dr.split(",") if s.strip()),
        methods_cl=tuple(s.strip() for s in methods_cl.split(",") if s.strip()),
        n_perm=n_perm, n_boot=n_boot,
    )


@eval_app.command("hdbscan-sweep")
def eval_hdbscan_sweep(
    embeddings_dir: Path = typer.Option(...),
    output_csv: Path = typer.Option(
        Path("results/clustering_ablation/param_sweep.csv"),
    ),
    seed: int = typer.Option(42),
) -> None:
    """A3 grid sweep over UMAP / t-SNE x HDBSCAN params."""
    _setup_logging()
    from tcc_itransformer.pipelines.hdbscan_param_sweep import run_hdbscan_sweep
    run_hdbscan_sweep(
        embeddings_dir=embeddings_dir, output_csv=output_csv, seed=seed,
    )


@eval_app.command("hdphmm")
def eval_hdphmm(
    config: Path = typer.Option(...),
    variant: str = typer.Option(
        "sticky", help="sticky | sdhdp", case_sensitive=False,
    ),
    n_states_max: int = typer.Option(10),
    n_iter: int = typer.Option(100),
    mlflow_experiment: str = typer.Option("hdphmm_baseline"),
) -> None:
    """Sticky / SDHDP-HMM baseline (requires `--extra baselines`)."""
    _setup_logging()
    from tcc_itransformer.pipelines.hdphmm import run_hdphmm
    run_hdphmm(
        config_path=config, variant=variant.lower(),
        n_states_max=n_states_max, n_iter=n_iter,
        mlflow_experiment=mlflow_experiment,
    )


@eval_app.command("falsify")
def eval_falsify(
    config: Path = typer.Option(...),
    usrec_csv: Path = typer.Option(Path("data/snapshots/nber_usrec.csv")),
    output_csv: Path = typer.Option(Path("results/falsification.csv")),
    d_lat: int = typer.Option(7),
    epochs: int = typer.Option(100),
    seed: int = typer.Option(42),
) -> None:
    """B2 falsification: linear AE / MLP AE / SVD encoders matched to d_lat."""
    _setup_logging()
    from tcc_itransformer.pipelines.falsification import run_falsification
    run_falsification(
        config_path=config, usrec_csv=usrec_csv, output_csv=output_csv,
        d_lat=d_lat, epochs=epochs, seed=seed,
    )


@eval_app.command("confound")
def eval_confound(
    embeddings_dir: Path = typer.Option(...),
    usrec_csv: Path = typer.Option(Path("data/snapshots/nber_usrec.csv")),
    panel_parquet: Path = typer.Option(
        Path("data/raw/fred_md_transformed_balanced_2026_04.parquet"),
    ),
    output: Path = typer.Option(Path("results/diagnostics/confound_check.md")),
    n_pcs: int = typer.Option(2),
    seed: int = typer.Option(42),
) -> None:
    """A1 confound check: NBER, pre-2008, |dINDPRO| chi-square + ARI w/o 2020-Q2."""
    _setup_logging()
    from tcc_itransformer.pipelines.confound import run_confound_check
    run_confound_check(
        embeddings_dir=embeddings_dir, usrec_csv=usrec_csv,
        panel_parquet=panel_parquet, output=output,
        n_pcs=n_pcs, seed=seed,
    )


# ============================================================================
# winners
# ============================================================================
winners_app = typer.Typer(no_args_is_help=True, help="Pick sweep winners.")
app.add_typer(winners_app, name="winners")


@winners_app.command("stage1")
def winners_stage1(
    jobs_file: Path = typer.Option(Path(".sm_sweep_jobs.txt")),
    bucket: str = typer.Option("tcc-regime-etl-sagemaker"),
    region: str = typer.Option("us-east-1"),
    summary_csv: Path = typer.Option(Path("results/stage1_summary.csv")),
    winner_yaml: Path = typer.Option(Path("configs/stage1_winner.yaml")),
    cache_dir: Path = typer.Option(Path("results/sm_outputs")),
) -> None:
    """Pick the stage-1 winner from completed SageMaker jobs."""
    _setup_logging()
    from tcc_itransformer.pipelines.winners import pick_stage1_winner
    raise typer.Exit(code=pick_stage1_winner(
        jobs_file=jobs_file, bucket=bucket, region=region,
        summary_csv=summary_csv, winner_yaml=winner_yaml, cache_dir=cache_dir,
    ))


@winners_app.command("stage2")
def winners_stage2(
    summary_csv: Path = typer.Option(Path("results/stage2_summary.csv")),
    winner_yaml: Path = typer.Option(Path("configs/stage2_winner.yaml")),
    tol: float = typer.Option(1e-4),
) -> None:
    """Pick the stage-2 winner using the pre-registered tiebreak."""
    _setup_logging()
    from tcc_itransformer.pipelines.winners import pick_stage2_winner
    raise typer.Exit(code=pick_stage2_winner(
        summary_csv=summary_csv, winner_yaml=winner_yaml, tol=tol,
    ))


# ============================================================================
# analysis
# ============================================================================
analysis_app = typer.Typer(no_args_is_help=True, help="Result export.")
app.add_typer(analysis_app, name="analysis")


@analysis_app.command("export")
def analysis_export(
    output_dir: Path = typer.Option(Path("results/export")),
) -> None:
    """Export MLflow runs to LaTeX tables."""
    _setup_logging()
    from tcc_itransformer.pipelines.export_results import export_results
    export_results(output_dir=output_dir)


if __name__ == "__main__":
    app()
