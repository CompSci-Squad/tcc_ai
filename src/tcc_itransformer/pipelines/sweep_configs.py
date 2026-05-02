"""Generate sweep YAML configs for stage-1 (LR x dropout) and stage-2 (W x d x K).

Stage 1 sweeps ``learning_rate`` x ``dropout`` at the primary (W=12, d_lat=8)
configuration to pick the best (lr, dropout) by VAL reconstruction MSE.
Stage 2 (the architectural W x d x K grid) freezes those values via
``--frozen-stage1`` so the comparison is apples-to-apples.

See ``docs/pre_analysis_plan.md`` Addendum 2026-04-29 §4.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tcc_itransformer.config import ExperimentConfig

STAGE2_DIR = Path("configs/sweep")
STAGE1_DIR = Path("configs/sweep_stage1")

WINDOW_SIZES = [6, 12, 24]
LATENT_DIMS = [6, 7, 8, 9]
N_CLUSTERS = [3, 4, 5]

LEARNING_RATES = [1e-4, 3e-4, 1e-3]
DROPOUTS = [0.0, 0.1, 0.2, 0.3]


def load_stage1_winner(path: Path) -> dict[str, float]:
    """Read frozen learning_rate + dropout from a stage-1 winner YAML."""
    cfg = yaml.safe_load(path.read_text())
    return {
        "learning_rate": float(cfg["learning_rate"]),
        "dropout": float(cfg["dropout"]),
    }


def generate_stage2(
    frozen: dict[str, float] | None = None,
    output_dir: Path = STAGE2_DIR,
) -> int:
    """Write 36 stage-2 YAMLs (W x d x K). Returns count written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = frozen or {}
    count = 0
    for w in WINDOW_SIZES:
        for d in LATENT_DIMS:
            for k in N_CLUSTERS:
                config = ExperimentConfig(
                    window_size=w,
                    latent_dim=d,
                    n_clusters=k,
                    experiment_name=f"sweep-W{w}-d{d}-K{k}",
                    run_clustering=False,
                    **overrides,
                )
                config.to_yaml(output_dir / f"W{w}_d{d}_K{k}.yaml")
                count += 1
    return count


def generate_stage1(output_dir: Path = STAGE1_DIR) -> int:
    """Write 12 stage-1 YAMLs (LR x dropout @ W=12, d=8). Returns count written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lr in LEARNING_RATES:
        for dr in DROPOUTS:
            tag = f"lr{lr:.0e}_drop{int(dr * 100):02d}"
            config = ExperimentConfig(
                window_size=12,
                latent_dim=8,
                n_clusters=4,
                learning_rate=lr,
                dropout=dr,
                experiment_name=f"stage1-{tag}",
            )
            config.to_yaml(output_dir / f"{tag}.yaml")
            count += 1
    return count
