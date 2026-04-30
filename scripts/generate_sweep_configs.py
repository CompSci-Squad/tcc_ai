"""Generate 36 sweep YAML configs: W∈{6,12,24} × d∈{6,7,8,9} × K∈{3,4,5}.

K is post-hoc (clustering only), so the same trained model serves K=3,4,5.
This script generates one config file per (W, d, K) combination for tracking.

Optionally inherit ``learning_rate`` and ``dropout`` from a stage-1 winner
YAML (two-stage HPO; see ``docs/pre_analysis_plan.md`` Addendum 2026-04-29).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tcc_itransformer.config import ExperimentConfig

SWEEP_DIR = Path("configs/sweep")
WINDOW_SIZES = [6, 12, 24]
LATENT_DIMS = [6, 7, 8, 9]
N_CLUSTERS = [3, 4, 5]


def _load_stage1_winner(path: Path) -> dict[str, float]:
    """Read frozen learning_rate + dropout from a stage-1 winner YAML."""
    cfg = yaml.safe_load(path.read_text())
    return {
        "learning_rate": float(cfg["learning_rate"]),
        "dropout": float(cfg["dropout"]),
    }


def generate_sweep_configs(frozen: dict[str, float] | None = None) -> int:
    """Generate all sweep configuration YAML files.

    Args:
        frozen: Optional dict with ``learning_rate`` and ``dropout`` to
            inherit from stage-1 winner.

    Returns:
        Number of configs generated.
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
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
                    run_clustering=False,  # cloud sweep is AE-only; clustering ablation runs locally on Z_*.parquet
                    **overrides,
                )
                filename = SWEEP_DIR / f"W{w}_d{d}_K{k}.yaml"
                config.to_yaml(filename)
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--frozen-stage1",
        type=Path,
        default=None,
        help="Stage-1 winner YAML to inherit learning_rate + dropout from.",
    )
    args = parser.parse_args()
    frozen = _load_stage1_winner(args.frozen_stage1) if args.frozen_stage1 else None
    n = generate_sweep_configs(frozen=frozen)
    note = f" (frozen from {args.frozen_stage1})" if frozen else ""
    print(f"Generated {n} sweep configs in {SWEEP_DIR}/{note}")


if __name__ == "__main__":
    main()
