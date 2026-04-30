"""Generate stage-1 LR x dropout grid (12 cells) for two-stage HPO.

Stage 1 (this script): sweep ``learning_rate`` x ``dropout`` at the **primary
configuration** (W=12, d_lat=8) to pick the best (lr, dropout) by VAL
reconstruction MSE. Local CPU run.

Stage 2 (existing ``generate_sweep_configs.py``): the W x d x K
architectural sweep, with LR/dropout frozen to the stage-1 winner via
``--frozen-stage1 configs/stage1_winner.yaml`` (see that script).

Per ``docs/pre_analysis_plan.md`` Addendum 2026-04-29 \u00a74.
"""

from pathlib import Path

from tcc_itransformer.config import ExperimentConfig

SWEEP_DIR = Path("configs/sweep_stage1")
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
DROPOUTS = [0.0, 0.1, 0.2, 0.3]


def generate_stage1_configs() -> int:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
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
            (SWEEP_DIR / f"{tag}.yaml").write_text("")  # ensure file present
            config.to_yaml(SWEEP_DIR / f"{tag}.yaml")
            count += 1
    return count


if __name__ == "__main__":
    n = generate_stage1_configs()
    print(f"Generated {n} stage-1 configs in {SWEEP_DIR}/")
