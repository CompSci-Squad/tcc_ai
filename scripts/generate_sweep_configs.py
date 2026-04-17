"""Generate 36 sweep YAML configs: W∈{6,12,24} × d∈{6,7,8,9} × K∈{3,4,5}.

K is post-hoc (clustering only), so the same trained model serves K=3,4,5.
This script generates one config file per (W, d, K) combination for tracking.
"""

from pathlib import Path

from tcc_itransformer.config import ExperimentConfig

SWEEP_DIR = Path("configs/sweep")
WINDOW_SIZES = [6, 12, 24]
LATENT_DIMS = [6, 7, 8, 9]
N_CLUSTERS = [3, 4, 5]


def generate_sweep_configs() -> int:
    """Generate all sweep configuration YAML files.

    Returns:
        Number of configs generated.
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for w in WINDOW_SIZES:
        for d in LATENT_DIMS:
            for k in N_CLUSTERS:
                config = ExperimentConfig(
                    window_size=w,
                    latent_dim=d,
                    n_clusters=k,
                    experiment_name=f"sweep-W{w}-d{d}-K{k}",
                )
                filename = SWEEP_DIR / f"W{w}_d{d}_K{k}.yaml"
                config.to_yaml(filename)
                count += 1

    return count


if __name__ == "__main__":
    n = generate_sweep_configs()
    print(f"Generated {n} sweep configs in {SWEEP_DIR}/")
