"""B2 falsification: alternative encoders matched to the iTransformer at d_lat.

Trains three cheap encoders on the same windowed FRED-MD panel as the
iTransformer winner, runs the same UMAP+HDBSCAN downstream, computes the
locked 7-metric panel for each.

Decision rule (panel-remediation-plan B2): if all three baselines land within
+/- 0.05 of the iTransformer on the regime-relevant panel metrics, the
iTransformer is not necessary -- thesis should refocus on windowed-PCA + HDBSCAN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, TensorDataset

from tcc_itransformer.data.fred_md import load_fred_md, transform_panel
from tcc_itransformer.data.preprocessing import (
    create_windows,
    drop_high_nan_series,
    fit_scaler,
    forward_fill_nans,
    load_etl_v2_panel,
    scale_splits,
    split_by_date,
)
from tcc_itransformer.evaluation.density_clustering import optimize_hdbscan_dbcv
from tcc_itransformer.evaluation.dim_reduction import UMAPConfig, apply_umap, fit_umap
from tcc_itransformer.evaluation.panel_metrics import PANEL_COLUMNS, compute_panel_metrics
from tcc_itransformer.seed import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    Z_train_in: np.ndarray
    Z_val_in: np.ndarray
    Z_test_in: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def _build_splits(cfg: dict) -> Splits:
    if cfg.get("data_format") == "etl_v2_parquet":
        panel_df, _ = load_etl_v2_panel(cfg["data_path"], cfg.get("mask_path"))
        train_df, val_df, test_df = split_by_date(
            panel_df, cfg["train_end"], cfg["val_end"],
        )
    else:
        data, tcodes = load_fred_md(cfg["data_path"])
        transformed = transform_panel(data, tcodes)
        cleaned, _ = drop_high_nan_series(transformed)
        filled = forward_fill_nans(cleaned)
        train_df, val_df, test_df = split_by_date(
            filled, cfg["train_end"], cfg["val_end"],
        )

    scaler = fit_scaler(train_df)
    train_s, val_s, test_s = scale_splits(train_df, val_df, test_df, scaler)

    W = int(cfg["window_size"])
    train_w = create_windows(train_s, W)
    val_w = create_windows(val_s, W)
    test_w = create_windows(test_s, W)

    def flat(a: np.ndarray) -> np.ndarray:
        return a.reshape(a.shape[0], -1).astype(np.float32)

    return Splits(
        Z_train_in=flat(train_w),
        Z_val_in=flat(val_w),
        Z_test_in=flat(test_w),
        train_dates=pd.DatetimeIndex(train_df.index[W - 1 :]),
        val_dates=pd.DatetimeIndex(val_df.index[W - 1 :]),
        test_dates=pd.DatetimeIndex(test_df.index[W - 1 :]),
    )


class LinearAE(nn.Module):
    def __init__(self, in_dim: int, d: int) -> None:
        super().__init__()
        self.enc = nn.Linear(in_dim, d)
        self.dec = nn.Linear(d, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.dec(self.enc(x))


class MlpAE(nn.Module):
    def __init__(self, in_dim: int, d: int, hidden: int = 128) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, d),
        )
        self.dec = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.dec(self.enc(x))


def _train_ae(
    model: nn.Module, X: np.ndarray, *,
    epochs: int = 100, batch_size: int = 128, lr: float = 1e-3, seed: int = 42,
) -> nn.Module:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        if (epoch + 1) % 20 == 0:
            logger.info("  epoch %d  mse=%.6f", epoch + 1, total / n)
    model.eval()
    return model


def _encode(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        z = model.enc(torch.from_numpy(X).to(device)).cpu().numpy()
    return z.astype(np.float32)


def _downstream_panel(
    Z_tr: np.ndarray, Z_va: np.ndarray, Z_te: np.ndarray,
    val_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
    *, usrec_csv: Path, seed: int = 42,
) -> dict:
    reducer = fit_umap(Z_tr, UMAPConfig(n_components=2, random_state=seed))
    Y_va = apply_umap(Z_va, reducer)
    Y_te = apply_umap(Z_te, reducer)
    Y_tr = apply_umap(Z_tr, reducer)
    best, _ = optimize_hdbscan_dbcv(Y_tr)
    try:
        import hdbscan as _hdbscan
        val_labels, _ = _hdbscan.approximate_predict(best.clusterer, Y_va)
        test_labels, _ = _hdbscan.approximate_predict(best.clusterer, Y_te)
    except Exception:  # pragma: no cover
        from tcc_itransformer.evaluation.density_clustering import fit_hdbscan
        val_labels = fit_hdbscan(
            Y_va, min_cluster_size=best.min_cluster_size,
            min_samples=best.min_samples,
        ).labels
        test_labels = fit_hdbscan(
            Y_te, min_cluster_size=best.min_cluster_size,
            min_samples=best.min_samples,
        ).labels
    panel = compute_panel_metrics(
        val_labels=np.asarray(val_labels), val_dates=val_dates,
        test_labels=np.asarray(test_labels), test_dates=test_dates,
        Y_test=Y_te, test_signal=Y_te,
        usrec_csv=usrec_csv, is_density_clusterer=True,
    )
    panel["_train_dbcv"] = float(best.dbcv)
    panel["_train_n_clusters"] = int(best.n_clusters)
    return panel


def run_falsification(
    config_path: Path,
    *,
    usrec_csv: Path = Path("data/snapshots/nber_usrec.csv"),
    output_csv: Path = Path("results/falsification.csv"),
    d_lat: int = 7, epochs: int = 100, seed: int = 42,
) -> pd.DataFrame:
    set_global_seed(seed)
    cfg = yaml.safe_load(config_path.read_text())
    sp = _build_splits(cfg)
    in_dim = sp.Z_train_in.shape[1]
    logger.info(
        "Splits: train=%s val=%s test=%s in_dim=%d",
        sp.Z_train_in.shape, sp.Z_val_in.shape, sp.Z_test_in.shape, in_dim,
    )

    rows: list[dict] = []
    encoders = [
        ("linear_ae", lambda: LinearAE(in_dim, d_lat)),
        ("mlp_ae", lambda: MlpAE(in_dim, d_lat)),
        ("svd", None),
    ]
    for name, factory in encoders:
        logger.info("=== encoder: %s ===", name)
        if name == "svd":
            svd = TruncatedSVD(n_components=d_lat, random_state=seed)
            svd.fit(sp.Z_train_in)
            Z_tr = svd.transform(sp.Z_train_in).astype(np.float32)
            Z_va = svd.transform(sp.Z_val_in).astype(np.float32)
            Z_te = svd.transform(sp.Z_test_in).astype(np.float32)
        else:
            model = _train_ae(factory(), sp.Z_train_in, epochs=epochs, seed=seed)
            Z_tr = _encode(model, sp.Z_train_in)
            Z_va = _encode(model, sp.Z_val_in)
            Z_te = _encode(model, sp.Z_test_in)

        panel = _downstream_panel(
            Z_tr, Z_va, Z_te, sp.val_dates, sp.test_dates,
            usrec_csv=usrec_csv, seed=seed,
        )
        row = {
            "encoder": name,
            **{k: panel.get(k, float("nan")) for k in PANEL_COLUMNS},
            "train_dbcv": panel["_train_dbcv"],
            "train_n_clusters": panel["_train_n_clusters"],
        }
        rows.append(row)
        logger.info("%s -> %s", name, {k: row[k] for k in PANEL_COLUMNS})

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %s", output_csv)
    print("\n=== Falsification panel ===")
    print(df.to_string(index=False))
    return df
