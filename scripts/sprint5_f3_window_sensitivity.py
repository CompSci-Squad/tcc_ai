#!/usr/bin/env python3
"""Sprint 5 — F3: Window Sensitivity Analysis (W ∈ {3, 6, 9, 12, 18, 24}).

Fragility addressed:
    The canonical model uses W=6 (6-month lookback window). This choice was
    motivated by the 1-quarter macro release cycle, but was not validated
    by comparison to neighbouring window sizes. If F1=0.571 is achieved only
    at W=6 and drops sharply elsewhere, the result is a local artefact of
    the window choice.

Method:
    For each W ∈ {3, 6, 9, 12, 18, 24}:
    1. Build rolling windows from FRED-MD transformed panel.
    2. Train iTransformerAE (d_model=64, d_latent=7, n_heads=4, n_layers=2,
       max_epochs=150, patience=15) — same hyperparams as canonical.
    3. Extract Z_train, Z_val, Z_test embeddings.
    4. Run PCA(90%) + KMeans(K=4) — same as canonical cell.
    5. Assign recession cluster on VAL (NBER overlap), evaluate on TEST.
    6. Report NBER F1, MCC, Silhouette, n_effective_test.

NOTE: W=6 result is re-run from scratch for strict comparability; the
canonical model's F1=0.571 is expected to replicate within ±0.05.

Outputs: results/sprint5/
    window_sensitivity.csv        — per-W metrics
    SUMMARY_f3_window_sens.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, silhouette_score
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tcc_itransformer.model.autoencoder import iTransformerAE
from tcc_itransformer.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results/sprint5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZES = [3, 6, 9, 12, 18, 24]
SEED = 42

# Architecture (same as canonical W6_d7_K4_b1)
D_MODEL = 64
D_LATENT = 7
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

# Training
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

N_CLUSTERS = 4
TOL_MONTHS = 6  # F1 tolerance for NBER comparison

# Split dates (canonical B1 split)
TRAIN_END = "1999-12-01"
VAL_END = "2009-12-01"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_fred() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (T, N) float32 array and DatetimeIndex."""
    df = pd.read_parquet(ROOT / "data/raw/fred_md_transformed_2026_04.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(df["date"])
    # Drop non-numeric columns
    series_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[series_cols].to_numpy(dtype=np.float32)
    # Forward-fill then zero-fill remaining NaNs
    for c in range(data.shape[1]):
        mask = np.isnan(data[:, c])
        if mask.any():
            df2 = pd.Series(data[:, c]).ffill().fillna(0.0)
            data[:, c] = df2.to_numpy(dtype=np.float32)
    return data, dates


def load_usrec() -> set[str]:
    """Return set of NBER recession month strings 'YYYY-MM'."""
    for p in [
        ROOT / "data/snapshots/nber_usrec.csv",
        ROOT / "data/snapshots/usrec.csv",
        ROOT / "data/raw/nber_usrec.csv",
        ROOT / "data/raw/usrec.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            date_col = "observation_date" if "observation_date" in df.columns else "date"
            df["_date"] = pd.to_datetime(df[date_col])
            return set(df[df["USREC"] == 1]["_date"].dt.strftime("%Y-%m"))
    raise FileNotFoundError("USREC CSV not found")


def make_windows(data: np.ndarray, w: int) -> np.ndarray:
    """Create (n_windows, w, n_features) rolling windows with stride=1."""
    n, n_feat = data.shape
    windows = np.lib.stride_tricks.sliding_window_view(data, (w, n_feat))
    return windows[:, 0, :, :]  # shape: (n-w+1, w, n_feat)


def split_windows(
    windows: np.ndarray,
    window_end_dates: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split windows by the end date of each window (index = last row of window)."""
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)
    tr = window_end_dates <= t_end
    va = (window_end_dates > t_end) & (window_end_dates <= v_end)
    te = window_end_dates > v_end
    return (
        windows[tr], windows[va], windows[te],
        window_end_dates[tr], window_end_dates[va], window_end_dates[te],
    )


def standardize(
    Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize by train mean/std, per feature (last dim)."""
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    return (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

# ── Training ─────────────────────────────────────────────────────────────────

def make_dataloader(windows: np.ndarray, shuffle: bool) -> DataLoader:
    t = torch.from_numpy(windows).float()  # (N, W, F)
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def train_ae(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    window_size: int,
    n_series: int,
    device: torch.device,
) -> iTransformerAE:
    """Train and return a converged iTransformerAE."""
    model = iTransformerAE(
        n_series=n_series,
        window_size=window_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        latent_dim=D_LATENT,
        dropout=DROPOUT,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=MAX_EPOCHS)

    tr_loader = make_dataloader(Xtr, shuffle=True)
    va_loader = make_dataloader(Xva, shuffle=False)

    best_val = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for (batch,) in tr_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for (batch,) in va_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += torch.nn.functional.mse_loss(recon, batch).item() * len(batch)
                n_val += len(batch)
        val_loss /= max(n_val, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("  Early stop at epoch %d (val_loss=%.4f)", epoch + 1, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def extract_embeddings(
    model: iTransformerAE, windows: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Return (N, latent_dim) embedding matrix."""
    model.eval()
    loader = make_dataloader(windows, shuffle=False)
    zs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, z = model(batch)
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)

# ── Clustering + evaluation ───────────────────────────────────────────────────

def nber_f1_tol(
    dates: pd.DatetimeIndex,
    labels: np.ndarray,
    rec_cluster: int,
    usrec_set: set[str],
    tol: int = TOL_MONTHS,
) -> float:
    """Compute F1 with ±tol month tolerance window."""
    # Build tolerance-expanded NBER mask
    date_strs = pd.DatetimeIndex(dates).strftime("%Y-%m")
    tol_rec = set()
    for ds in usrec_set:
        ts = pd.Timestamp(ds + "-01")
        for d in range(-tol, tol + 1):
            tol_rec.add((ts + pd.DateOffset(months=d)).strftime("%Y-%m"))

    pred = (labels == rec_cluster).astype(int)
    true = np.array([1 if ds in tol_rec else 0 for ds in date_strs])

    tp = int((pred & true).sum())
    fp = int((pred & ~true.astype(bool)).sum())
    fn = int((~pred.astype(bool) & true).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def run_one_window(
    w: int,
    data: np.ndarray,
    dates: pd.DatetimeIndex,
    usrec_set: set[str],
    device: torch.device,
) -> dict:
    set_global_seed(SEED)
    logger.info("─── W=%d ───", w)

    # Windows: end_date = date[i + w - 1] for window starting at i
    windows = make_windows(data, w)
    # end_date index in original data array
    end_dates = dates[w - 1:]  # shape = n_windows
    assert len(windows) == len(end_dates)

    Xtr, Xva, Xte, dtr, dva, dte = split_windows(windows, end_dates, TRAIN_END, VAL_END)
    logger.info("  splits: train=%d, val=%d, test=%d", len(Xtr), len(Xva), len(Xte))

    if len(Xtr) < 20 or len(Xva) < 5 or len(Xte) < 5:
        logger.warning("  W=%d too small for reliable split — skipping", w)
        return {}

    n_series = data.shape[1]
    Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)

    # Train
    model = train_ae(Xtr_s, Xva_s, window_size=w, n_series=n_series, device=device)

    # Embeddings
    Z_tr = extract_embeddings(model, Xtr_s, device)
    Z_va = extract_embeddings(model, Xva_s, device)
    Z_te = extract_embeddings(model, Xte_s, device)

    # PCA (90% variance) — fit on train, apply to all
    pca = PCA(n_components=min(5, Z_tr.shape[1]))
    pca.fit(Z_tr)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum_var, 0.90)) + 1
    n_comp = max(2, min(n_comp, 5))
    pca_k = PCA(n_components=n_comp)
    pca_k.fit(Z_tr)
    Y_tr = pca_k.transform(Z_tr)
    Y_va = pca_k.transform(Z_va)
    Y_te = pca_k.transform(Z_te)

    # KMeans(K=4) on train
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    km.fit(Y_tr)
    lbl_va = km.predict(Y_va)
    lbl_te = km.predict(Y_te)

    # Recession cluster: best NBER overlap on VAL
    va_strs = dva.strftime("%Y-%m")
    usrec_va = np.array([1 if d in usrec_set else 0 for d in va_strs])
    best_cluster = max(range(N_CLUSTERS), key=lambda k: (lbl_va == k).astype(int) @ usrec_va)
    nber_overlap_va = int(((lbl_va == best_cluster) & (usrec_va == 1)).sum())

    # Test metrics
    f1_tol = nber_f1_tol(dte, lbl_te, best_cluster, usrec_set)
    sil = silhouette_score(Y_te, lbl_te) if len(set(lbl_te)) > 1 else float("nan")
    mcc_vals = np.array([1 if d in usrec_set else 0
                         for d in dte.strftime("%Y-%m")])
    mcc_pred = (lbl_te == best_cluster).astype(int)
    mcc = matthews_corrcoef(mcc_vals, mcc_pred) if mcc_vals.sum() > 0 and mcc_pred.sum() > 0 else 0.0

    row = {
        "window_size": w,
        "n_train": len(Xtr),
        "n_val": len(Xva),
        "n_test": len(Xte),
        "n_pca_components": n_comp,
        "pca_var_explained": round(float(pca_k.explained_variance_ratio_.sum()), 4),
        "recession_cluster": int(best_cluster),
        "nber_overlap_val": nber_overlap_va,
        "nber_f1_tol": round(f1_tol, 4),
        "mcc": round(float(mcc), 4),
        "test_silhouette": round(float(sil), 4),
    }
    logger.info("  W=%d → F1_tol=%.4f, MCC=%.4f, Sil=%.4f", w, f1_tol, mcc, sil)
    return row


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data, dates = load_fred()
    usrec_set = load_usrec()
    logger.info("FRED-MD: %d months × %d series", *data.shape)
    logger.info("NBER recession months: %d", len(usrec_set))

    rows = []
    for w in WINDOW_SIZES:
        row = run_one_window(w, data, dates, usrec_set, device)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "window_sensitivity.csv", index=False)
    logger.info("\n%s", df.to_string(index=False))

    # Build summary
    best_row = df.loc[df["nber_f1_tol"].idxmax()]
    canonical_row = df[df["window_size"] == 6]
    canonical_f1 = float(canonical_row["nber_f1_tol"].iloc[0]) if len(canonical_row) > 0 else None

    summary = {
        "method": "iTransformerAE retrained for each W, pca_kmeans clustering",
        "window_sizes_tested": WINDOW_SIZES,
        "seed": SEED,
        "d_model": D_MODEL,
        "d_latent": D_LATENT,
        "n_clusters": N_CLUSTERS,
        "max_epochs": MAX_EPOCHS,
        "results": df[["window_size", "nber_f1_tol", "mcc", "test_silhouette"]].to_dict("records"),
        "best_window": int(best_row["window_size"]),
        "best_f1_tol": round(float(best_row["nber_f1_tol"]), 4),
        "w6_f1_tol_replicated": canonical_f1,
        "f1_range": round(float(df["nber_f1_tol"].max() - df["nber_f1_tol"].min()), 4),
        "f1_std": round(float(df["nber_f1_tol"].std()), 4),
        "conclusion": (
            f"W=6 achieves F1_tol={canonical_f1:.4f} (canonical). "
            f"Best W={int(best_row['window_size'])} achieves F1={float(best_row['nber_f1_tol']):.4f}. "
            f"F1 range across windows: {float(df['nber_f1_tol'].max() - df['nber_f1_tol'].min()):.4f}. "
            "If the range is < 0.15, the W=6 choice is stable and not cherry-picked."
        ),
    }

    with open(OUT_DIR / "SUMMARY_f3_window_sens.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("F3 complete. W sensitivity results saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
