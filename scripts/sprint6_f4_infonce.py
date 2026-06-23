#!/usr/bin/env python3
"""Sprint 6 — F4: InfoNCE Contrastive Training.

Fragility addressed:
    The iTransformer uses pure MSE reconstruction loss. Contrastive objectives
    (e.g. InfoNCE, SimCLR) are known to produce better-separated representation
    spaces. If adding a contrastive term significantly improves recession cluster
    separability, the MSE-only baseline may be under-performing due to its
    loss function, not its architecture.

Method:
    Train two variants of iTransformerAE on FRED-MD (W=6, canonical config):
    A) MSE-only (reconstruction baseline — replicates canonical training)
    B) MSE + α*InfoNCE, where α=0.1

    InfoNCE positive pairs: window pairs (i, j) where |end_date_i - end_date_j|
    ≤ 3 months (temporally adjacent — likely same macro regime).
    InfoNCE negative pairs: all other windows in the same batch (in-batch negatives).

    InfoNCE loss (NT-Xent formulation):
        L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
    where τ=0.5 (temperature) and sim = cosine similarity.

    Evaluation: same as canonical — PCA(90%) + KMeans(K=4) + VAL recession
    cluster assignment + TEST NBER F1/MCC/Silhouette.

Outputs: results/sprint6/
    infonce_vs_mse.csv            — comparison table (4 metrics × 2 variants)
    SUMMARY_f4_infonce.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, silhouette_score
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
OUT_DIR = ROOT / "results/sprint6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 6   # canonical W
SEED = 42

D_MODEL = 64
D_LATENT = 7
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

N_CLUSTERS = 4
TOL_MONTHS = 6

# InfoNCE
INFONCE_ALPHA = 0.1       # weight of contrastive loss
INFONCE_TAU = 0.5         # temperature
POSITIVE_LAG_MONTHS = 3   # |t_i - t_j| ≤ this → positive pair

TRAIN_END = "1999-12-01"
VAL_END = "2009-12-01"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_fred() -> tuple[np.ndarray, pd.DatetimeIndex]:
    df = pd.read_parquet(ROOT / "data/raw/fred_md_transformed_2026_04.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(df["date"])
    series_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[series_cols].to_numpy(dtype=np.float32)
    for c in range(data.shape[1]):
        col = pd.Series(data[:, c]).ffill().fillna(0.0)
        data[:, c] = col.to_numpy(dtype=np.float32)
    return data, dates


def load_usrec() -> set[str]:
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


def make_windows(data: np.ndarray, w: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (windows, end_dates) where end_dates[i] = date of last row in window i."""
    windows = np.lib.stride_tricks.sliding_window_view(data, (w, data.shape[1]))[:, 0]
    return windows, None  # end_dates computed from caller


def split_data(
    data: np.ndarray, dates: pd.DatetimeIndex, w: int,
) -> tuple:
    """Create windows and split by date."""
    n = len(data)
    windows = []
    end_dates = []
    for i in range(n - w + 1):
        windows.append(data[i : i + w])
        end_dates.append(dates[i + w - 1])
    windows = np.array(windows, dtype=np.float32)
    end_dates = pd.DatetimeIndex(end_dates)

    t_end = pd.Timestamp(TRAIN_END)
    v_end = pd.Timestamp(VAL_END)
    tr = end_dates <= t_end
    va = (end_dates > t_end) & (end_dates <= v_end)
    te = end_dates > v_end

    return (
        windows[tr], windows[va], windows[te],
        end_dates[tr], end_dates[va], end_dates[te],
    )


def standardize(Xtr, Xva, Xte):
    m = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    s = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    return (Xtr - m) / s, (Xva - m) / s, (Xte - m) / s


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def build_positive_mask(dates: pd.DatetimeIndex, lag: int) -> torch.Tensor:
    """Binary (N, N) mask: 1 if |i-j| ≤ lag months, 0 otherwise, diagonal=0."""
    n = len(dates)
    months = np.array([d.year * 12 + d.month for d in dates])
    diff = np.abs(months[:, None] - months[None, :])  # (N, N)
    mask = (diff > 0) & (diff <= lag)
    return torch.from_numpy(mask.astype(np.float32))


def nt_xent_loss(z: torch.Tensor, pos_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """NT-Xent loss for a batch. z: (N, D). pos_mask: (N, N) float."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / tau  # (N, N)
    # Mask out self-comparisons
    n = z.size(0)
    eye = torch.eye(n, device=z.device)
    sim = sim - eye * 1e9  # large negative for diagonal
    # For each anchor i, positives are pos_mask[i], negatives are everything else
    # Loss = -log(sum_pos exp(sim_pos) / sum_all exp(sim))
    log_denom = torch.logsumexp(sim, dim=1)  # (N,)
    # Numerator: log-sum of positive similarities
    pos_on_device = pos_mask.to(z.device)
    # If no positives for an anchor, skip that anchor
    has_pos = pos_on_device.sum(dim=1) > 0
    if not has_pos.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # Numerator per anchor: logsumexp over positive pairs
    sim_pos = sim + (1.0 - pos_on_device) * (-1e9)
    log_numer = torch.logsumexp(sim_pos, dim=1)  # (N,)
    losses = -(log_numer - log_denom)
    return losses[has_pos].mean()


# ── Dataset that returns window index ─────────────────────────────────────────

class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx], idx


# ── Training ─────────────────────────────────────────────────────────────────

def train_ae(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    dtr: pd.DatetimeIndex,
    window_size: int,
    n_series: int,
    device: torch.device,
    use_infonce: bool,
    alpha: float = INFONCE_ALPHA,
    tau: float = INFONCE_TAU,
) -> iTransformerAE:
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

    # Pre-compute positive mask for the whole training set
    pos_mask_full = build_positive_mask(dtr, POSITIVE_LAG_MONTHS) if use_infonce else None

    tr_ds = IndexedTensorDataset(Xtr)
    va_ds = IndexedTensorDataset(Xva)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for (batch_x, batch_idx) in tr_loader:
            batch_x = batch_x.to(device)
            recon, z = model(batch_x)
            mse = F.mse_loss(recon, batch_x)
            if use_infonce:
                # Slice positive mask for this batch
                bidx = batch_idx.numpy()
                bpos_mask = pos_mask_full[np.ix_(bidx, bidx)]
                contrastive = nt_xent_loss(z, bpos_mask, tau)
                loss = mse + alpha * contrastive
            else:
                loss = mse
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for (batch_x, _) in va_loader:
                batch_x = batch_x.to(device)
                recon, _ = model(batch_x)
                val_loss += F.mse_loss(recon, batch_x).item() * len(batch_x)
                n_val += len(batch_x)
        val_loss /= max(n_val, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("  Early stop @ epoch %d (val_loss=%.5f)", epoch + 1, best_val)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def extract_embeddings(model: iTransformerAE, windows: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    ds = IndexedTensorDataset(windows)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    zs = []
    with torch.no_grad():
        for (batch, _) in loader:
            _, z = model(batch.to(device))
            zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def evaluate(
    Z_tr, Z_va, Z_te,
    dva: pd.DatetimeIndex,
    dte: pd.DatetimeIndex,
    usrec_set: set[str],
) -> dict:
    # PCA
    pca = PCA(n_components=min(5, Z_tr.shape[1]))
    pca.fit(Z_tr)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = max(2, int(np.searchsorted(cum_var, 0.90)) + 1)
    n_comp = min(n_comp, 5)
    pca_k = PCA(n_components=n_comp).fit(Z_tr)
    Y_tr, Y_va, Y_te = pca_k.transform(Z_tr), pca_k.transform(Z_va), pca_k.transform(Z_te)

    # KMeans
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit(Y_tr)
    lbl_va = km.predict(Y_va)
    lbl_te = km.predict(Y_te)

    # Recession cluster from VAL
    va_strs = dva.strftime("%Y-%m")
    usrec_va = np.array([1 if d in usrec_set else 0 for d in va_strs])
    best_k = max(range(N_CLUSTERS), key=lambda k: (lbl_va == k).astype(int) @ usrec_va)

    # NBER F1 (tol) on test
    te_strs = dte.strftime("%Y-%m")
    tol_rec = set()
    for ds in usrec_set:
        ts = pd.Timestamp(ds + "-01")
        for d in range(-TOL_MONTHS, TOL_MONTHS + 1):
            tol_rec.add((ts + pd.DateOffset(months=d)).strftime("%Y-%m"))
    pred = (lbl_te == best_k).astype(int)
    true = np.array([1 if d in tol_rec else 0 for d in te_strs])
    tp = int((pred & true).sum())
    fp = int((pred & ~true.astype(bool)).sum())
    fn = int((~pred.astype(bool) & true).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # MCC
    true_raw = np.array([1 if d in usrec_set else 0 for d in te_strs])
    mcc = float(matthews_corrcoef(true_raw, pred)) if true_raw.sum() > 0 and pred.sum() > 0 else 0.0

    # Silhouette
    sil = silhouette_score(Y_te, lbl_te) if len(set(lbl_te)) > 1 else float("nan")

    # Inter-cluster distance ratio: mean centroid dist / mean intra-cluster dist
    centroids = np.array([Y_te[lbl_te == k].mean(0) for k in range(N_CLUSTERS)])
    intra = np.mean([np.mean(np.linalg.norm(Y_te[lbl_te == k] - centroids[k], axis=1))
                     for k in range(N_CLUSTERS) if (lbl_te == k).sum() > 0])
    inter = np.mean([np.linalg.norm(centroids[i] - centroids[j])
                     for i in range(N_CLUSTERS) for j in range(i + 1, N_CLUSTERS)])
    cluster_sep = float(inter / (intra + 1e-8))

    return {
        "recession_cluster": int(best_k),
        "nber_f1_tol": round(f1, 4),
        "mcc": round(mcc, 4),
        "test_silhouette": round(float(sil), 4),
        "cluster_separation": round(cluster_sep, 4),
        "n_pca_components": n_comp,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data, dates = load_fred()
    usrec_set = load_usrec()
    n_series = data.shape[1]

    Xtr, Xva, Xte, dtr, dva, dte = split_data(data, dates, WINDOW_SIZE)
    Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)
    logger.info("Splits: train=%d, val=%d, test=%d", len(Xtr), len(Xva), len(Xte))
    logger.info("Positive pairs (lag≤%d months): %d/%d pairs",
                POSITIVE_LAG_MONTHS,
                int(build_positive_mask(dtr, POSITIVE_LAG_MONTHS).sum().item()),
                len(dtr) * (len(dtr) - 1))

    rows = []
    for use_infonce in [False, True]:
        variant = "MSE + InfoNCE" if use_infonce else "MSE only"
        logger.info("Training: %s", variant)
        set_global_seed(SEED)
        model = train_ae(
            Xtr_s, Xva_s, dtr, WINDOW_SIZE, n_series, device, use_infonce=use_infonce
        )
        Z_tr = extract_embeddings(model, Xtr_s, device)
        Z_va = extract_embeddings(model, Xva_s, device)
        Z_te = extract_embeddings(model, Xte_s, device)
        metrics = evaluate(Z_tr, Z_va, Z_te, dva, dte, usrec_set)
        rows.append({"variant": variant, "infonce_alpha": INFONCE_ALPHA if use_infonce else 0.0, **metrics})
        logger.info("  %s → F1=%.4f, MCC=%.4f, Sil=%.4f, Sep=%.4f",
                    variant, metrics["nber_f1_tol"], metrics["mcc"],
                    metrics["test_silhouette"], metrics["cluster_separation"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "infonce_vs_mse.csv", index=False)
    logger.info("\n%s", df.to_string(index=False))

    mse_row = df[df["variant"] == "MSE only"].iloc[0]
    inc_row = df[df["variant"] == "MSE + InfoNCE"].iloc[0]
    delta_f1 = float(inc_row["nber_f1_tol"]) - float(mse_row["nber_f1_tol"])
    delta_sil = float(inc_row["test_silhouette"]) - float(mse_row["test_silhouette"])
    delta_sep = float(inc_row["cluster_separation"]) - float(mse_row["cluster_separation"])

    summary = {
        "method": "iTransformerAE, MSE vs MSE+InfoNCE (α=0.1, τ=0.5, lag≤3mo)",
        "window_size": WINDOW_SIZE,
        "d_latent": D_LATENT,
        "infonce_alpha": INFONCE_ALPHA,
        "infonce_tau": INFONCE_TAU,
        "positive_lag_months": POSITIVE_LAG_MONTHS,
        "results": {
            "mse_only": {k: v for k, v in mse_row.items() if k != "variant"},
            "mse_infonce": {k: v for k, v in inc_row.items() if k != "variant"},
        },
        "deltas": {
            "delta_nber_f1_tol": round(delta_f1, 4),
            "delta_silhouette": round(delta_sil, 4),
            "delta_cluster_separation": round(delta_sep, 4),
        },
        "conclusion": (
            f"InfoNCE (α={INFONCE_ALPHA}) {'IMPROVES' if delta_f1 > 0.02 else 'DOES NOT IMPROVE'} "
            f"NBER F1 (ΔNBER_F1={delta_f1:+.4f}). "
            f"Silhouette change: {delta_sil:+.4f}. "
            f"Cluster separation change: {delta_sep:+.4f}. "
            "If delta_f1 > 0.05, contrastive training helps recession detection."
        ),
    }

    with open(OUT_DIR / "SUMMARY_f4_infonce.json", "w") as f:
        # Convert numpy scalars to Python native types for JSON serialisation
        def _native(obj: object) -> object:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        def _convert(d: dict) -> dict:
            return {k: _native(v) for k, v in d.items()}

        summary["results"]["mse_only"] = _convert(summary["results"]["mse_only"])
        summary["results"]["mse_infonce"] = _convert(summary["results"]["mse_infonce"])
        json.dump(summary, f, indent=2)
    logger.info("F4 complete. ΔNBER_F1=%+.4f", delta_f1)


if __name__ == "__main__":
    main()
