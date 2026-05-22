# Preprocessing Description — FRED-MD Panel for Regime Detection

**Version:** 1.0  
**Date:** 2026-05-30  
**Vintage:** FRED-MD 2026-04  
**Script:** `tcc_etl/` Lambda ETL (FRED API → S3 → local) + `scripts/data/prepare_panel.py`

---

## 1. Data Source

FRED-MD is a large monthly macroeconomic database maintained by McCracken & Ng
(2016). It provides 128 series at monthly frequency. The April 2026 vintage was
downloaded via the FRED API.

---

## 2. Series Selection (Balanced Panel)

The raw FRED-MD download (128 series) is filtered to a **balanced panel** using:

1. **Missingness filter:** any series with > 50% NaN values before 1965-01-01 is
   dropped. This removes early-vintage series (e.g., some survey measures) that
   start after 1970.
2. **Result:** 122 series remain. The panel has complete data from 1965-01 to
   2026-04 (734 months before splitting; 420 TRAIN + 120 VAL + 194 TEST).
3. **File:** `data/raw/fred_md_transformed_balanced_2026_04.parquet`

---

## 3. Transformation Codes

Each FRED-MD series carries a transformation code `tcode` (McCracken & Ng 2016,
Table A1). The codes applied are:

| tcode | Transformation | Example series |
|---|---|---|
| 1 | Level (no change) | FEDFUNDS |
| 2 | First difference | HOUST (housing starts) |
| 4 | Log level | — |
| 5 | First log difference (≈ growth rate) | INDPRO, PAYEMS, S&P 500 |
| 6 | Second log difference | — |

Transformation is applied **before** any scaling. This converts non-stationary
series to approximately stationary ones.

After transformation, the first 1–2 observations of differenced series are NaN
and are dropped row-wise (only for the warming-up period before 1965-01).

---

## 4. Winsorization

Extreme outliers are **winsorized at 5σ** using TRAIN statistics:

```
clip_low  = μ_train − 5 × σ_train
clip_high = μ_train + 5 × σ_train
```

Applied to all 122 transformed series. This prevents leverage effects from
extreme macro shocks (e.g., 2020-03 spike in unemployment) from distorting
the embedding space.

---

## 5. Standardization (Z-Score)

After winsorization, each series is standardized to zero mean and unit variance
using **TRAIN statistics only** (preventing data leakage into VAL/TEST):

```
X_scaled = (X − μ_train) / σ_train
```

where `μ_train` and `σ_train` are computed on TRAIN (1965-01 to 1999-12).

The same scaler is applied to VAL and TEST at prediction time.

---

## 6. Window Construction

The iTransformer operates on sliding windows of length W=6 months. Each
embedding covers the 6-month feature vector `[X_{t-5}, X_{t-4}, ..., X_t]`
concatenated across all 122 series:

```
Input shape per window: (W=6) × (F=122) = 732 features
```

Windows are constructed with stride=1 month (dense windowing). The date
assigned to each embedding window is the **last month of the window** (t).

**Warm-up:** The first W-1=5 months of each split are consumed as context and
do not produce labelled embeddings. Effective window counts:
- TRAIN: 420 - 5 = 415 windows
- VAL: 120 - 5 = 115 windows (but boundary windows re-use TRAIN context)
- TEST: 194 months

---

## 7. Stress Proxy Computation (for label-free cluster selection)

For diagnostic purposes, four stress proxies are derived from the standardized
panel (used as alternative label-free cluster selection on VAL; see protocol.md):

| Proxy | Formula | Recession direction |
|---|---|---|
| ΔUNRATE | `UNRATE_t − UNRATE_{t-1}` after tcode transform | Higher = worse |
| −INDPRO_growth | `−(INDPRO_t − INDPRO_{t-1})` (already log-diff) | Higher = worse |
| −PAYEMS_growth | `−(PAYEMS_t)` (already log-diff) | Higher = worse |
| −SP500_return | `−("S&P 500"_t)` (already log-diff) | Higher = worse |

Composite stress index: unweighted mean of the 4 proxies, each standardized
using TRAIN (1965-01 to 1999-12) mean and standard deviation.

The `"S&P 500"` series is available in the FRED-MD balanced panel under the
column name `"S&P 500"` (tcode=5, monthly log return).

---

## 8. Distribution Shift Diagnostics

For VAL and TEST, we report the fraction of series that have shifted
significantly relative to TRAIN:

- **KS test:** per-series two-sample Kolmogorov-Smirnov test (TRAIN vs TEST).
- **Threshold:** p < 0.05 after Bonferroni correction (122 tests).
- **Result:** see `outputs/tables/b1_distribution_shift.csv`

---

## 9. Scaler Variants Sensitivity

Three scaler variants are reported as a robustness check:

| Variant | Description |
|---|---|
| `standard_train` | Z-score on TRAIN (baseline, used in all main results) |
| `robust_train` | Median / IQR on TRAIN (less sensitive to outliers) |
| `standard_full` | Z-score on full panel (mild leakage; reported as upper bound) |

Results are in `outputs/tables/b1_scaler_variants.csv`.

---

## 10. Notes on Missing Variants NOT in Scope

- No calendar-time effects correction (demeaning by month of year)
- No series with publication lag adjustment
- No real-time data revision adjustment (uses vintage-of-download)
- No seasonal adjustment (FRED-MD series are already SA where applicable)
