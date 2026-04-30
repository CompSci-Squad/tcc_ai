# Panel Remediation Plan — Post-Sweep Critique Response

> Generated 2026-04-30 from the six-specialist review (Vargas/macro, Sato/ML, QA, Root-cause, Architect, DS).
> Source: chat session `fd27b18a` end-of-session synthesis.
> Status legend: ⏳ not started · 🟡 in progress · ✅ done · ❌ killed

---

## VERDICT (2026-04-30 night) — Phase A + B complete, **falsification TRIGGERED**

Pre-registered decision rule from B2 fired unambiguously. Reframe to **"Honest-ablation"** (Sato option). Summary table:

| Encoder (B1 split, same UMAP+HDBSCAN downstream) | Best TEST DBCV | Source |
|---|---|---|
| MLP-AE (1 hidden layer) | **0.789** | B2 falsification |
| Truncated SVD (no learning) | **0.666** | B2 falsification |
| linear-AE | 0.253 | B2 falsification |
| **iTransformer W6_d7_K4** (B3 retrain) | **0.166** | SM job `itransformer-1777581449-0d38`, $0.024 |

Combined evidence:
- **A1** confound check: `cluster × pre-2008` Cramer's V≈**0.43** (strong); `cluster × NBER` n.s. (p=0.83); ARI(full vs no-2020Q2)=0.29 (unstable to COVID).
- **B3** training diagnostics: train loss 0.93 vs val loss 57 → iTransformer fails to generalize 1965-99 → 2000-09 (2008 GFC scale unprecedented in TRAIN).
- **A3** sweep ceiling: best achievable DBCV across UMAP/t-SNE × HDBSCAN grid ≈ 0.33 — far below MLP-AE/SVD on the same data.

**Thesis pivot**: principal model is **windowed-PCA/SVD + HDBSCAN**; iTransformer demoted to a negative ablation. Phase C (multi-label panel, HDP-HMM proper, Bai-Perron) and Phase D (reproducibility) remain.

---

## Consensus damage report (what to fix)

1. **Headline `sil=0.638` is not defensible**: ablation selected on TEST (leakage), silhouette biased toward PCA, no permutation p-value, n_eff=11.
2. **Split is the root sin**: VAL=0 / TEST=0 NBER recession months → F1=0 mathematically forced. Confirmed root-cause confidence 0.95.
3. **2-cluster finding probably not "expansion vs contraction"**: confounded by COVID outliers, 2008 structural break, vol regimes.
4. **iTransformer may be unnecessary**: PCA-2D wins, linear-AE ≈ windowed-PCA in baselines, W=6 winning is partly a sweep artifact (different recon targets per W).
5. **HDP-HMM 1-state collapse is convergence failure, not fair negative baseline**: 80 Gibbs iters << standard >500; needs winsorization.

---

## Phase A — Diagnostic + cleanup (1 day, no AWS)

Goal: determine whether anything in current results survives scrutiny before spending more compute.

- [x] **A0 — Restore the LOCKED 7-metric panel** (~1 h) [QA, DS] **NON-NEGOTIABLE** ✅

  Per [`tcc_ai/docs/pre_analysis_plan.md` §2](../docs/pre_analysis_plan.md), every cell of every sweep must report exactly these 7 numbers on TEST, in this order:

  | # | Metric | Status in current `ALL.csv` | Action |
  |---|---|---|---|
  | 1 | `dbcv` (primary) | ❌ only `fit_train_dbcv` exists; TEST DBCV missing | Add `hdbscan.validity.validity_index(Z_test_2D, labels_test)` for HDBSCAN cells; for KMeans cells log as `NaN` with `dbcv_applicable=false` flag |
  | 2 | `hdbscan_test_n_clusters` | ✅ present as `n_clusters_test` | rename to canonical key |
  | 3 | `hdbscan_test_noise_fraction` | ✅ present as `noise_fraction` | rename to canonical key |
  | 4 | `nber_f1` (Hungarian-on-VAL frozen) | ✅ present (=0 structural until B1 re-split) | keep |
  | 5 | `nber_f1_legacy_maxF1` (bias witness) | ✅ present | keep |
  | 6 | `bai_perron_f1` | ❌ **MISSING from clustering ablation** | Wire `evaluation/regime_validation.bai_perron_overlap()` into `run_clustering_ablation.py`; tolerance=2 months on PC1 of test panel |
  | 7 | `crisis_window_coverage` | ❌ **MISSING** | Wire fraction of {dotcom, GFC, COVID} canonical crises whose dominant cluster matches the recession-mapped one from §3 |

  Edit: `scripts/run_clustering_ablation.py` (emit all 7 + provenance); `scripts/aggregate_ablation.py` (D3) must preserve column order; thesis tables must follow this exact order.

  **Cross-script wiring audit** — every results-producing script must emit the same 7 keys with the same names:

  | Script | dbcv | n_clusters | noise_frac | nber_f1 | nber_legacy | bai_perron_f1 | crisis_coverage |
  |---|---|---|---|---|---|---|---|
  | `run_clustering_ablation.py` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
  | `run_baselines.py` (B0–B3) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
  | `run_hdphmm_baseline.py` | ✅ (NaN ok) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

  → patch `run_clustering_ablation.py` and `run_baselines.py` to call `bai_perron_alignment()` and `crisis_window_coverage()` from `evaluation/regime_validation.py` (already implemented); add `dbcv` computation. Use HDP-HMM script as the reference template — it's already panel-complete.

- [x] **A1 — Confound check on `Z_test.parquet`** (~30 min) [DS] ✅ **TRIGGERED** (Cramer's V=0.43 on pre-2008)
  - cluster × NBER χ², cluster × pre-2008, cluster × `|ΔINDPRO|`, drop 2020-Q2 and refit
  - Output: `results/diagnostics/confound_check.md`
  - Decision gate: if cluster ↔ pre-2008 dominates, much of the writeup needs rethinking
- [x] **A2 — Replace silhouette → DBCV for HDBSCAN cells** (~30 min) [DS] ✅ folded into A0
  - `hdbscan.validity.validity_index(X, labels)`; keep silhouette for KMeans only (still in panel as exploratory under BH correction)
  - Add ARI vs NBER as exploratory headline aid (not in the 7 — but report in MLflow)
  - Edit: `src/tcc_itransformer/evaluation/clustering.py`, re-aggregate `ALL.csv`
  - **Folds into A0** — DBCV is metric #1 of the 7-panel, this just operationalizes it
- [x] **A3 — HDBSCAN/DR param sweep on failing cells** (~1 h) [DS] ✅ max DBCV ≈ 0.33
  - Grid: `min_cluster_size ∈ {5,10,15,20}` × `min_samples ∈ {1,3,5}`, UMAP `n_neighbors ∈ {5,15,30}`, t-SNE perplexity ∈ {5,15,30,50}
  - Plot HDBSCAN condensed-tree persistence for failing cells (UMAP+W6_d8, tSNE+W6_d9)
  - Output: `results/clustering_ablation/param_sweep.csv` + `figures/condensed_trees/`
- [x] **A4 — Pre-register tiebreak rule** (~1 h) [QA, Architect] ✅ winner = W6_d7_K3
  - Encode in `scripts/pick_stage2_winner.py`: tiebreak = (lowest K, fewest params, earliest best-val epoch)
  - Re-derive winner; if it flips back to W6_d9_K3, document and use it
- [x] **A5 — Add permutation p-values + bootstrap CIs to ablation report** (~1 h) [QA] ✅
  - Already implemented in `evaluation/statistical_tests.py` — just call from `run_clustering_ablation.py`
  - B=10000 perms, 95% CI bootstrap n=1000
  - Report `sil = 0.638 [CI: x.xx–x.xx], p = 0.0xx` for every cell

**Phase A decision gate:** if A1 shows COVID/era confound dominates, escalate Phase B priority and consider rescoping thesis framing (see Sato/Vargas contrarian takes, §Reframing).

---

## Phase B — Re-split + falsification (~$3 + 1 day)

Goal: settle the three biggest objections (split, silhouette inflation, architecture necessity) in one batch.

- [x] **B1 — Re-split** [Vargas, Root-cause, QA] ✅
  - **TRAIN 1965-01..1999-12** (420 mo, ~10 recessions)
  - **VAL 2000-01..2009-12** (120 mo, dot-com 8mo + GFC 18mo = 26 recession mo, 21.7%)
  - **TEST 2010-01..2026-01** (193 mo, COVID 2mo only — known thin, document)
  - Update: `src/tcc_itransformer/config.py` defaults, `configs/stage2_winner.yaml`, all sweep configs
  - Re-train operational winner only (1 SM job, ~$0.20)
- [x] **B2 — Falsification ablation** [Sato] ✅ MLP-AE 0.789 / SVD 0.666 / linear-AE 0.253 vs iTransformer 0.166 → **drop iTransformer**
  - Replace iTransformer with: (i) linear-AE matched in d, (ii) 2-layer MLP-AE, (iii) truncated SVD on flattened W=6 windows
  - Same downstream clustering pipeline; report silhouette + DBCV + ARI
  - Decision rule: if all within ±0.05 of iTransformer on regime-relevant metrics → **drop iTransformer, refocus thesis on windowed-PCA + HDBSCAN as main model**
  - Cost: ~2h CPU local, $0
- [x] **B3 — Re-run baselines + ablation on new split** (auto follow-on from B1) ✅ job `itransformer-1777581449-0d38` ($0.024)
  - `make baselines CONFIG=configs/baselines_op/W6_d7_K4.yaml`
  - `python scripts/run_clustering_ablation.py` (with A2/A3/A5 fixes applied)

**Phase B decision gate:** B2 result decides whether thesis is "iTransformer-based regime detection" or "lightweight regime detection with iTransformer as honest negative ablation."

---

## Phase C — Publication-grade evaluation (~1 week, mostly local)

- [ ] **C1 — Multi-label validation panel** [Vargas]
  - Pull from FRED: `RECPROUSM156N` (Chauvet-Piger MSI), `SAHMREALTIME` (Sahm rule), `CFNAI` (need MA3 < -0.7), OECD CLI Bry-Boschan dates
  - Add to `scripts/pull_nber.py` as `pull_external_labels.py`
  - Compute F1/AUC/ARI against each + composite (majority vote over 3 labels)
  - Output: extended `evaluation/regime_validation.py` with `multi_label_overlap()`
- [ ] **C2 — Re-run HDP-HMM properly** [Vargas, Architect]
  - Winsorize at 1%/99%, standardize per series
  - ≥500 Gibbs iters (sticky), 200 (sdhdp)
  - Log to MLflow: ELBO trajectory, active-state count per iter, concentration α posterior, state-occupancy histogram
  - Edit: `scripts/run_hdphmm_baseline.py`
- [ ] **C3 — Bai-Perron break agreement** [Vargas]
  - Run on headline series (INDPRO, PAYEMS, UNRATE, T10Y3M) with `trimming=0.15, max_breaks=5`
  - Compare break dates to cluster transitions with ±3-month tolerance window (Hamilton & Perez-Quiros 1996)
- [ ] **C4 — Cluster stability via bootstrap** (Ben-Hur 2002) [DS]
  - Resample 80% of windows ×100, compute Jaccard between each cluster and its best match per replicate
  - Report mean stability per cluster; clusters with Jaccard < 0.6 are not real (Hennig 2007)
  - Decision: drop unstable clusters from headline

---

## Phase D — Reproducibility cleanup (~4 h) [Architect]

- [ ] **D1** — Make `usrec.csv → nber_usrec.csv` scripted (3 LOC in `pull_nber.py`)
- [ ] **D2** — Compute + populate `data_sha256` in `stage2_winner.yaml` via `make freeze-config`
- [ ] **D3** — Promote ablation aggregation heredoc → `scripts/aggregate_ablation.py` (`make ablation-aggregate`)
- [ ] **D4** — Extend `ExperimentConfig` with `selection: {auto_winner, operational_winner, override_rationale, override_score}`; fail validation if override w/o score
- [ ] **D5** — `make reproduce-thesis`: pull NBER → verify SHAs → aggregate ablation → regenerate every figure
- [ ] **D6** — MLflow run linking: tag every SM job with `parent_sweep_id`, `stage`, `aws_job_arn`
- [ ] **D7** — Document Vocareum `LabRole` + m5.xlarge=20 quota + batch+backfill pattern at top of `tcc_ai/docs/deploy_aws.md`

---

## Reframing options (decide after Phase A + B2)

| Option | Trigger | Thesis title becomes |
|---|---|---|
| Status quo | A1 shows clean NBER alignment, B2 shows iTransformer beats baselines on ARI by >0.1 | "iTransformer-based unsupervised macroeconomic regime detection" |
| **Honest-ablation reframe (Sato)** | B2 shows iTransformer ≤ baselines on ARI ±0.05 | "Lightweight regime detection on FRED-MD: a benchmark study with iTransformer as a negative result" |
| **Monetary-regime reframe (Vargas)** | A1 shows era/monetary-policy confound dominates | "Unsupervised monetary-regime detection on FRED-MD validated against Wu-Xia shadow rate and Romer-Romer shocks" |
| **Multi-label benchmark reframe** | C1 shows model agrees with Sahm/Chauvet-Piger but not NBER | "Comparing unsupervised macro-regime detectors against multi-source recession labels" |

---

## Cost / time summary

| Phase | Wall time | AWS $ | Decision unlocked |
|---|---|---|---|
| A | 1 day local | 0 | Are current results salvageable? |
| B | 1 day local + 1h SM | ~$3 | Is the architecture necessary? Is the split fixable? |
| C | ~3-4 days local | 0 | Is the result publication-grade? |
| D | ~4 h | 0 | Is it reproducible? |

**Total to thesis-ready: ~1-2 weeks, ~$3 AWS.**

---

## Execution order for next session

1. **A1 first** (30 min) — single highest-information action; result determines whether to escalate Phase B priority
2. A4 + A5 in parallel with A2 + A3 (afternoon)
3. End-of-session checkpoint: decide if Phase B re-split is needed (almost certainly yes)
4. Update `SESSION_LOG.md` with Phase A findings + Phase B plan

## References to cite in thesis (collected from panel)

- Hamilton (1989) *Econometrica* — regime-switching baseline
- Chauvet (1998) *IER*; Chauvet & Piger (2008) *JBES* — MSI smoothed probabilities
- Stock & Watson (2002, 2016) *JBES* — factor models, structural breaks
- McCracken & Ng (2016) *JBES* — FRED-MD construction
- Bai & Perron (1998, 2003) *Econometrica/JAE* — multiple structural breaks
- Liu et al. (2024) *ICLR* — iTransformer
- Yue et al. (2022) *AAAI* — TS2Vec contrastive baseline
- Eldele et al. (2023) *NeurIPS* — TF-C
- Fox et al. (2011) *AoAS* — sticky HDP-HMM
- Campello et al. (2013); Moulavi et al. (2014) — HDBSCAN + DBCV
- McInnes et al. (2018) — UMAP
- Ben-Hur et al. (2002); Hennig (2007) — cluster stability
- Wu & Xia (2016) *JMCB* — shadow-rate (for monetary-regime reframe)
- Sahm (2019) — Sahm rule
