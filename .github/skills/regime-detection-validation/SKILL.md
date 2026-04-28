---
name: regime-detection-validation
description: Validação econômica e estrutural de regimes detectados em painéis macroeconômicos não supervisionados. Encapsula NBER overlap, breakpoints de Bai-Perron, métrica DBCV para clusters de densidade, momentos condicionais por regime, matriz de transição e duração média.
risk: safe
source: project (TCC)
date_added: '2026-04-28'
author: orchestrator
tags:
  - regime-switching
  - clustering-validation
  - macroeconomics
  - hdbscan
  - bai-perron
  - nber
tools:
  - python
  - statsmodels
  - hdbscan
  - scipy
---

# Regime Detection Validation

## Overview

Skill consolidada para a **camada de validação externa e interpretabilidade econômica** do pipeline de detecção de regimes do TCC. Fecha o gap entre clustering geométrico (silhouette/DBCV) e validade econômica (NBER, breakpoints, narrativa de crises).

Cobre as três camadas de avaliação exigidas pelo pré-projeto (Seções 4.4 e 5):

1. **Métricas internas para clusters de densidade** — DBCV (Density-Based Clustering Validation) como métrica nativa para HDBSCAN, com Silhouette/DB/CH calculados *somente sobre não-ruído* e fração de ruído reportada explicitamente.
2. **Alinhamento com pseudo-rótulos econômicos** — overlap (hit rate com lead/lag) entre o regime "crise" e a série binária NBER USREC; correspondência entre fronteiras de regime e breakpoints estruturais Bai-Perron (2003); inspeção de janelas canônicas (GFC 2007–09, dot-com 2001, COVID-março/2020).
3. **Interpretabilidade econômica por regime** — momentos condicionais (média, volatilidade, correlação intra-cluster) das séries-chave; persistência (matriz de transição empírica e duração média por regime).

## When to Use This Skill

- Sempre que `compute_clustering_metrics` for chamado em embeddings macroeconômicos.
- Antes de aprovar qualquer rótulo de regime para a tese — sem essa validação o resultado é apenas "clustering bonito".
- Em relatórios de explicabilidade (módulo 4 do pipeline).

## Do Not Use This Skill When

- Domínio não-econômico (sensores, NLP) — não há série NBER análoga.
- Clusters supervisionados com rótulo verdadeiro disponível.

## API esperada (a implementar em `src/tcc_itransformer/evaluation/regime_validation.py`)

```python
def density_based_metrics(X, labels) -> dict:
    """DBCV + silhouette/DB/CH no subset não-ruído + noise_fraction."""

def nber_overlap(regime_labels, dates, nber_series, lead=0, lag=2) -> dict:
    """Hit rate, precision, recall, F1 entre regime 'crise' e USREC, com janela de tolerância."""

def bai_perron_alignment(regime_labels, dates, key_series, max_breaks=5) -> dict:
    """Distância média entre fronteiras de regime e breakpoints estimados via statsmodels/ruptures."""

def crisis_window_coverage(regime_labels, dates,
    windows=[("2001-03","2001-11"), ("2007-12","2009-06"), ("2020-02","2020-04")]) -> dict:
    """Para cada janela canônica: regime modal, fração coberta, label presence."""

def regime_conditional_moments(panel_df, labels, key_cols) -> pd.DataFrame:
    """Para cada (cluster × variável): média, std, correlação intra-cluster."""

def transition_matrix(labels) -> np.ndarray:
    """Matriz P[i,j] = P(regime_t+1=j | regime_t=i)."""

def regime_durations(labels) -> pd.DataFrame:
    """Duração média (em meses) e n de episódios por regime."""

def explain_assignment(x_window, model, cluster_centroids, panel_means, top_k=5) -> dict:
    """Para uma nova janela: regime, distância normalizada / soft prob, top-k features com maior |z-score| vs perfil médio do cluster."""
```

## Dependências adicionais a declarar

```toml
hdbscan = ">=0.8.33"
umap-learn = ">=0.5.5"
ruptures = ">=1.1.9"      # backup para Bai-Perron
fredapi = ">=0.5.1"        # baixar USREC
```

## Decisões críticas

- **HDBSCAN é o método principal**, K-Means+PCA é baseline (alinhamento com pré-projeto §4.3).
- **Fração de ruído reportada SEMPRE** ao usar HDBSCAN; silhouette nativa não suporta ruído.
- **NBER USREC apenas como pseudo-rótulo de validação**, nunca como input.
- **Tolerância padrão lead=0, lag=2 meses** para hit rate (NBER tem lag de datação).
- **Janelas de crise canônicas** vêm do pré-projeto §3.2 — não inventar outras sem justificar.

## Referências

- Campello, Moulavi, Sander (2013) — HDBSCAN
- Moulavi et al. (2014) — DBCV
- Bai & Perron (2003) — Multiple structural breaks
- McCracken & Ng (2016) — FRED-MD
- pre_projeto_tcc.md §3.2, §4.3, §4.4, §5
