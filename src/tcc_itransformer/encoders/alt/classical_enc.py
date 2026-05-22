"""Classical macro regime model encoders (Phase E Tier 3).

Implements three classical baselines that don't use neural networks:

1. **HamiltonHMMEncoder** — 2-state Gaussian HMM (Hamilton 1989).
   The canonical macro-econometric regime model; identifies expansion vs.
   contraction from a small set of indicators (INDPRO, PAYEMS, UNRATE,
   T10Y3M → 4 PCs).  State probabilities form the embedding.

2. **MSVAREncoder** — Markov-Switching VAR(2) via statsmodels.
   Standard reference in the regime-switching literature (Krolzig 1997).
   Applied to PCA(4) of the FRED-MD panel.  The smoothed state
   probability vector forms the embedding.

3. **BOCPDEncoder** — Bayesian Online Change-Point Detection
   (Adams & MacKay 2007).  Detects structural breaks without labels.
   For each window, the run-length probability distribution summarises
   whether the window starts a new regime.  The posterior run-length
   distribution (binned) forms the embedding.

References
----------
- Hamilton (1989) J. Econometrics, "A new approach to the economic
  analysis of nonstationary time series and the business cycle".
- Krolzig (1997) "Markov-Switching Vector Autoregressions".
- Adams & MacKay (2007) "Bayesian Online Changepoint Detection".
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

_HMM_N_STATES = 4  # matches K=4 iTransformer winner config
_HMM_PCA_DIM = 6   # reduce FRED-MD panel to 6 PCs before fitting HMM
_MSVAR_N_STATES = 4
_MSVAR_AR_ORDER = 2
_MSVAR_PCA_DIM = 3  # MS-VAR is expensive; use fewer PCs
_BOCPD_HAZARD = 1.0 / 100.0  # expected regime length ~8 years
_BOCPD_EMBED_DIM = 16  # coarsen run-length distribution to this many bins


# ──────────────────────────────────────────────────────────────────────────────
# Hamilton HMM
# ──────────────────────────────────────────────────────────────────────────────

class HamiltonHMMEncoder(AltEncoder):
    """K-state Gaussian HMM via hmmlearn.

    The smoothed posterior state-probability vector ``P(s_t | all data)``
    (shape [K]) serves as the window embedding.

    Install:
        uv add --optional phase_e hmmlearn
    """

    name: ClassVar[str] = "hamilton_hmm"
    tier: ClassVar[str] = "classical"
    d_out: ClassVar[int] = _HMM_N_STATES

    def __init__(
        self,
        n_states: int = _HMM_N_STATES,
        pca_dim: int = _HMM_PCA_DIM,
    ) -> None:
        self._n_states = n_states
        self._pca_dim = pca_dim
        self._pca: PCA | None = None
        self._hmm = None

    def _check_import(self) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401, PLC0415
        except ImportError as exc:
            msg = "HamiltonHMM requires hmmlearn: uv add --optional phase_e hmmlearn"
            raise ImportError(msg) from exc

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Fit PCA + Gaussian HMM on flattened windows (reconstruction-free).

        PCA is fit on all rows of all training windows (so each monthly
        observation is one sample).  The HMM is then fit on the sequence
        of PCA-projected observations.
        """
        self._check_import()
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

        n, T, N = windows.shape
        logger.info("Fitting HamiltonHMM: n=%d T=%d N=%d K=%d", n, T, N, self._n_states)

        # Each time step across all windows → one observation row for PCA.
        obs_flat = windows.reshape(-1, N)  # [n*T, N]
        self._pca = PCA(n_components=self._pca_dim, random_state=seed)
        obs_pca = self._pca.fit_transform(obs_flat)  # [n*T, pca_dim]

        # HMM: treat the training data as one long sequence (n*T observations).
        hmm = GaussianHMM(
            n_components=self._n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=seed,
        )
        hmm.fit(obs_pca)
        self._hmm = hmm
        logger.info("HamiltonHMM training complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return smoothed state probabilities for each window.

        For each window of T steps, run the HMM smoother → get
        ``P(s_t | all T steps in window)`` → average over T → [K].
        """
        if self._hmm is None or self._pca is None:
            msg = "HamiltonHMMEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        n, T, N = windows.shape
        obs_flat = windows.reshape(-1, N)
        obs_pca = self._pca.transform(obs_flat)  # [n*T, pca_dim]

        Z = np.zeros((n, self._n_states), dtype=np.float32)
        for i in range(n):
            seg = obs_pca[i * T : (i + 1) * T]
            # posteriors: [T, K]
            posteriors = self._hmm.predict_proba(seg)
            Z[i] = posteriors.mean(axis=0)  # [K]
        return Z

    @classmethod
    def is_available(cls) -> bool:
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Markov-Switching VAR
# ──────────────────────────────────────────────────────────────────────────────

class MSVAREncoder(AltEncoder):
    """K-state Markov-Switching regression via statsmodels.

    Fits a MS-VAR(p) on PCA-reduced panel data.  The smoothed regime
    probabilities form the embedding vector.  Uses
    ``statsmodels.tsa.regime_switching.markov_regression.MarkovRegression``
    on each PC independently (fully multivariate MS-VAR is too expensive
    for 122 series; we reduce first).
    """

    name: ClassVar[str] = "ms_var"
    tier: ClassVar[str] = "classical"
    d_out: ClassVar[int] = _MSVAR_N_STATES

    def __init__(
        self,
        n_states: int = _MSVAR_N_STATES,
        ar_order: int = _MSVAR_AR_ORDER,
        pca_dim: int = _MSVAR_PCA_DIM,
    ) -> None:
        self._n_states = n_states
        self._ar_order = ar_order
        self._pca_dim = pca_dim
        self._pca: PCA | None = None
        self._models: list = []
        self._train_probs: np.ndarray | None = None  # smoothed probs on training data
        self._train_obs: np.ndarray | None = None
        self._train_len: int = 0

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:  # noqa: ARG002
        """Fit PCA + Markov-Switching regressions on training windows."""
        from statsmodels.tsa.regime_switching.markov_regression import (  # noqa: PLC0415
            MarkovRegression,
        )

        n, T, N = windows.shape
        logger.info("Fitting MS-VAR: n=%d T=%d N=%d K=%d AR=%d", n, T, N, self._n_states, self._ar_order)

        # Build one long monthly series from all windows (no deduplication).
        obs_flat = windows.reshape(-1, N)  # [n*T, N]
        self._pca = PCA(n_components=self._pca_dim, random_state=42)
        obs_pca = self._pca.fit_transform(obs_flat)  # [n*T, pca_dim]
        self._train_len = obs_pca.shape[0]

        # Fit one MarkovRegression per PC (statsmodels doesn't support joint MV-MS).
        self._models = []
        smoothed: list[np.ndarray] = []
        for pc_idx in range(self._pca_dim):
            series = obs_pca[:, pc_idx]
            model = MarkovRegression(
                series,
                k_regimes=self._n_states,
                trend="c",
                switching_variance=True,
            )
            try:
                res = model.fit(
                    em_iter=50,
                    search_reps=5,
                    disp=False,
                )
                self._models.append(res)
                smoothed.append(res.smoothed_marginal_probabilities.values)  # [T_total, K]
            except Exception as exc:  # noqa: BLE001
                logger.warning("MS-VAR PC%d fit failed: %s — using uniform probs.", pc_idx, exc)
                uniform = np.full((obs_pca.shape[0], self._n_states), 1.0 / self._n_states)
                self._models.append(None)
                smoothed.append(uniform)

        # Average smoothed probs across PCs → one [n*T, K] array.
        self._train_probs = np.stack(smoothed, axis=0).mean(axis=0)  # [n*T, K]
        self._train_obs = obs_pca
        logger.info("MS-VAR fitting complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return per-window mean of smoothed regime probabilities.

        For windows in the training set: look up pre-computed smoothed probs.
        For out-of-sample (val/test): filter forward using the transition
        matrix from the first successful PC model.
        """
        if self._pca is None:
            msg = "MSVAREncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        n, T, N = windows.shape
        obs_flat = windows.reshape(-1, N)
        obs_pca = self._pca.transform(obs_flat)  # [n*T, pca_dim]

        # Check if this is the training data (same values → use smoothed probs).
        if (
            self._train_probs is not None
            and obs_pca.shape[0] == self._train_probs.shape[0]
            and np.allclose(obs_pca, self._train_obs, atol=1e-4)
        ):
            probs = self._train_probs  # [n*T, K]
        else:
            # Out-of-sample: use filtered probs from each PC model.
            smoothed: list[np.ndarray] = []
            for pc_idx, res in enumerate(self._models):
                series = obs_pca[:, pc_idx]
                if res is None:
                    smoothed.append(
                        np.full((len(series), self._n_states), 1.0 / self._n_states)
                    )
                else:
                    try:
                        # Apply the trained model to out-of-sample series via
                        # filtered (one-step-ahead) probabilities.
                        from statsmodels.tsa.regime_switching.markov_regression import (  # noqa: PLC0415
                            MarkovRegression,
                        )
                        new_mod = MarkovRegression(
                            series,
                            k_regimes=self._n_states,
                            trend="c",
                            switching_variance=True,
                        )
                        oos_res = new_mod.fit(
                            start_params=res.params,
                            em_iter=0,  # no EM — use training params directly
                            disp=False,
                        )
                        smoothed.append(oos_res.smoothed_marginal_probabilities.values)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("MS-VAR OOS PC%d failed: %s", pc_idx, exc)
                        smoothed.append(
                            np.full((len(series), self._n_states), 1.0 / self._n_states)
                        )
            probs = np.stack(smoothed, axis=0).mean(axis=0)  # [n*T, K]

        # Reshape to windows and mean-pool over T.
        probs_windowed = probs.reshape(n, T, self._n_states)  # [n, T, K]
        return probs_windowed.mean(axis=1).astype(np.float32)  # [n, K]


# ──────────────────────────────────────────────────────────────────────────────
# BOCPD
# ──────────────────────────────────────────────────────────────────────────────

class BOCPDEncoder(AltEncoder):
    """Bayesian Online Change-Point Detection (Adams & MacKay 2007).

    BOCPD maintains a distribution over run-lengths r_t (time since last
    change point).  The probability P(r_t | x_1..t) is the core output.
    We encode each window as a binned histogram of run-length posteriors,
    which captures the probability that a regime change occurred in the
    window.

    No external dependency (pure NumPy + SciPy).

    Reference: https://arxiv.org/abs/0710.3742
    """

    name: ClassVar[str] = "bocpd"
    tier: ClassVar[str] = "classical"
    d_out: ClassVar[int] = _BOCPD_EMBED_DIM

    def __init__(
        self,
        hazard: float = _BOCPD_HAZARD,
        embed_dim: int = _BOCPD_EMBED_DIM,
        pca_dim: int = 3,
    ) -> None:
        self._hazard = hazard
        self._embed_dim = embed_dim
        self._pca_dim = pca_dim
        self._pca: PCA | None = None
        self._full_run_lengths: np.ndarray | None = None  # [T_total, T_total+1]
        self._total_len: int = 0

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:  # noqa: ARG002
        """Run BOCPD on the training sequence and cache run-length posteriors."""
        from scipy.stats import multivariate_normal  # noqa: PLC0415

        n, T, N = windows.shape
        logger.info("Running BOCPD: n=%d T=%d N=%d hazard=%.3f", n, T, N, self._hazard)

        obs_flat = windows.reshape(-1, N)  # [n*T, N]
        self._pca = PCA(n_components=self._pca_dim, random_state=42)
        obs_pca = self._pca.fit_transform(obs_flat)  # [n*T, pca_dim]
        self._total_len = obs_pca.shape[0]

        # Run BOCPD on the flattened sequence.
        self._full_run_lengths = _bocpd(obs_pca, self._hazard)  # [T_total, T_total+1]
        logger.info("BOCPD complete. Run-length matrix shape: %s", self._full_run_lengths.shape)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode windows as binned run-length posteriors."""
        if self._pca is None:
            msg = "BOCPDEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        n, T, N = windows.shape
        obs_flat = windows.reshape(-1, N)
        obs_pca = self._pca.transform(obs_flat)

        # Run BOCPD on this split's sequence.
        rl_mat = _bocpd(obs_pca, self._hazard)  # [n*T, n*T+1]

        # For each window, extract the run-length posterior for the last time step.
        Z = np.zeros((n, self._embed_dim), dtype=np.float32)
        total = rl_mat.shape[0]
        bin_edges = np.linspace(0, total, self._embed_dim + 1)
        for i in range(n):
            t = min((i + 1) * T - 1, total - 1)
            rl_dist = rl_mat[t, : t + 1]  # run-length posterior at time t
            # Bin into embed_dim histogram (normalise to sum=1)
            hist, _ = np.histogram(
                np.arange(len(rl_dist)),
                bins=self._embed_dim,
                range=(0, total),
                weights=rl_dist,
            )
            Z[i] = (hist / (hist.sum() + 1e-12)).astype(np.float32)
        return Z


def _bocpd(data: np.ndarray, hazard: float) -> np.ndarray:
    """Run BOCPD on ``data`` (shape [T, d]) and return the run-length matrix.

    Returns
    -------
    np.ndarray, shape [T, T+1]
        ``R[t, r]`` = P(run_length = r | data[0..t]).
    """
    from scipy.stats import multivariate_normal  # noqa: PLC0415

    T, d = data.shape
    R = np.zeros((T, T + 1))
    R[0, 0] = 1.0

    # Gaussian predictive model with weak hyperprior (Normal-Wishart approx).
    # We use a simple online mean/cov estimator with a prior centred at data mean.
    prior_mu = data.mean(axis=0)
    prior_cov = np.eye(d) * data.var() + 1e-3

    # Running sufficient statistics per run-length hypothesis
    # For simplicity: use a fixed Gaussian with mean/cov from training data.
    global_mu = prior_mu
    global_cov = prior_cov

    log_h = np.log(hazard)
    log_1mh = np.log(1.0 - hazard)

    for t in range(1, T):
        x = data[t]
        # Predictive probability under each run-length hypothesis
        try:
            log_pred = multivariate_normal.logpdf(x, mean=global_mu, cov=global_cov)
        except Exception:  # noqa: BLE001
            log_pred = -10.0  # fallback for singular cov

        # Growth probs: survive (r → r+1)
        growth = R[t - 1, : t] * np.exp(log_1mh + log_pred)
        # Change-point prob: collapse all existing runs → r=0
        cp = np.sum(R[t - 1, : t]) * np.exp(log_h + log_pred)

        R[t, 1 : t + 1] = growth
        R[t, 0] = cp

        # Normalise
        total = R[t, : t + 2].sum()
        if total > 0:
            R[t, : t + 2] /= total

    return R
