"""Tests for regime_validation: NBER, Bai-Perron, crises, moments, transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc_itransformer.evaluation.regime_validation import (
    crisis_window_coverage,
    nber_overlap,
    regime_conditional_moments,
    regime_durations,
    transition_matrix,
)


@pytest.fixture()
def two_regime_data() -> tuple[pd.DatetimeIndex, np.ndarray, pd.Series]:
    """50 months: first 25 = expansion (cluster 0), last 25 = recession (cluster 1)."""
    dates = pd.date_range("2000-01-01", periods=50, freq="MS")
    labels = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])
    usrec = pd.Series(
        np.concatenate([np.zeros(25), np.ones(25)]).astype(int),
        index=dates,
    )
    return dates, labels, usrec


def test_nber_overlap_perfect_match(two_regime_data):
    dates, labels, usrec = two_regime_data
    res = nber_overlap(labels, dates, usrec, lead=0, lag=0)
    assert res.matched_cluster == 1
    assert res.precision == pytest.approx(1.0)
    assert res.recall == pytest.approx(1.0)
    assert res.f1 == pytest.approx(1.0)


def test_nber_overlap_excludes_noise():
    dates = pd.date_range("2000-01-01", periods=10, freq="MS")
    labels = np.array([-1] * 5 + [0] * 5)
    usrec = pd.Series([0] * 5 + [1] * 5, index=dates)
    res = nber_overlap(labels, dates, usrec)
    assert res.matched_cluster == 0


def test_transition_matrix_two_regime():
    labels = np.array([0, 0, 0, 1, 1, 0, 0])
    P = transition_matrix(labels)
    assert list(P.index) == [0, 1]
    # 0->0 happens 3 times, 0->1 once -> P[0,0]=0.75, P[0,1]=0.25
    assert P.loc[0, 0] == pytest.approx(0.75)
    assert P.loc[0, 1] == pytest.approx(0.25)
    assert P.loc[1, 1] == pytest.approx(0.5)


def test_transition_matrix_excludes_noise():
    labels = np.array([0, -1, 1, 1])
    P = transition_matrix(labels)
    assert -1 not in P.index


def test_regime_durations_basic():
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
    out = regime_durations(labels)
    assert out.loc[0, "n_runs"] == 2
    assert out.loc[0, "max_duration"] == 4
    assert out.loc[1, "total_months"] == 2


def test_regime_conditional_moments_shape():
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.standard_normal((20, 3)), columns=list("abc"))
    labels = np.array([0] * 10 + [1] * 10)
    out = regime_conditional_moments(panel, labels)
    assert set(out.index.get_level_values("regime")) == {0, 1}
    assert set(out.index.get_level_values("statistic")) == {"mean", "std"}


def test_crisis_window_coverage_dominant_cluster():
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    labels = np.zeros(len(dates), dtype=int)
    # Force cluster 1 during GFC
    gfc_mask = (dates >= "2007-12-01") & (dates <= "2009-06-30")
    labels[gfc_mask] = 1
    out = crisis_window_coverage(labels, dates)
    assert out["gfc"]["dominant_cluster"] == 1
