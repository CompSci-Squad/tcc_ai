"""Tests for explain.py — regime explainability payload."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tcc_itransformer.evaluation.explain import (
    explain_assignment,
    explanations_to_frame,
)


def test_explain_with_soft_membership():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=20, freq="MS")
    panel = pd.DataFrame(rng.standard_normal((20, 4)), index=dates, columns=list("abcd"))
    labels = np.array([0] * 10 + [1] * 10)
    probs = np.full(20, 0.8)

    out = explain_assignment(panel, labels, probabilities=probs, top_k=3)
    assert len(out) == 20
    assert all(len(e.top_features) == 3 for e in out)
    assert out[0].membership == 0.8


def test_explain_with_distance_membership():
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(rng.standard_normal((10, 3)), columns=list("xyz"))
    labels = np.array([0] * 5 + [1] * 5)
    embeddings = rng.standard_normal((10, 2))
    centroids = np.array([embeddings[:5].mean(axis=0), embeddings[5:].mean(axis=0)])

    out = explain_assignment(
        panel,
        labels,
        centroids=centroids,
        embeddings=embeddings,
        membership_source="distance",
        top_k=2,
    )
    assert len(out) == 10
    # membership in [0, 1] when distance source
    for e in out:
        assert 0.0 <= e.membership <= 1.0


def test_explain_skips_noise():
    panel = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
    labels = np.array([-1, 0, 0, 0, 0])
    out = explain_assignment(
        panel, labels, probabilities=np.full(5, 0.9), top_k=2
    )
    assert out[0].top_features == []
    assert out[0].regime == -1


def test_explanations_to_frame():
    panel = pd.DataFrame(np.arange(12).reshape(4, 3).astype(float), columns=["a", "b", "c"])
    labels = np.array([0, 0, 1, 1])
    out = explain_assignment(panel, labels, probabilities=np.full(4, 1.0), top_k=2)
    df = explanations_to_frame(out)
    assert "feature" in df.columns
    assert "z_score" in df.columns
    assert len(df) == 4 * 2
