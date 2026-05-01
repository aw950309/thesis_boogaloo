"""Smoke tests for src.visualisation — Phase 5 Band G (Rows 13, 15, 17-20).

Verifies that each plotting function returns the documented (figure,
underlying_data) shape and that the data layer is well-formed. Pixel-level
figure parity is intentionally not tested — Phase 6 verifies parity at the
data layer per FIGURE_MAP.md.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import visualisation


@pytest.fixture
def oof() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 1000
    labels = rng.integers(0, 2, size=n)
    probs = labels * 0.6 + rng.random(n) * 0.4  # weakly correlated
    return probs, labels


@pytest.fixture
def mean_imp() -> pd.Series:
    return pd.Series(
        {f"feat_{i}": 1.0 / (i + 1) for i in range(20)},
        name="importance",
    ).sort_values(ascending=False)


def test_plot_calibration_returns_fig_and_xy(oof) -> None:
    probs, labels = oof
    fig, data = visualisation.plot_calibration(probs, labels)
    assert isinstance(fig, Figure)
    assert set(data.columns) == {"prob_pred", "prob_true"}
    assert (data["prob_pred"] >= 0).all() and (data["prob_pred"] <= 1).all()


def test_plot_top_features_returns_top_n(mean_imp) -> None:
    fig, data = visualisation.plot_top_features(mean_imp, top_n=5)
    assert isinstance(fig, Figure)
    assert len(data) == 5
    assert {"feature", "importance"}.issubset(data.columns)
    # Sorted descending.
    assert data["importance"].is_monotonic_decreasing


def test_plot_roc_returns_three_tuple(oof) -> None:
    probs, labels = oof
    fig, (fpr, tpr, thresholds) = visualisation.plot_roc(probs, labels)
    assert isinstance(fig, Figure)
    assert fpr.shape == tpr.shape == thresholds.shape
    assert (fpr >= 0).all() and (fpr <= 1).all()
    assert (tpr >= 0).all() and (tpr <= 1).all()


def test_plot_precision_recall_returns_four_tuple(oof) -> None:
    probs, labels = oof
    fig, (precision, recall, thresholds, ap) = visualisation.plot_precision_recall(probs, labels)
    assert isinstance(fig, Figure)
    assert precision.shape == recall.shape
    assert thresholds.shape[0] == precision.shape[0] - 1
    assert 0 <= ap <= 1


def test_plot_feature_importance_by_group(mean_imp) -> None:
    groups = {"group_a": ["feat_0", "feat_1"], "group_b": ["feat_2"]}
    fig, data = visualisation.plot_feature_importance_by_group(mean_imp, groups)
    assert isinstance(fig, Figure)
    assert set(data["group"]) == {"group_a", "group_b"}
    a = data.set_index("group")["importance_sum"]
    assert a["group_a"] == pytest.approx(mean_imp["feat_0"] + mean_imp["feat_1"])
    assert a["group_b"] == pytest.approx(mean_imp["feat_2"])
