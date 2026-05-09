"""Smoke tests for src.models — Phase 5 Rows 11, 12, 14."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import models


def test_load_hyperparameters_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "hp.yaml"
    yaml_path.write_text(
        "random_forest:\n  n_estimators: 50\nlogistic_regression:\n  max_iter: 100\n"
    )
    hp = models.load_hyperparameters(yaml_path)
    assert hp["random_forest"]["n_estimators"] == 50
    assert hp["logistic_regression"]["max_iter"] == 100


def test_make_expanding_time_splits_count_and_window() -> None:
    """132 unique months → 120 folds with min_train=12, test_horizon=1."""
    months = pd.date_range("2015-01-01", periods=132, freq="MS")
    splits = models.make_expanding_time_splits(months, min_train_months=12, test_horizon=1)
    assert len(splits) == 120

    # First fold: 12-month train + 1-month test.
    train_0, test_0 = splits[0]
    assert len(train_0) == 12
    assert len(test_0) == 1

    # Expanding: last fold's train is everything but the last month.
    train_last, test_last = splits[-1]
    assert len(train_last) == 131
    assert len(test_last) == 1

    # No fold's train and test overlap.
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))


def test_make_expanding_time_splits_test_horizon() -> None:
    months = pd.date_range("2020-01-01", periods=20, freq="MS")
    splits = models.make_expanding_time_splits(months, min_train_months=6, test_horizon=3)
    # range(6, 20 - 3 + 1) = range(6, 18) → 12 folds
    assert len(splits) == 12
    for tr, te in splits:
        assert len(te) == 3


def test_make_expanding_year_splits_count_and_window() -> None:
    """11 calendar years (2015..2025) → 10 folds with min_train_years=1."""
    months = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    splits = models.make_expanding_year_splits(months, min_train_years=1, test_horizon=1)
    assert len(splits) == 10

    # First fold: train = 2015 (12 months), test = 2016 (12 months).
    train_0, test_0 = splits[0]
    assert len(train_0) == 12
    assert len(test_0) == 12
    assert all(pd.Timestamp(m).year == 2015 for m in train_0)
    assert all(pd.Timestamp(m).year == 2016 for m in test_0)

    # Last fold: train = 2015..2024 (120 months), test = 2025 (12 months).
    train_last, test_last = splits[-1]
    assert len(train_last) == 120
    assert len(test_last) == 12
    assert all(pd.Timestamp(m).year == 2025 for m in test_last)

    # No fold's train and test overlap.
    for tr, te in splits:
        assert set(tr).isdisjoint(set(te))


def test_make_expanding_year_splits_partial_year() -> None:
    """Year folds tolerate partial years (e.g. 2015 starting in March)."""
    months = pd.date_range("2015-03-01", "2017-12-01", freq="MS")
    splits = models.make_expanding_year_splits(months, min_train_years=1, test_horizon=1)
    # Years are {2015, 2016, 2017} → 2 folds with min_train_years=1, test_horizon=1.
    assert len(splits) == 2
    train_0, test_0 = splits[0]
    assert len(train_0) == 10  # March-Dec 2015
    assert len(test_0) == 12   # 2016
    assert all(pd.Timestamp(m).year == 2015 for m in train_0)


def test_make_expanding_year_splits_test_horizon_two() -> None:
    months = pd.date_range("2015-01-01", "2024-12-01", freq="MS")  # 10 years
    splits = models.make_expanding_year_splits(months, min_train_years=2, test_horizon=2)
    # range(2, 10 - 2 + 1) = range(2, 9) → 7 folds
    assert len(splits) == 7
    for tr, te in splits:
        assert len(te) == 24  # 2 years × 12 months


def _toy_model_df(n_months: int = 24, n_cells: int = 50) -> pd.DataFrame:
    """Two-feature, two-class fixture with enough variety for AUC computation."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for m in months:
        for c in range(n_cells):
            x1 = rng.standard_normal()
            x2 = rng.standard_normal()
            y = int((x1 + x2 + rng.standard_normal() * 0.5) > 0.5)
            rows.append({"period_start": m, "cell_id": c, "x1": x1, "x2": x2, "risk": y})
    return pd.DataFrame(rows)


def test_evaluate_time_splits_smoke_on_toy() -> None:
    df = _toy_model_df()
    months = sorted(df["period_start"].unique())
    splits = models.make_expanding_time_splits(months, min_train_months=12, test_horizon=1)
    hp = {
        "logistic_regression": {"max_iter": 200, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 10, "random_state": 42, "n_jobs": 1},
    }
    results_df, oof_probs, oof_labels, mean_imp = models.evaluate_time_splits(
        df, ["x1", "x2"], "risk", splits, hp,
    )

    # Every non-skipped fold yields 2 rows (LR + RF).
    assert len(results_df) % 2 == 0
    assert len(results_df) > 0
    assert {"logreg", "rf"}.issubset(set(results_df["model"]))
    assert oof_probs.shape == oof_labels.shape
    assert ((oof_probs >= 0) & (oof_probs <= 1)).all()
    assert set(oof_labels.tolist()).issubset({0, 1})
    assert set(mean_imp.index) == {"x1", "x2"}


def test_fit_final_model_calibrated_probs_in_unit_interval() -> None:
    df = _toy_model_df(n_months=12, n_cells=20)
    hp = {
        "random_forest": {"n_estimators": 10, "random_state": 42, "n_jobs": 1},
        "calibration": {"method": "isotonic", "cv": 3},
    }
    rf_final, rf_cal = models.fit_final_model(df, ["x1", "x2"], "risk", hp)

    p_final = rf_final.predict_proba(df[["x1", "x2"]])[:, 1]
    p_cal = rf_cal.predict_proba(df[["x1", "x2"]])[:, 1]
    assert ((p_final >= 0) & (p_final <= 1)).all()
    assert ((p_cal >= 0) & (p_cal <= 1)).all()
