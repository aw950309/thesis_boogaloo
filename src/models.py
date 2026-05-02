"""Model training and evaluation — WVC pipeline.

Public API:
    load_hyperparameters(path)
    make_expanding_time_splits(months, min_train_months=12, test_horizon=1)
    evaluate_time_splits(model_df, features, target, splits, hyperparameters)
    fit_final_model(model_df, features, target, hyperparameters)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_hyperparameters(path: Path | str) -> dict:
    """Load the hyperparameters YAML."""
    return yaml.safe_load(Path(path).read_text())


def make_expanding_time_splits(
    months,
    min_train_months: int = 12,
    test_horizon: int = 1,
) -> list[tuple[list, list]]:
    """Build expanding-window time-based train/test splits (cell 13 verbatim).

    For each split index ``i`` in ``[min_train_months, len(months) - test_horizon]``,
    the train window is ``months[:i]`` and the test window is
    ``months[i:i + test_horizon]``. The window grows ("expands") with each fold.
    """
    months = list(months)
    return [
        (months[:i], months[i:i + test_horizon])
        for i in range(min_train_months, len(months) - test_horizon + 1)
    ]


def evaluate_time_splits(
    model_df: pd.DataFrame,
    features: list[str],
    target: str,
    splits: list[tuple[list, list]],
    hyperparameters: dict,
    return_lr_probs: bool = False,
) -> tuple:
    """Run LR + RF across expanding-window folds (cell 14 verbatim).

    For each fold: train LogisticRegression (with StandardScaler) and
    RandomForestClassifier; evaluate AUC / precision / recall / F1 / accuracy
    on the held-out test month. Accumulate RF out-of-fold probabilities and
    labels plus per-fold feature importances.

    Returns (results_df, oof_probs, oof_labels, mean_importance) by default.
    When return_lr_probs=True, returns (results_df, oof_rf_probs, oof_lr_probs,
    oof_labels, mean_importance) — 5 values. Existing callers (pooled pipeline)
    are unaffected by the default.

        results_df       — 2 × n_folds rows (one per (fold, model)).
        oof_probs        — np.array of RF out-of-fold predicted probabilities.
        oof_labels       — np.array of true labels matching oof_probs.
        mean_importance  — pd.Series of per-feature importance averaged across folds.

    `hyperparameters` shape:
        {"logistic_regression": {...}, "random_forest": {...}}
    """
    lr_params = hyperparameters["logistic_regression"]
    rf_params = hyperparameters["random_forest"]

    results = []
    oof_probs_rf: list[float] = []
    oof_probs_lr: list[float] = []
    oof_labels: list[int] = []
    fold_importances: list[pd.Series] = []

    for fold_idx, (train_months, test_months) in enumerate(splits, start=1):
        train = model_df[model_df["period_start"].isin(train_months)].copy()
        test = model_df[model_df["period_start"].isin(test_months)].copy()

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]

        if len(test) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        # ── Logistic Regression ──
        logreg = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**lr_params)),
        ])
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        y_prob_lr = logreg.predict_proba(X_test)[:, 1]

        results.append({
            "fold": fold_idx, "model": "logreg",
            "auc":       roc_auc_score(y_test, y_prob_lr),
            "precision": precision_score(y_test, y_pred_lr, zero_division=0),
            "recall":    recall_score(y_test, y_pred_lr, zero_division=0),
            "f1":        f1_score(y_test, y_pred_lr, zero_division=0),
            "accuracy":  accuracy_score(y_test, y_pred_lr),
        })

        # ── Random Forest ──
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]

        results.append({
            "fold": fold_idx, "model": "rf",
            "auc":       roc_auc_score(y_test, y_prob_rf),
            "precision": precision_score(y_test, y_pred_rf, zero_division=0),
            "recall":    recall_score(y_test, y_pred_rf, zero_division=0),
            "f1":        f1_score(y_test, y_pred_rf, zero_division=0),
            "accuracy":  accuracy_score(y_test, y_pred_rf),
        })

        oof_probs_rf.extend(y_prob_rf.tolist())
        oof_probs_lr.extend(y_prob_lr.tolist())
        oof_labels.extend(y_test.tolist())
        fold_importances.append(pd.Series(rf.feature_importances_, index=features))

    results_df = pd.DataFrame(results)
    mean_importance = (
        pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    )

    if return_lr_probs:
        return (
            results_df,
            np.array(oof_probs_rf),
            np.array(oof_probs_lr),
            np.array(oof_labels),
            mean_importance,
        )
    return results_df, np.array(oof_probs_rf), np.array(oof_labels), mean_importance


def fit_final_model(
    model_df: pd.DataFrame,
    features: list[str],
    target: str,
    hyperparameters: dict,
) -> tuple[RandomForestClassifier, CalibratedClassifierCV]:
    """Fit the final RF on all data and an isotonic-calibrated wrapper (cell 16 verbatim).

    `hyperparameters` shape:
        {"random_forest": {...}, "calibration": {"method": ..., "cv": ...}}
    """
    rf_params = hyperparameters["random_forest"]
    calib_params = hyperparameters["calibration"]

    X_all = model_df[features]
    y_all = model_df[target]

    rf_final = RandomForestClassifier(**rf_params)
    rf_final.fit(X_all, y_all)

    rf_calibrated = CalibratedClassifierCV(rf_final, **calib_params)
    rf_calibrated.fit(X_all, y_all)

    return rf_final, rf_calibrated
