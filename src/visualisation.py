"""Visualisation — WVC pipeline.

Each public function returns ``(figure, underlying_data)`` so Phase 6 parity
verification compares at the data layer rather than at the rendered-PNG layer
(matplotlib backend / font / DPI noise would otherwise cause spurious failures).

Public API (Phase 5 rows 13, 15, 17-20):
    plot_calibration(oof_probs, oof_labels)
    plot_top_features(mean_importance, top_n=15)
    plot_spatial_risk_maps(rf_final, model_df_clean, grid)
    plot_roc(oof_probs, oof_labels)
    plot_precision_recall(oof_probs, oof_labels)
    plot_feature_importance_by_group(mean_importance, groups_dict)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def plot_calibration(oof_probs, oof_labels) -> tuple[Figure, pd.DataFrame]:
    """Calibration plot from RF OOF predictions (cell 15 verbatim).

    Returns ``(figure, DataFrame[prob_pred, prob_true])``.
    """
    prob_true, prob_pred = calibration_curve(oof_labels, oof_probs, n_bins=10)

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="RF (OOF)")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plot — Random Forest (out-of-fold)")
    ax.legend()

    data = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    return fig, data


def plot_top_features(
    mean_importance: pd.Series, top_n: int = 15
) -> tuple[Figure, pd.DataFrame]:
    """Top-N feature importance horizontal bar plot (cell 17 verbatim)."""
    top = mean_importance.head(top_n)

    fig, ax = plt.subplots()
    top.iloc[::-1].plot(kind="barh", ax=ax)  # reverse so the largest is on top
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Random Forest)")

    data = top.rename("importance").reset_index().rename(columns={"index": "feature"})
    return fig, data


def plot_spatial_risk_maps(
    rf_final,
    model_df_clean: pd.DataFrame,
    features: list[str],
    grid: gpd.GeoDataFrame,
    joined: gpd.GeoDataFrame,
) -> tuple[Figure, pd.DataFrame]:
    """Two-panel spatial map: observed counts + predicted risk (cell 19 verbatim).

    Returns ``(figure, DataFrame[cell_id, risk_prob])`` — per-cell mean of
    ``rf_final.predict_proba(...)[:, 1]``. Does not mutate ``model_df_clean``;
    the orchestrator adds ``risk_prob`` to ``model_df_clean`` explicitly
    before export (Row 22) so cell-24 export behaviour is preserved.

    Deviation from architecture_map.md §3.8: signature adds explicit
    ``features`` and ``joined`` arguments. ``features`` is needed because
    `predict_proba` requires the FEATURES column subset (which lives at the
    orchestrator top, not on the model object). ``joined`` is the
    spatial-joined points GeoDataFrame from Row 3 — needed for the
    observed-counts heatmap panel (cell 19 uses it directly).
    """
    df = model_df_clean.copy()
    df["risk_prob"] = rf_final.predict_proba(df[features])[:, 1]

    cell_risk = df.groupby("cell_id")["risk_prob"].mean().reset_index()
    grid_risk = grid.merge(cell_risk, on="cell_id", how="left")
    grid_risk["risk_prob"] = grid_risk["risk_prob"].fillna(0)

    cell_totals = joined.groupby("cell_id").size().reset_index(name="collision_count")
    grid_heatmap = grid.merge(cell_totals, on="cell_id", how="left")
    grid_heatmap["collision_count"] = grid_heatmap["collision_count"].fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    grid_heatmap.plot(
        column="collision_count", cmap="hot",
        linewidth=0.1, edgecolor="grey", legend=True, ax=axes[0],
    )
    axes[0].set_title("Wildlife collisions — observed count")
    axes[0].set_axis_off()

    grid_risk.plot(
        column="risk_prob", cmap="RdYlGn_r",
        linewidth=0.1, edgecolor="grey", legend=True, ax=axes[1],
    )
    axes[1].set_title("Predicted collision risk (probability)")
    axes[1].set_axis_off()
    plt.tight_layout()

    return fig, cell_risk


def plot_roc(oof_probs, oof_labels) -> tuple[Figure, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ROC curve (cell 20 verbatim, with thresholds preserved for parity)."""
    fpr, tpr, thresholds = roc_curve(oof_labels, oof_probs)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="RF")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return fig, (fpr, tpr, thresholds)


def plot_precision_recall(
    oof_probs, oof_labels
) -> tuple[Figure, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """PR curve + average-precision scalar (cell 21 verbatim, thresholds preserved)."""
    precision, recall, thresholds = precision_recall_curve(oof_labels, oof_probs)
    ap = average_precision_score(oof_labels, oof_probs)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"RF AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    return fig, (precision, recall, thresholds, ap)


def plot_feature_importance_by_group(
    mean_importance: pd.Series, groups_dict: dict[str, list[str]]
) -> tuple[Figure, pd.DataFrame]:
    """Per-group feature-importance sums (cell 22 verbatim).

    Returns ``(figure, DataFrame[group, importance_sum])``. The ``groups_dict``
    is supplied by the orchestrator (`scripts/train_final_model.py::GROUPS`).
    Per-group sum = sum of ``mean_importance[col]`` for each column listed
    in ``groups_dict[group]``.
    """
    group_importance = {
        k: float(mean_importance[v].sum()) for k, v in groups_dict.items()
    }
    series = pd.Series(group_importance).sort_values()

    fig, ax = plt.subplots()
    series.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance by Group")

    data = series.rename("importance_sum").reset_index().rename(columns={"index": "group"})
    return fig, data
