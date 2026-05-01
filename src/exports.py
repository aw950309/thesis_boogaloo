"""Disk-export helpers for the WVC pipeline.

Public API:
    export_artefacts(model_df_clean, mean_importance, results_df, output_dir)

The exported CSVs are the FLAG_017 behaviour contract — the migrated
pipeline must produce these byte-equivalent to the Phase 1 baseline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_artefacts(
    model_df_clean: pd.DataFrame,
    mean_importance: pd.Series,
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the three CSV artefacts (cell 24 verbatim).

    FLAG_017 contract surface: model_df_clean.csv, feature_importance.csv,
    model_summary.csv must be byte-identical to the Phase 1 baseline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        results_df.groupby("model")[["auc", "precision", "recall", "f1", "accuracy"]]
        .agg(["mean", "std"])
        .round(3)
    )

    model_df_clean.to_csv(output_dir / "model_df_clean.csv", index=False)
    mean_importance.to_csv(output_dir / "feature_importance.csv")
    summary.to_csv(output_dir / "model_summary.csv")
