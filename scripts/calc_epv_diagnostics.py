import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path so we can import models
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import models

SPECIES = ["roe_deer", "moose", "wild_boar", "fallow_deer"]
MODES = ["default", "road", "rail"]
VARIANTS = ["lag", "no_lag"]
COMBINATIONS = [(s, m, v) for s in SPECIES for m in MODES for v in VARIANTS]

def calc_epv_for_dir(base_dir: str, fold_unit: str):
    base_path = project_root / base_dir
    per_species_dir = base_path / "per_species"
    model_comp_csv = per_species_dir / "model_comparison.csv"

    if not model_comp_csv.exists():
        print(f"Skipping {base_dir}: model_comparison.csv not found")
        return

    print(f"Processing {base_dir} (fold unit: {fold_unit})...")

    # Load model_comparison.csv
    df_comp = pd.read_csv(model_comp_csv)

    results = []

    for sp, mode, variant in COMBINATIONS:
        combo_dir = per_species_dir / f"{sp}_{mode}_{variant}"
        if not combo_dir.exists():
            continue

        model_df_path = combo_dir / f"model_df_{sp}.csv"
        feat_imp_path = combo_dir / f"feature_importance_{sp}.csv"

        if not model_df_path.exists() or not feat_imp_path.exists():
            continue

        # 1. Load data
        df_var = pd.read_csv(model_df_path)
        df_var["period_start"] = pd.to_datetime(df_var["period_start"])
        feat_imp = pd.read_csv(feat_imp_path)
        n_features = len(feat_imp)

        # 2. Build splits
        months_s = sorted(df_var["period_start"].unique())
        if fold_unit == "month":
            splits = models.make_expanding_time_splits(months_s, min_train_months=12, test_horizon=1)
        else:
            splits = models.make_expanding_year_splits(months_s, min_train_years=1, test_horizon=1)

        # 3. Count positives per fold
        positives_per_fold = []
        for train_months, test_months in splits:
            test_mask = df_var["period_start"].isin(test_months)
            y_test = df_var[test_mask]["risk"]
            positives_per_fold.append(int(y_test.sum()))

        if not positives_per_fold:
            continue

        # 4. Compute metrics
        min_pos = int(np.min(positives_per_fold))
        median_pos = int(np.median(positives_per_fold))
        max_pos = int(np.max(positives_per_fold))

        min_epv = min_pos / n_features
        median_epv = median_pos / n_features

        results.append({
            "species": sp,
            "mode": mode,
            "variant": variant,
            "min_positives_per_fold": min_pos,
            "median_positives_per_fold": median_pos,
            "max_positives_per_fold": max_pos,
            "n_predictors": n_features,
            "min_epv_per_fold": min_epv,
            "median_epv_per_fold": median_epv,
        })

    df_epv = pd.DataFrame(results)

    # Save standalone summary
    epv_summary_path = per_species_dir / "epv_and_positives_summary.csv"
    df_epv.to_csv(epv_summary_path, index=False, float_format="%.2f")
    print(f"Saved standalone EPV summary to {epv_summary_path.relative_to(project_root)}")

    # Merge into model_comparison.csv
    # Drop existing EPV/pos columns if they exist to prevent duplication
    cols_to_drop = [c for c in df_comp.columns if c in df_epv.columns and c not in ["species", "mode", "variant"]]
    if cols_to_drop:
        df_comp = df_comp.drop(columns=cols_to_drop)

    df_merged = df_comp.merge(df_epv, on=["species", "mode", "variant"], how="left")
    df_merged.to_csv(model_comp_csv, index=False, float_format="%.6f")
    print(f"Updated {model_comp_csv.relative_to(project_root)} with EPV and positive-class statistics!\n")

if __name__ == "__main__":
    calc_epv_for_dir("outputs", "month")
    calc_epv_for_dir("outputs_year", "year")
