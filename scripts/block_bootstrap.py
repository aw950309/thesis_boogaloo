"""Naive and block bootstrap CIs on per-fold paired AUC differences.

Both bootstraps are computed independently per (species, mode, variant)
combination and merged into the existing ``model_comparison.csv`` produced
by ``scripts/compare_models.py``. Run after ``compare_models.py`` (which
writes the base file) and ``calc_epv_diagnostics.py`` (which merges the
EPV diagnostic columns).

Naive bootstrap (``run_naive_bootstrap``): i.i.d. percentile resample via
``scipy.stats.bootstrap``. Reported for completeness; not load-bearing,
because expanding-window CV induces serial dependence between adjacent
folds that violates the i.i.d. assumption. Columns merged in:
``boot_ci_lo``, ``boot_ci_hi``.

Block bootstrap (``run_block_bootstrap``): circular block resample via
``arch.bootstrap.CircularBlockBootstrap`` with ``block_length=12`` for
monthly folds (the annual physical cycle) and ``block_length=2`` for
yearly folds (smallest meaningful block with only 10 folds). Preserves
within-block serial dependence. Load-bearing CI for the rail no-lag
findings. Columns merged in: ``block_ci_lo``, ``block_ci_hi``,
``block_excludes_zero``, ``block_length``.

The two functions iterate the 24 combinations independently and each does
its own merge into ``model_comparison.csv``. They share a single loader
(``_load_paired_diffs``) that returns the per-combination paired delta
array, or ``None`` if the combination is absent or empty.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from arch.bootstrap import CircularBlockBootstrap
from scipy.stats import bootstrap

project_root = Path(__file__).resolve().parent.parent

SPECIES = ["roe_deer", "moose", "wild_boar", "fallow_deer"]
MODES = ["default", "road", "rail"]
VARIANTS = ["lag", "no_lag"]
COMBINATIONS = [(s, m, v) for s in SPECIES for m in MODES for v in VARIANTS]


def mean_func(x):
    return np.mean(x)


def _load_paired_diffs(per_species_dir: Path, sp: str, mode: str, variant: str, metric: str = "auc"):
    """Return the per-fold paired RF-LR delta array for the named metric, or None if missing.

    ``metric`` defaults to ``"auc"`` for back-compat. Pass ``"average_precision"``
    for AP paired diffs. Returns None if the cv_results file is absent, the
    metric column is missing, or no folds are paired.
    """
    cv_csv = per_species_dir / f"{sp}_{mode}_{variant}" / f"cv_results_{sp}.csv"
    if not cv_csv.exists():
        return None

    df_cv = pd.read_csv(cv_csv)
    if metric not in df_cv.columns:
        return None
    rf_s = df_cv[df_cv["model"] == "rf"].set_index("fold")[metric].sort_index()
    lr_s = df_cv[df_cv["model"] == "logreg"].set_index("fold")[metric].sort_index()
    common = rf_s.index.intersection(lr_s.index)
    if len(common) == 0:
        return None

    return rf_s.loc[common].values - lr_s.loc[common].values


def _merge_into_model_comparison(model_comp_csv: Path, df_res: pd.DataFrame) -> None:
    """Drop existing instances of df_res's value columns from the model_comparison
    CSV (if any) and re-merge, then write back."""
    df_comp = pd.read_csv(model_comp_csv)
    cols_to_drop = [
        c for c in df_comp.columns
        if c in df_res.columns and c not in ["species", "mode", "variant"]
    ]
    if cols_to_drop:
        df_comp = df_comp.drop(columns=cols_to_drop)
    df_comp = df_comp.merge(df_res, on=["species", "mode", "variant"], how="left")
    df_comp.to_csv(model_comp_csv, index=False, float_format="%.6f")


def _naive_bootstrap_one_metric(per_species_dir: Path, metric: str, col_lo: str, col_hi: str) -> list[dict]:
    """Run naive percentile bootstrap for one metric, return rows ready to merge."""
    results = []
    for sp, mode, variant in COMBINATIONS:
        paired_diffs = _load_paired_diffs(per_species_dir, sp, mode, variant, metric=metric)
        if paired_diffs is None:
            continue
        res = bootstrap(
            (paired_diffs,),
            mean_func,
            confidence_level=0.95,
            n_resamples=1000,
            method="percentile",
        )
        results.append({
            "species": sp,
            "mode": mode,
            "variant": variant,
            col_lo: float(res.confidence_interval.low),
            col_hi: float(res.confidence_interval.high),
        })
    return results


def run_naive_bootstrap(base_dir: str) -> None:
    """i.i.d. percentile bootstrap CI on the mean of the paired-diff series.

    Merges ``boot_ci_lo``, ``boot_ci_hi`` (AUC) and ``boot_ci_lo_ap``,
    ``boot_ci_hi_ap`` (AP) into ``model_comparison.csv``. AP columns will be
    NaN for combinations whose cv_results lack the ``average_precision`` column.
    """
    base_path = project_root / base_dir
    per_species_dir = base_path / "per_species"
    model_comp_csv = per_species_dir / "model_comparison.csv"

    if not model_comp_csv.exists():
        print(f"Skipping naive bootstrap for {base_dir}: model_comparison.csv not found")
        return

    print(f"Applying naive (i.i.d.) bootstrap for {base_dir}...")

    # AUC pass — existing column names preserved.
    auc_results = _naive_bootstrap_one_metric(
        per_species_dir, "auc", "boot_ci_lo", "boot_ci_hi"
    )
    if auc_results:
        _merge_into_model_comparison(model_comp_csv, pd.DataFrame(auc_results))

    # AP pass — new *_ap column names.
    ap_results = _naive_bootstrap_one_metric(
        per_species_dir, "average_precision", "boot_ci_lo_ap", "boot_ci_hi_ap"
    )
    if ap_results:
        _merge_into_model_comparison(model_comp_csv, pd.DataFrame(ap_results))
        print(f"Updated {model_comp_csv.relative_to(project_root)} with naive bootstrap CI (AUC + AP).\n")
    else:
        print(f"Updated {model_comp_csv.relative_to(project_root)} with naive bootstrap CI (AUC only — AP column absent).\n")


def _block_bootstrap_one_metric(
    per_species_dir: Path, block_length: int, metric: str,
    col_lo: str, col_hi: str, col_excl: str, col_len: str,
) -> list[dict]:
    """Run circular block bootstrap for one metric, return rows ready to merge."""
    results = []
    for sp, mode, variant in COMBINATIONS:
        paired_diffs = _load_paired_diffs(per_species_dir, sp, mode, variant, metric=metric)
        if paired_diffs is None:
            continue
        bs = CircularBlockBootstrap(block_length, paired_diffs)
        ci = bs.conf_int(mean_func, reps=1000, method="percentile")
        ci_lo = float(ci[0, 0])
        ci_hi = float(ci[1, 0])
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        results.append({
            "species": sp,
            "mode": mode,
            "variant": variant,
            col_lo: ci_lo,
            col_hi: ci_hi,
            col_excl: excludes_zero,
            col_len: block_length,
        })
    return results


def run_block_bootstrap(base_dir: str) -> None:
    """Circular block bootstrap CI on the mean, preserving serial dependence.

    Merges ``block_ci_lo``, ``block_ci_hi``, ``block_excludes_zero``,
    ``block_length`` (AUC) and ``block_ci_lo_ap``, ``block_ci_hi_ap``,
    ``block_excludes_zero_ap``, ``block_length_ap`` (AP) into
    ``model_comparison.csv``. AP columns will be NaN for combinations whose
    cv_results lack the ``average_precision`` column.
    """
    base_path = project_root / base_dir
    per_species_dir = base_path / "per_species"
    model_comp_csv = per_species_dir / "model_comparison.csv"

    if not model_comp_csv.exists():
        print(f"Skipping block bootstrap for {base_dir}: model_comparison.csv not found")
        return

    print(f"Applying Circular Block Bootstrap for {base_dir}...")

    block_length = 2 if "year" in base_dir else 12

    # AUC pass — existing column names preserved.
    auc_results = _block_bootstrap_one_metric(
        per_species_dir, block_length, "auc",
        "block_ci_lo", "block_ci_hi", "block_excludes_zero", "block_length",
    )
    if auc_results:
        df_auc = pd.DataFrame(auc_results)
        _merge_into_model_comparison(model_comp_csv, df_auc)
        print(f"Updated {model_comp_csv.relative_to(project_root)} with block bootstrap CI (AUC).")
        print("AUC Block Bootstrap Results Summary:")
        print(df_auc.to_string(index=False))
        print()

    # AP pass — new *_ap column names.
    ap_results = _block_bootstrap_one_metric(
        per_species_dir, block_length, "average_precision",
        "block_ci_lo_ap", "block_ci_hi_ap", "block_excludes_zero_ap", "block_length_ap",
    )
    if ap_results:
        df_ap = pd.DataFrame(ap_results)
        _merge_into_model_comparison(model_comp_csv, df_ap)
        print(f"Updated {model_comp_csv.relative_to(project_root)} with block bootstrap CI (AP).")
        print("AP Block Bootstrap Results Summary:")
        print(df_ap.to_string(index=False))
        print()
    else:
        print(f"(AP block bootstrap skipped: average_precision column absent from this tree's cv_results)\n")


if __name__ == "__main__":
    run_naive_bootstrap("outputs")
    run_naive_bootstrap("outputs_year")
    run_block_bootstrap("outputs")
    run_block_bootstrap("outputs_year")
