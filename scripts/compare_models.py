#!/usr/bin/env python3
"""Paired RF vs LR AUC tests across the 24 species-mode-variant combinations.

Run from code/: .venv/bin/python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import statsmodels
from statsmodels.stats.multitest import multipletests


SPECIES = ["roe_deer", "moose", "wild_boar", "fallow_deer"]
MODES = ["default", "road", "rail"]
VARIANTS = ["lag", "no_lag"]

COMBINATIONS = [(s, m, v) for s in SPECIES for m in MODES for v in VARIANTS]


def load_cv_results(base_dir: Path, species: str, mode: str, variant: str) -> pd.DataFrame:
    csv_path = base_dir / f"{species}_{mode}_{variant}" / f"cv_results_{species}.csv"
    return pd.read_csv(csv_path)


def extract_paired_aucs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rf_s = df[df["model"] == "rf"].set_index("fold")["auc"].sort_index()
    lr_s = df[df["model"] == "logreg"].set_index("fold")["auc"].sort_index()
    common = rf_s.index.intersection(lr_s.index)
    return rf_s.loc[common].values, lr_s.loc[common].values


def paired_stats(rf: np.ndarray, lr: np.ndarray) -> dict:
    delta = rf - lr
    n = len(delta)
    mean_d = float(np.mean(delta))
    sd_d = float(np.std(delta, ddof=1))
    se_d = sd_d / np.sqrt(n)
    t_crit = scipy.stats.t.ppf(0.975, df=n - 1)
    ci_lo = mean_d - t_crit * se_d
    ci_hi = mean_d + t_crit * se_d

    _, t_p = scipy.stats.ttest_rel(rf, lr)

    try:
        _, w_p = scipy.stats.wilcoxon(delta, alternative="two-sided", correction=False)
    except ValueError as exc:
        w_p = 1.0
        print(f"  WARNING wilcoxon: {exc}", file=sys.stderr)

    return {
        "n": n,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "se_delta": se_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "zero_in_ci": ci_lo <= 0 <= ci_hi,
        "ttest_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "rf_wins": mean_d > 0,
    }


def apply_corrections(pvals: list[float]) -> tuple[list[float], list[float]]:
    arr = np.array(pvals)
    _, bonf, _, _ = multipletests(arr, alpha=0.05, method="bonferroni")
    _, fdr, _, _ = multipletests(arr, alpha=0.05, method="fdr_bh")
    return list(bonf), list(fdr)


def binomial_test(k: int, n: int) -> tuple[float, float]:
    one_p = scipy.stats.binomtest(k, n, p=0.5, alternative="greater").pvalue
    two_p = scipy.stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue
    return one_p, two_p


def compute_all(per_species_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    wilcoxon_pvals = []

    for species, mode, variant in COMBINATIONS:
        df = load_cv_results(per_species_dir, species, mode, variant)
        rf_aucs, lr_aucs = extract_paired_aucs(df)
        stats = paired_stats(rf_aucs, lr_aucs)
        wilcoxon_pvals.append(stats["wilcoxon_p"])
        rows.append({"species": species, "mode": mode, "variant": variant, **stats})

    bonf_pvals, fdr_pvals = apply_corrections(wilcoxon_pvals)
    for i, r in enumerate(rows):
        r["bonf_p"] = bonf_pvals[i]
        r["fdr_p"] = fdr_pvals[i]

    def wins(mode_filter):
        subset = [r for r in rows if mode_filter is None or r["mode"] in mode_filter]
        return sum(1 for r in subset if r["rf_wins"]), len(subset)

    binomial_results = {}
    groups = [
        ("all_24", None),
        ("default", ["default"]),
        ("road", ["road"]),
        ("rail", ["rail"]),
        ("default_road", ["default", "road"]),
    ]
    for key, modes in groups:
        k, n = wins(modes)
        one_p, two_p = binomial_test(k, n)
        binomial_results[key] = {"k": k, "n": n, "one_sided_p": one_p, "two_sided_p": two_p}

    return rows, binomial_results


def fmt_p(val, decimals=4):
    s = f"{val:.{decimals}f}"
    return s + ("*" if val < 0.05 else " ")


def ascii_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  ".join("-" * w for w in widths)
    hdr = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body = "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )
    return f"{hdr}\n{sep}\n{body}"


def print_paired_table(rows):
    headers = ["species", "mode", "variant", "n",
               "mean_delta", "SD", "ci_lo", "ci_hi",
               "t_p", "wilcox_p", "bonf_p", "fdr_p"]
    data = []
    for r in rows:
        data.append([
            r["species"],
            r["mode"],
            r["variant"],
            str(r["n"]),
            f"{r['mean_delta']:+.4f}",
            f"{r['sd_delta']:.4f}",
            f"{r['ci_lo']:+.4f}",
            f"{r['ci_hi']:+.4f}",
            fmt_p(r["ttest_p"]),
            fmt_p(r["wilcoxon_p"]),
            fmt_p(r["bonf_p"]),
            fmt_p(r["fdr_p"]),
        ])
    print(ascii_table(headers, data))
    print("* p < 0.05")


def print_binomial_table(binomial_results):
    labels = {
        "all_24": "all (24)",
        "default": "default (8)",
        "road": "road (8)",
        "rail": "rail (8)",
        "default_road": "default+road (16)",
    }
    headers = ["group", "RF_wins", "total", "one_sided_p", "two_sided_p"]
    data = []
    for key, br in binomial_results.items():
        data.append([
            labels[key],
            str(br["k"]),
            str(br["n"]),
            fmt_p(br["one_sided_p"]),
            fmt_p(br["two_sided_p"]),
        ])
    print(ascii_table(headers, data))
    print("* p < 0.05")


def main():
    here = Path(__file__).parent.parent
    per_species_dir = here / "outputs" / "per_species"

    print(f"scipy {scipy.__version__}  statsmodels {statsmodels.__version__}")
    print()

    rows, binomial_results = compute_all(per_species_dir)

    csv_cols = ["species", "mode", "variant", "n",
                "mean_delta", "sd_delta", "se_delta",
                "ci_lo", "ci_hi", "zero_in_ci",
                "ttest_p", "wilcoxon_p", "bonf_p", "fdr_p", "rf_wins"]
    pd.DataFrame(rows)[csv_cols].to_csv(
        per_species_dir / "model_comparison.csv", index=False, float_format="%.6f"
    )

    print("paired delta AUC (RF - LR)")
    print()
    print_paired_table(rows)
    print()
    print("binomial win-pattern (RF AUC > LR AUC)")
    print()
    print_binomial_table(binomial_results)
    print()
    print("CSV: outputs/per_species/model_comparison.csv")


if __name__ == "__main__":
    main()
