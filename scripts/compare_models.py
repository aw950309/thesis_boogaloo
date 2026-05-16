#!/usr/bin/env python3
"""Paired RF vs LR AUC tests across the 24 species-mode-variant combinations.

Run from code/: .venv/bin/python scripts/compare_models.py

Detects both outputs/per_species/ (monthly folds) and outputs_year/per_species/
(yearly folds) and runs the comparison for every fold strategy that has its
sweep present. Results land in each tree separately (per-tree
``model_comparison.csv``).
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


def _candidate_trees(repo_root: Path) -> list[tuple[str, Path]]:
    """List the (label, per_species_dir) trees we know how to compare.

    Returns every candidate even if the directory does not exist; the caller
    decides what to do with missing trees. Order is canonical: monthly first,
    then yearly.
    """
    return [
        ("monthly folds", repo_root / "outputs" / "per_species"),
        ("yearly folds",  repo_root / "outputs_year" / "per_species"),
    ]


def _is_complete(per_species_dir: Path) -> tuple[bool, list[str]]:
    """Verify that every (species, mode, variant) cv_results CSV is present.

    Returns (True, []) if complete, otherwise (False, [missing relative paths]).
    """
    missing: list[str] = []
    if not per_species_dir.exists():
        return False, ["<directory absent>"]
    for species, mode, variant in COMBINATIONS:
        rel = f"{species}_{mode}_{variant}/cv_results_{species}.csv"
        if not (per_species_dir / rel).exists():
            missing.append(rel)
    return (not missing), missing


def load_cv_results(base_dir: Path, species: str, mode: str, variant: str) -> pd.DataFrame:
    csv_path = base_dir / f"{species}_{mode}_{variant}" / f"cv_results_{species}.csv"
    return pd.read_csv(csv_path)


def extract_paired_metric(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return paired (rf_values, lr_values) for the named per-fold metric column.

    Folds present in both models are matched by index and sorted; folds missing
    in either model are dropped silently. Generalised version of
    ``extract_paired_aucs`` — use this for any per-fold metric column produced
    by ``evaluate_time_splits``.
    """
    rf_s = df[df["model"] == "rf"].set_index("fold")[metric].sort_index()
    lr_s = df[df["model"] == "logreg"].set_index("fold")[metric].sort_index()
    common = rf_s.index.intersection(lr_s.index)
    return rf_s.loc[common].values, lr_s.loc[common].values


def extract_paired_aucs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat wrapper. Equivalent to ``extract_paired_metric(df, 'auc')``.

    Preserved for ``_internal_audit_check.py`` which imports this symbol.
    """
    return extract_paired_metric(df, "auc")


def per_model_means(df: pd.DataFrame) -> dict:
    """Return mean AUC and mean Average Precision per model for this combination.

    AUC is included alongside AP for symmetry — both per-fold metrics produced by
    ``evaluate_time_splits``. Paired Δ stats remain AUC-only (see ``paired_stats``).
    """
    out = {}
    for model_name, suffix in [("logreg", "lr"), ("rf", "rf")]:
        sub = df[df["model"] == model_name]
        out[f"mean_auc_{suffix}"] = float(sub["auc"].mean()) if len(sub) else float("nan")
        if "average_precision" in sub.columns:
            out[f"mean_ap_{suffix}"] = float(sub["average_precision"].mean()) if len(sub) else float("nan")
        else:
            out[f"mean_ap_{suffix}"] = float("nan")
    return out


def paired_stats(rf: np.ndarray, lr: np.ndarray) -> dict:
    delta = rf - lr
    n = len(delta)
    mean_d = float(np.mean(delta))
    sd_d = float(np.std(delta, ddof=1))
    se_d = sd_d / np.sqrt(n)
    t_crit = scipy.stats.t.ppf(0.975, df=n - 1)
    ci_lo = mean_d - t_crit * se_d
    ci_hi = mean_d + t_crit * se_d

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
    """Compute paired AUC stats + binomial tests across the 24 combinations.

    Behaviour unchanged: AUC-only. Used by ``_internal_audit_check.py``.
    Includes ``per_model_means`` (mean AUC and mean AP per model) in each row.
    """
    rows = []
    wilcoxon_pvals = []

    for species, mode, variant in COMBINATIONS:
        df = load_cv_results(per_species_dir, species, mode, variant)
        rf_aucs, lr_aucs = extract_paired_aucs(df)
        stats = paired_stats(rf_aucs, lr_aucs)
        means = per_model_means(df)
        wilcoxon_pvals.append(stats["wilcoxon_p"])
        rows.append({"species": species, "mode": mode, "variant": variant, **stats, **means})

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


def compute_metric_all(per_species_dir: Path, metric: str) -> tuple[list[dict], dict]:
    """Compute paired stats + binomial tests across the 24 combinations for one metric column.

    Mirrors ``compute_all`` but reads the named per-fold metric column instead
    of hard-coded ``auc``. Combinations whose cv_results lack the metric column
    yield a row with only ``species/mode/variant`` (no stats); their Wilcoxon p
    is NOT included in the FDR / Bonferroni family, so the family size adjusts
    automatically. ``rf_wins`` is set to ``None`` for skipped rows and ignored
    by the binomial-test pass.

    Bonferroni and FDR corrections apply within this metric's family, not
    jointly with AUC. Returns ``(rows, binomial_results)`` in the same shape
    as ``compute_all``.
    """
    rows = []
    wilcoxon_pvals = []
    pval_indices = []  # row indices that contributed a Wilcoxon p

    for species, mode, variant in COMBINATIONS:
        df = load_cv_results(per_species_dir, species, mode, variant)
        if metric not in df.columns:
            rows.append({"species": species, "mode": mode, "variant": variant})
            continue
        rf_vals, lr_vals = extract_paired_metric(df, metric)
        if len(rf_vals) == 0:
            rows.append({"species": species, "mode": mode, "variant": variant})
            continue
        stats = paired_stats(rf_vals, lr_vals)
        wilcoxon_pvals.append(stats["wilcoxon_p"])
        pval_indices.append(len(rows))
        rows.append({"species": species, "mode": mode, "variant": variant, **stats})

    if wilcoxon_pvals:
        bonf_pvals, fdr_pvals = apply_corrections(wilcoxon_pvals)
        for j, i in enumerate(pval_indices):
            rows[i]["bonf_p"] = bonf_pvals[j]
            rows[i]["fdr_p"] = fdr_pvals[j]

    def wins(mode_filter):
        subset = [
            r for r in rows
            if (mode_filter is None or r["mode"] in mode_filter) and "rf_wins" in r
        ]
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
        if n > 0:
            one_p, two_p = binomial_test(k, n)
        else:
            one_p, two_p = float("nan"), float("nan")
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
               "wilcox_p", "bonf_p", "fdr_p"]
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


def _load_comparison(csv_path: Path) -> pd.DataFrame:
    """Load a model_comparison.csv and index it by (species, mode, variant)."""
    return (
        pd.read_csv(csv_path)
        .set_index(["species", "mode", "variant"])
        .sort_index()
    )


def _build_aggregate_table(diff: pd.DataFrame) -> list[str]:
    """Markdown aggregate table comparing the two fold strategies."""
    n_combos = len(diff)
    rf_wins_m = int(diff["rf_wins_m"].fillna(False).astype(bool).sum())
    rf_wins_y = int(diff["rf_wins_y"].fillna(False).astype(bool).sum())
    sig_rf_m = int(((diff["mean_delta_m"] > 0) & (diff["fdr_p_m"] < 0.05)).sum())
    sig_rf_y = int(((diff["mean_delta_y"] > 0) & (diff["fdr_p_y"] < 0.05)).sum())
    sig_lr_m = int(((diff["mean_delta_m"] < 0) & (diff["fdr_p_m"] < 0.05)).sum())
    sig_lr_y = int(((diff["mean_delta_y"] < 0) & (diff["fdr_p_y"] < 0.05)).sum())
    direction_changes = int(diff["direction_change"].fillna(False).astype(bool).sum())
    sig_changes = int(diff["sig_change"].fillna(False).astype(bool).sum())
    mean_abs_m = float(diff["mean_delta_m"].abs().mean())
    mean_abs_y = float(diff["mean_delta_y"].abs().mean())
    median_n_m = int(diff["n_m"].median()) if diff["n_m"].notna().any() else 0
    median_n_y = int(diff["n_y"].median()) if diff["n_y"].notna().any() else 0

    rows = [
        ("Combinations",                            f"{n_combos}",            f"{n_combos}",            "0"),
        ("Median folds per combination",            f"{median_n_m}",          f"{median_n_y}",          f"{median_n_y - median_n_m:+d}"),
        ("RF wins (mean_delta > 0)",                f"{rf_wins_m} / {n_combos}", f"{rf_wins_y} / {n_combos}", f"{rf_wins_y - rf_wins_m:+d}"),
        ("FDR-significant RF wins (p<0.05)",        f"{sig_rf_m} / {n_combos}", f"{sig_rf_y} / {n_combos}", f"{sig_rf_y - sig_rf_m:+d}"),
        ("FDR-significant LR wins (p<0.05)",        f"{sig_lr_m} / {n_combos}", f"{sig_lr_y} / {n_combos}", f"{sig_lr_y - sig_lr_m:+d}"),
        ("Direction changes (RF↔LR)",               "—",                       "—",                       f"{direction_changes}"),
        ("Significance changes (sig↔non-sig)",      "—",                       "—",                       f"{sig_changes}"),
        ("Mean |ΔAUC| across combinations",         f"{mean_abs_m:.4f}",      f"{mean_abs_y:.4f}",      f"{mean_abs_y - mean_abs_m:+.4f}"),
    ]
    out = [
        "| metric | monthly | yearly | Δ (yearly − monthly) |",
        "|--------|---------|--------|----------------------|",
    ]
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    return out


def _build_side_by_side_table(diff: pd.DataFrame) -> list[str]:
    """Markdown per-combination side-by-side table (one row per species/mode/variant)."""
    headers = [
        "species", "mode", "variant",
        "n_m", "mean_Δ_m", "fdr_p_m",
        "n_y", "mean_Δ_y", "fdr_p_y",
        "Δ_diff", "direction", "sig",
    ]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for (sp, mode, var), row in diff.iterrows():
        n_m   = "" if pd.isna(row.get("n_m"))            else f"{int(row['n_m'])}"
        n_y   = "" if pd.isna(row.get("n_y"))            else f"{int(row['n_y'])}"
        md_m  = "" if pd.isna(row.get("mean_delta_m"))   else f"{row['mean_delta_m']:+.4f}"
        md_y  = "" if pd.isna(row.get("mean_delta_y"))   else f"{row['mean_delta_y']:+.4f}"
        fp_m  = "" if pd.isna(row.get("fdr_p_m"))        else f"{row['fdr_p_m']:.4f}"
        fp_y  = "" if pd.isna(row.get("fdr_p_y"))        else f"{row['fdr_p_y']:.4f}"
        dd    = "" if pd.isna(row.get("delta_mean_delta")) else f"{row['delta_mean_delta']:+.4f}"
        if pd.isna(row.get("direction_change")):
            direction = ""
        else:
            direction = "FLIPPED" if bool(row["direction_change"]) else "same"
        if pd.isna(row.get("sig_change")):
            sig = ""
        elif bool(row["sig_change"]):
            sig = f"FLIPPED ({'sig' if row['sig_m'] else 'ns'} → {'sig' if row['sig_y'] else 'ns'})"
        else:
            sig = "sig" if bool(row.get("sig_m", False)) else "ns"
        out.append(
            f"| {sp} | {mode} | {var} | {n_m} | {md_m} | {fp_m} | {n_y} | {md_y} | {fp_y} | {dd} | {direction} | {sig} |"
        )
    return out


def _build_notable_changes(diff: pd.DataFrame, threshold: float = 0.02) -> list[str]:
    """Markdown bullet list of combinations with direction flip, sig flip, or |Δ_diff| > threshold."""
    notable_mask = (
        diff["direction_change"].fillna(False).astype(bool)
        | diff["sig_change"].fillna(False).astype(bool)
        | (diff["delta_mean_delta"].abs() > threshold)
    )
    notable = diff[notable_mask]

    if len(notable) == 0:
        return [
            f"_None — every combination keeps the same RF↔LR direction, the same significance flag, "
            f"and |Δ_diff| ≤ {threshold:.2f} AUC across the two fold strategies._"
        ]

    out = [
        f"_{len(notable)} combination(s) flagged: direction flip, significance flip, "
        f"or |Δ_diff| > {threshold:.2f} AUC._",
        "",
    ]
    for (sp, mode, var), row in notable.iterrows():
        flags = []
        if bool(row.get("direction_change", False)):
            m_winner = "RF" if bool(row["rf_wins_m"]) else "LR"
            y_winner = "RF" if bool(row["rf_wins_y"]) else "LR"
            flags.append(f"direction flip (monthly → {m_winner}; yearly → {y_winner})")
        if bool(row.get("sig_change", False)):
            m_sig = "sig" if bool(row.get("sig_m", False)) else "ns"
            y_sig = "sig" if bool(row.get("sig_y", False)) else "ns"
            flags.append(f"significance flip (monthly {m_sig} → yearly {y_sig})")
        dd = row.get("delta_mean_delta")
        if pd.notna(dd) and abs(dd) > threshold:
            flags.append(f"|Δ_diff| = {dd:+.4f} AUC")
        out.append(f"- **{sp} / {mode} / {var}** — {'; '.join(flags)}")
    return out


def _write_difference_report(repo_root: Path) -> bool:
    """Write monthly_vs_yearly_diff.{md,csv} into both per_species trees.

    Returns True if the report was written, False if either tree's
    model_comparison.csv was missing.
    """
    monthly_dir = repo_root / "outputs" / "per_species"
    yearly_dir  = repo_root / "outputs_year" / "per_species"
    monthly_csv = monthly_dir / "model_comparison.csv"
    yearly_csv  = yearly_dir  / "model_comparison.csv"

    bar = "═" * 78
    print(bar)
    print("  MONTHLY vs YEARLY DIFF  ➜  side-by-side comparison")
    print(bar)

    if not monthly_csv.exists() or not yearly_csv.exists():
        missing = [p for p in (monthly_csv, yearly_csv) if not p.exists()]
        for p in missing:
            print(f"  (skipping diff report — {p.relative_to(repo_root)} not present yet)")
        print()
        return False

    m = _load_comparison(monthly_csv)
    y = _load_comparison(yearly_csv)

    keep_cols = ["n", "mean_delta", "ci_lo", "ci_hi", "fdr_p", "rf_wins"]

    diff = (
        m[keep_cols].add_suffix("_m")
        .join(y[keep_cols].add_suffix("_y"), how="outer")
    )
    diff["delta_mean_delta"] = diff["mean_delta_y"] - diff["mean_delta_m"]
    diff["direction_change"] = (
        diff["rf_wins_m"].astype("boolean") != diff["rf_wins_y"].astype("boolean")
    )
    diff["sig_m"] = diff["fdr_p_m"] < 0.05
    diff["sig_y"] = diff["fdr_p_y"] < 0.05
    diff["sig_change"] = diff["sig_m"] != diff["sig_y"]
    diff = diff.sort_index()

    agg_lines = _build_aggregate_table(diff)
    table_lines = _build_side_by_side_table(diff)
    notable_lines = _build_notable_changes(diff)

    md_lines: list[str] = []
    md_lines.append("# Monthly vs yearly fold comparison")
    md_lines.append("")
    md_lines.append("Auto-generated by `scripts/compare_models.py`. Compares paired RF-vs-LR")
    md_lines.append("statistics from the monthly-fold sweep (`outputs/per_species/`) against the")
    md_lines.append("yearly-fold sweep (`outputs_year/per_species/`).")
    md_lines.append("")
    md_lines.append("Column key:")
    md_lines.append("")
    md_lines.append("- `n_m`, `n_y` — number of folds for that combination (monthly / yearly).")
    md_lines.append("- `mean_Δ_m`, `mean_Δ_y` — fold-averaged `(RF AUC − LR AUC)`. Positive ⇒ RF wins.")
    md_lines.append("- `fdr_p_m`, `fdr_p_y` — Wilcoxon p-value with Benjamini–Hochberg FDR correction at α=0.05.")
    md_lines.append("- `Δ_diff` — `mean_Δ_y − mean_Δ_m`; how much yearly disagrees with monthly.")
    md_lines.append("- `direction` — `FLIPPED` if RF wins under one strategy and LR under the other.")
    md_lines.append("- `sig` — `sig` / `ns` under monthly, `FLIPPED` if the FDR significance disagrees.")
    md_lines.append("")
    md_lines.append("## Aggregate")
    md_lines.append("")
    md_lines.extend(agg_lines)
    md_lines.append("")
    md_lines.append("## Per-combination side-by-side")
    md_lines.append("")
    md_lines.extend(table_lines)
    md_lines.append("")
    md_lines.append("## Notable changes")
    md_lines.append("")
    md_lines.extend(notable_lines)
    md_lines.append("")

    md_text = "\n".join(md_lines)
    csv_diff = diff.reset_index()

    written: list[Path] = []
    for tree_dir in (monthly_dir, yearly_dir):
        md_path  = tree_dir / "monthly_vs_yearly_diff.md"
        csv_path = tree_dir / "monthly_vs_yearly_diff.csv"
        md_path.write_text(md_text, encoding="utf-8")
        csv_diff.to_csv(csv_path, index=False, float_format="%.6f")
        written.extend([md_path, csv_path])

    # Terminal summary
    for line in agg_lines:
        print(f"  {line}")
    print()
    for line in notable_lines:
        print(f"  {line}")
    print()
    print("  written:")
    for p in written:
        print(f"    {p.relative_to(repo_root)}")
    print()
    return True


def _run_for_tree(label: str, per_species_dir: Path) -> bool:
    """Run the paired-stats + binomial comparison for one fold-strategy tree.

    Returns True if the comparison ran, False if the tree was skipped (absent
    or incomplete).
    """
    ok, missing = _is_complete(per_species_dir)
    rel = per_species_dir.relative_to(per_species_dir.parent.parent)
    bar = "═" * 78
    print(bar)
    print(f"  {label.upper()}  ➜  {rel}")
    print(bar)

    if not ok:
        if missing == ["<directory absent>"]:
            print(f"  (skipping — {rel} does not exist; run the per-species sweep first)")
        else:
            print(f"  (skipping — {len(missing)} cv_results CSV(s) missing under {rel}, e.g. {missing[0]})")
        print()
        return False

    # AUC pass — unchanged behaviour.
    rows, binomial_results = compute_all(per_species_dir)

    # AP pass — parallel paired stats on the average_precision column.
    # Missing-column combinations (e.g. yearly cv_results pre-dating the AP
    # addition) yield rows with only species/mode/variant and contribute NaN
    # to the AP columns; the Bonferroni/FDR family auto-shrinks.
    ap_rows, ap_binomial = compute_metric_all(per_species_dir, "average_precision")
    ap_by_key = {(r["species"], r["mode"], r["variant"]): r for r in ap_rows}

    AP_SUFFIXED_KEYS = [
        "n", "mean_delta", "sd_delta", "se_delta",
        "ci_lo", "ci_hi",
        "wilcoxon_p", "bonf_p", "fdr_p", "rf_wins",
    ]
    for r in rows:
        key = (r["species"], r["mode"], r["variant"])
        ap_r = ap_by_key.get(key, {})
        for k in AP_SUFFIXED_KEYS:
            r[f"{k}_ap"] = ap_r.get(k)  # None / NaN when AP missing

    csv_cols = [
        "species", "mode", "variant", "n",
        # AUC paired stats:
        "mean_delta", "sd_delta", "se_delta",
        "ci_lo", "ci_hi",
        "wilcoxon_p", "bonf_p", "fdr_p", "rf_wins",
        # Per-model means:
        "mean_auc_lr", "mean_auc_rf",
        "mean_ap_lr", "mean_ap_rf",
        # AP paired stats (NEW):
        "n_ap",
        "mean_delta_ap", "sd_delta_ap", "se_delta_ap",
        "ci_lo_ap", "ci_hi_ap",
        "wilcoxon_p_ap", "bonf_p_ap", "fdr_p_ap", "rf_wins_ap",
    ]
    out_csv = per_species_dir / "model_comparison.csv"
    pd.DataFrame(rows)[csv_cols].to_csv(
        out_csv, index=False, float_format="%.6f"
    )

    binom_order = ["all_24", "default", "road", "rail", "default_road"]
    binom_rows = [
        {
            "group": key,
            "rf_wins": binomial_results[key]["k"],
            "total": binomial_results[key]["n"],
            "one_sided_p": binomial_results[key]["one_sided_p"],
            "two_sided_p": binomial_results[key]["two_sided_p"],
        }
        for key in binom_order
    ]
    binom_csv = per_species_dir / "binomial_tests.csv"
    pd.DataFrame(binom_rows).to_csv(binom_csv, index=False, float_format="%.6f")

    binom_rows_ap = [
        {
            "group": key,
            "rf_wins": ap_binomial[key]["k"],
            "total": ap_binomial[key]["n"],
            "one_sided_p": ap_binomial[key]["one_sided_p"],
            "two_sided_p": ap_binomial[key]["two_sided_p"],
        }
        for key in binom_order
    ]
    binom_ap_csv = per_species_dir / "binomial_tests_ap.csv"
    pd.DataFrame(binom_rows_ap).to_csv(binom_ap_csv, index=False, float_format="%.6f")

    print("paired delta AUC (RF - LR)")
    print()
    print_paired_table(rows)
    print()
    print("binomial win-pattern (RF AUC > LR AUC)")
    print()
    print_binomial_table(binomial_results)
    print()

    ap_rows_with_stats = [r for r in ap_rows if "mean_delta" in r]
    if ap_rows_with_stats:
        print("paired delta AP (RF - LR)")
        print()
        print_paired_table(ap_rows_with_stats)
        print()
        print("binomial win-pattern (RF AP > LR AP)")
        print()
        print_binomial_table(ap_binomial)
        print()
    else:
        print("(AP paired stats skipped: average_precision column absent from this tree's cv_results)")
        print()

    print(f"CSV: {out_csv.relative_to(per_species_dir.parent.parent)}")
    print()
    return True


def main():
    here = Path(__file__).parent.parent

    print(f"scipy {scipy.__version__}  statsmodels {statsmodels.__version__}")
    print()

    trees = _candidate_trees(here)
    ran_any = False
    for label, per_species_dir in trees:
        ran_any = _run_for_tree(label, per_species_dir) or ran_any

    if not ran_any:
        print("No fold-strategy tree had a complete sweep. Run scripts/train_final_model.py "
              "with --fold-unit month and/or --fold-unit year (or 'both') to populate "
              "outputs/per_species/ or outputs_year/per_species/ first.")
        sys.exit(1)

    # Diff report runs only after both per-tree model_comparison.csv files exist —
    # i.e. after a 'both' run, or after a monthly run followed by a yearly run.
    _write_difference_report(here)


if __name__ == "__main__":
    main()
