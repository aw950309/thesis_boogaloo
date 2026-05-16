"""Unit tests for scripts/compare_models.py.

Cover the helpers that drive paired-metric statistics across the 24
combinations sweep: ``extract_paired_metric``, ``paired_stats``,
``apply_corrections``, ``binomial_test``, and ``compute_metric_all``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# scripts/ is not a package; mirror _internal_audit_check.py's import style.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_models as cm  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _toy_cv_df() -> pd.DataFrame:
    """Synthetic cv_results-like DataFrame with auc and average_precision."""
    return pd.DataFrame([
        # fold 1
        {"fold": 1, "model": "logreg", "auc": 0.80, "average_precision": 0.40},
        {"fold": 1, "model": "rf",     "auc": 0.85, "average_precision": 0.50},
        # fold 2
        {"fold": 2, "model": "logreg", "auc": 0.70, "average_precision": 0.30},
        {"fold": 2, "model": "rf",     "auc": 0.78, "average_precision": 0.42},
        # fold 3
        {"fold": 3, "model": "logreg", "auc": 0.90, "average_precision": 0.55},
        {"fold": 3, "model": "rf",     "auc": 0.93, "average_precision": 0.62},
    ])


def _toy_cv_df_no_ap() -> pd.DataFrame:
    """Synthetic cv_results-like DataFrame WITHOUT average_precision."""
    df = _toy_cv_df()
    return df.drop(columns=["average_precision"])


# ---------------------------------------------------------------------------
# extract_paired_metric
# ---------------------------------------------------------------------------

def test_extract_paired_metric_auc():
    df = _toy_cv_df()
    rf, lr = cm.extract_paired_metric(df, "auc")
    assert rf.tolist() == [0.85, 0.78, 0.93]
    assert lr.tolist() == [0.80, 0.70, 0.90]


def test_extract_paired_metric_average_precision():
    df = _toy_cv_df()
    rf, lr = cm.extract_paired_metric(df, "average_precision")
    assert rf.tolist() == pytest.approx([0.50, 0.42, 0.62])
    assert lr.tolist() == pytest.approx([0.40, 0.30, 0.55])


def test_extract_paired_metric_missing_column_raises():
    df = _toy_cv_df_no_ap()
    with pytest.raises(KeyError):
        cm.extract_paired_metric(df, "average_precision")


def test_extract_paired_aucs_back_compat_wrapper():
    """The wrapper exists and matches extract_paired_metric(df, 'auc')."""
    df = _toy_cv_df()
    rf_a, lr_a = cm.extract_paired_aucs(df)
    rf_b, lr_b = cm.extract_paired_metric(df, "auc")
    np.testing.assert_array_equal(rf_a, rf_b)
    np.testing.assert_array_equal(lr_a, lr_b)


def test_extract_paired_metric_drops_unpaired_folds():
    """If RF has fold 4 but LR doesn't, the unpaired fold is dropped."""
    df = pd.DataFrame([
        {"fold": 1, "model": "logreg", "auc": 0.80},
        {"fold": 1, "model": "rf",     "auc": 0.85},
        {"fold": 4, "model": "rf",     "auc": 0.99},  # RF only
        {"fold": 2, "model": "logreg", "auc": 0.70},
        {"fold": 2, "model": "rf",     "auc": 0.78},
    ])
    rf, lr = cm.extract_paired_metric(df, "auc")
    assert rf.tolist() == [0.85, 0.78]
    assert lr.tolist() == [0.80, 0.70]


# ---------------------------------------------------------------------------
# paired_stats
# ---------------------------------------------------------------------------

def test_paired_stats_correctness_simple():
    """Hand-computed paired stats on a small synthetic series."""
    rf = np.array([0.85, 0.78, 0.93])
    lr = np.array([0.80, 0.70, 0.90])
    stats = cm.paired_stats(rf, lr)

    delta = rf - lr  # [0.05, 0.08, 0.03]
    expected_mean = float(np.mean(delta))
    expected_sd = float(np.std(delta, ddof=1))

    assert stats["n"] == 3
    assert stats["mean_delta"] == pytest.approx(expected_mean)
    assert stats["sd_delta"] == pytest.approx(expected_sd)
    assert stats["rf_wins"] is True
    assert 0.0 <= stats["wilcoxon_p"] <= 1.0


def test_paired_stats_negative_delta_rf_loses():
    rf = np.array([0.50, 0.45, 0.55])
    lr = np.array([0.60, 0.55, 0.65])
    stats = cm.paired_stats(rf, lr)
    assert stats["rf_wins"] is False
    assert stats["mean_delta"] < 0


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------

def test_apply_corrections_bonferroni_simple():
    """Bonferroni multiplies each p by family size, capped at 1.0."""
    pvals = [0.001, 0.02, 0.05, 0.5]
    bonf, _fdr = cm.apply_corrections(pvals)
    # With K=4, Bonferroni gives min(p*4, 1.0).
    assert bonf == pytest.approx([0.004, 0.08, 0.20, 1.0])


def test_apply_corrections_fdr_bh_ordering():
    """FDR-BH p-values preserve the ordering of raw p-values."""
    pvals = [0.001, 0.02, 0.04, 0.5]
    _bonf, fdr = cm.apply_corrections(pvals)
    # All FDR-corrected values should be >= raw p-values.
    for raw, corr in zip(pvals, fdr):
        assert corr >= raw - 1e-12


# ---------------------------------------------------------------------------
# binomial_test
# ---------------------------------------------------------------------------

def test_binomial_test_unanimous_one_sided():
    """24/24 RF wins → one-sided p is essentially zero."""
    one_p, two_p = cm.binomial_test(24, 24)
    assert one_p == pytest.approx(0.5 ** 24, abs=1e-12)
    assert two_p == pytest.approx(0.5 ** 24 * 2, abs=1e-12) or two_p < 1e-6


def test_binomial_test_chance():
    """12/24 → p ≈ 0.5 one-sided."""
    one_p, _two_p = cm.binomial_test(12, 24)
    assert 0.4 <= one_p <= 0.6


# ---------------------------------------------------------------------------
# compute_metric_all — integration with synthetic cv_results files
# ---------------------------------------------------------------------------

def _write_synthetic_sweep(root: Path, with_ap: bool = True) -> Path:
    """Write the 24-combination tree of cv_results CSVs into ``root``."""
    per_species_dir = root / "per_species"
    per_species_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for sp, mode, variant in cm.COMBINATIONS:
        sub = per_species_dir / f"{sp}_{mode}_{variant}"
        sub.mkdir(exist_ok=True)

        rows = []
        for fold in range(1, 13):  # 12 folds
            # RF AUC slightly higher than LR on average (consistent positive ΔAUC).
            lr_auc = float(rng.uniform(0.7, 0.9))
            rf_auc = lr_auc + float(rng.uniform(-0.01, 0.05))
            lr_ap = float(rng.uniform(0.2, 0.5))
            rf_ap = lr_ap + float(rng.uniform(-0.02, 0.10))

            r_lr = {"fold": fold, "model": "logreg",
                    "auc": lr_auc, "precision": 0.3, "recall": 0.6, "f1": 0.4, "accuracy": 0.8}
            r_rf = {"fold": fold, "model": "rf",
                    "auc": rf_auc, "precision": 0.7, "recall": 0.3, "f1": 0.4, "accuracy": 0.9}
            if with_ap:
                r_lr["average_precision"] = lr_ap
                r_rf["average_precision"] = rf_ap
            rows.extend([r_lr, r_rf])

        pd.DataFrame(rows).to_csv(sub / f"cv_results_{sp}.csv", index=False)
    return per_species_dir


def test_compute_metric_all_auc_full(tmp_path):
    """compute_metric_all('auc') matches the AUC behaviour of compute_all."""
    per_species_dir = _write_synthetic_sweep(tmp_path, with_ap=True)
    rows_metric, binom_metric = cm.compute_metric_all(per_species_dir, "auc")
    rows_legacy, binom_legacy = cm.compute_all(per_species_dir)

    # Same row count, same combinations, same paired stats.
    assert len(rows_metric) == 24
    assert len(rows_legacy) == 24
    for r_m, r_l in zip(rows_metric, rows_legacy):
        assert r_m["species"] == r_l["species"]
        assert r_m["mode"] == r_l["mode"]
        assert r_m["variant"] == r_l["variant"]
        assert r_m["mean_delta"] == pytest.approx(r_l["mean_delta"])
        assert r_m["sd_delta"] == pytest.approx(r_l["sd_delta"])
        assert r_m["wilcoxon_p"] == pytest.approx(r_l["wilcoxon_p"])
        assert r_m["bonf_p"] == pytest.approx(r_l["bonf_p"])
        assert r_m["fdr_p"] == pytest.approx(r_l["fdr_p"])

    # Same binomial results.
    for key in binom_metric:
        assert binom_metric[key]["k"] == binom_legacy[key]["k"]
        assert binom_metric[key]["n"] == binom_legacy[key]["n"]
        assert binom_metric[key]["one_sided_p"] == pytest.approx(binom_legacy[key]["one_sided_p"])


def test_compute_metric_all_ap_present(tmp_path):
    """compute_metric_all('average_precision') populates AP stats for all 24."""
    per_species_dir = _write_synthetic_sweep(tmp_path, with_ap=True)
    rows, binom = cm.compute_metric_all(per_species_dir, "average_precision")

    assert len(rows) == 24
    for r in rows:
        assert "mean_delta" in r, f"missing mean_delta for {r}"
        assert "wilcoxon_p" in r
        assert "bonf_p" in r
        assert "fdr_p" in r
        assert "rf_wins" in r
    assert binom["all_24"]["n"] == 24


def test_compute_metric_all_skips_missing_metric(tmp_path):
    """If average_precision column is absent everywhere, AP rows have no stats."""
    per_species_dir = _write_synthetic_sweep(tmp_path, with_ap=False)
    rows, binom = cm.compute_metric_all(per_species_dir, "average_precision")

    assert len(rows) == 24
    for r in rows:
        # Only species/mode/variant — no stats.
        assert "mean_delta" not in r
        assert "rf_wins" not in r
    assert binom["all_24"]["n"] == 0  # no rows contribute


def test_compute_metric_all_bonferroni_within_metric_family(tmp_path):
    """Bonferroni applies within metric (K=24), not jointly with AUC."""
    per_species_dir = _write_synthetic_sweep(tmp_path, with_ap=True)
    rows, _ = cm.compute_metric_all(per_species_dir, "average_precision")
    for r in rows:
        # Bonferroni-corrected p should equal min(raw * 24, 1.0) within float tolerance.
        expected = min(r["wilcoxon_p"] * 24, 1.0)
        assert r["bonf_p"] == pytest.approx(expected, abs=1e-9)
