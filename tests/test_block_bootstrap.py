"""Unit tests for scripts/block_bootstrap.py — the per-metric paired-diff loader."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is not a package; mirror _internal_audit_check.py's import style.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import block_bootstrap as bb  # noqa: E402


def _write_cv(tmp_path: Path, sp: str, mode: str, variant: str, with_ap: bool = True) -> Path:
    per_species_dir = tmp_path / "per_species"
    sub = per_species_dir / f"{sp}_{mode}_{variant}"
    sub.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(1, 13):
        lr = {"fold": fold, "model": "logreg", "auc": 0.80 + 0.001 * fold}
        rf = {"fold": fold, "model": "rf",     "auc": 0.85 + 0.001 * fold}
        if with_ap:
            lr["average_precision"] = 0.40 + 0.002 * fold
            rf["average_precision"] = 0.50 + 0.002 * fold
        rows.extend([lr, rf])
    pd.DataFrame(rows).to_csv(sub / f"cv_results_{sp}.csv", index=False)
    return per_species_dir


def test_load_paired_diffs_default_metric_auc(tmp_path):
    per_species_dir = _write_cv(tmp_path, "moose", "default", "lag", with_ap=True)
    diffs = bb._load_paired_diffs(per_species_dir, "moose", "default", "lag")
    assert diffs is not None
    assert len(diffs) == 12
    # ΔAUC is constant = 0.05 by construction.
    assert all(abs(d - 0.05) < 1e-9 for d in diffs)


def test_load_paired_diffs_average_precision(tmp_path):
    per_species_dir = _write_cv(tmp_path, "moose", "default", "lag", with_ap=True)
    diffs = bb._load_paired_diffs(
        per_species_dir, "moose", "default", "lag", metric="average_precision",
    )
    assert diffs is not None
    assert len(diffs) == 12
    # ΔAP is constant = 0.10 by construction.
    assert all(abs(d - 0.10) < 1e-9 for d in diffs)


def test_load_paired_diffs_missing_metric_returns_none(tmp_path):
    per_species_dir = _write_cv(tmp_path, "moose", "default", "lag", with_ap=False)
    diffs = bb._load_paired_diffs(
        per_species_dir, "moose", "default", "lag", metric="average_precision",
    )
    assert diffs is None


def test_load_paired_diffs_missing_file_returns_none(tmp_path):
    per_species_dir = tmp_path / "per_species"
    per_species_dir.mkdir()
    diffs = bb._load_paired_diffs(per_species_dir, "moose", "default", "lag")
    assert diffs is None
