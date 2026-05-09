#!/usr/bin/env python3
"""Internal audit: verify compare_models.py output against A2 inline computations.

Cross-checks every computed statistic against the values reported inline in:
  notes/notes_paper/T0_audit_outputs/A2_signal_vs_noise_2026-05-05.md

Not for external distribution. Not referenced by compare_models.py.

Outputs:
  scripts/audit_verification.md   — durable record of match/diverge per field

Run from code/ directory:
  .venv/bin/python scripts/_internal_audit_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import core computation from compare_models
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from compare_models import COMBINATIONS, compute_all  # noqa: E402

# ---------------------------------------------------------------------------
# A2 binomial label map
# ---------------------------------------------------------------------------

BINOMIAL_LABELS = {
    "Headline (AUC, all 24)":  "all_24",
    "Mode-stratified default": "default",
    "Mode-stratified road":    "road",
    "Mode-stratified rail":    "rail",
    "Default + road combined": "default_road",
}


# ---------------------------------------------------------------------------
# Parse A2 markdown
# ---------------------------------------------------------------------------

def _strip(cell: str) -> str:
    return cell.replace("**", "").replace("*", "").replace("−", "-").strip()


def parse_a2_markdown(path: Path) -> tuple[dict, dict, dict]:
    text  = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    a2_table: dict[tuple[str, str, str], dict] = {}
    a2_binomial: dict[str, dict] = {}
    a2_counts: dict[str, int] = {}

    # per-combination table
    in_main = False
    for line in lines:
        s = line.strip()
        if "| species | mode | variant |" in s:
            in_main = True
            continue
        if in_main:
            if s.startswith("|---"):
                continue
            if not s.startswith("|"):
                in_main = False
                continue
            cells = [_strip(c) for c in s.split("|")[1:-1]]
            if len(cells) < 11:
                continue
            species, mode, variant = cells[0], cells[1], cells[2]
            if species not in {"roe_deer", "moose", "wild_boar", "fallow_deer"}:
                continue
            ci_raw   = cells[6].replace("[", "").replace("]", "")
            ci_parts = [p.strip() for p in ci_raw.split(",")]
            a2_table[(species, mode, variant)] = {
                "n":          int(cells[3]),
                "mean":       float(cells[4]),
                "sd":         float(cells[5]),
                "ci_lo":      float(ci_parts[0]),
                "ci_hi":      float(ci_parts[1]),
                "wilcoxon_p": float(cells[8]),
                "bonf_p":     float(cells[9]),
                "fdr_p":      float(cells[10]),
            }

    # binomial table
    in_binom = False
    for line in lines:
        s = line.strip()
        if "| comparison | numerator/denominator |" in s:
            in_binom = True
            continue
        if in_binom:
            if s.startswith("|---"):
                continue
            if not s.startswith("|"):
                in_binom = False
                continue
            cells = [_strip(c) for c in s.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            key = BINOMIAL_LABELS.get(cells[0])
            if key is None:
                continue
            k_str, n_str = cells[1].split("/")
            a2_binomial[key] = {
                "k":           int(k_str),
                "n":           int(n_str),
                "one_sided_p": float(cells[2]),
                "two_sided_p": float(cells[3]),
            }

    # aggregate counts from "Net-net." sentence
    m = re.search(
        r"(\d+) are uncorrected p<0\.05.*?(\d+) survive Bonferroni.*?(\d+) survive FDR",
        text,
    )
    if m:
        a2_counts = {
            "uncorrected_p05": int(m.group(1)),
            "bonferroni_sig":  int(m.group(2)),
            "fdr_sig":         int(m.group(3)),
        }

    return a2_table, a2_binomial, a2_counts


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _check(computed: float, reported: float, tol: float, label: str) -> tuple[str, bool]:
    diff   = abs(computed - reported)
    passed = diff <= tol
    status = "MATCH  " if passed else "DIVERGE"
    line   = (f"{status}  computed={computed:.6f}  reported={reported:.4f}"
              f"  |diff|={diff:.6f}  {label}")
    return line, passed


# ---------------------------------------------------------------------------
# Build audit markdown
# ---------------------------------------------------------------------------

def build_audit_markdown(
    rows: list[dict],
    binomial_results: dict,
    a2_table: dict,
    a2_binomial: dict,
    a2_counts: dict,
) -> str:
    TOL_P  = 0.005
    TOL_CI = 0.010
    TOL_SD = 0.001

    n_uncorr = sum(1 for r in rows if r["wilcoxon_p"] < 0.05)
    n_bonf   = sum(1 for r in rows if r["bonf_p"]    < 0.05)
    n_fdr    = sum(1 for r in rows if r["fdr_p"]     < 0.05)

    lines = [
        "# Audit verification — computed vs A2-reported values\n",
        "Internal cross-check. Not for external distribution.\n",
        "Source: `notes/notes_paper/T0_audit_outputs/A2_signal_vs_noise_2026-05-05.md`\n",
        "Tolerances: p-values |diff| < 0.005 · CI bounds < 0.010 · SD < 0.001 · n exact\n",
        "## Aggregate counts\n",
    ]

    for label, computed, reported in [
        ("uncorrected p<0.05", n_uncorr, a2_counts.get("uncorrected_p05", "?")),
        ("bonferroni_sig",     n_bonf,   a2_counts.get("bonferroni_sig",  "?")),
        ("fdr_sig",            n_fdr,    a2_counts.get("fdr_sig",         "?")),
    ]:
        ok = "MATCH" if computed == reported else "DIVERGE"
        lines.append(f"- {label}: computed={computed}  reported={reported}  **{ok}**")

    lines += ["", "## Binomial tests\n"]
    for key, br in binomial_results.items():
        ref    = a2_binomial.get(key, {})
        ok_k   = "MATCH" if br["k"] == ref.get("k") else "DIVERGE"
        ok_one = "MATCH" if abs(br["one_sided_p"] - ref.get("one_sided_p", 99)) <= TOL_P else "DIVERGE"
        ok_two = "MATCH" if abs(br["two_sided_p"] - ref.get("two_sided_p", 99)) <= TOL_P else "DIVERGE"
        lines.append(
            f"- **{key}**: k={br['k']}/{br['n']} {ok_k} | "
            f"one-sided={br['one_sided_p']:.4f} {ok_one} "
            f"(A2: {ref.get('one_sided_p', '?'):.4f}) | "
            f"two-sided={br['two_sided_p']:.4f} {ok_two} "
            f"(A2: {ref.get('two_sided_p', '?'):.4f})"
        )

    lines += ["", "## Per-combination row checks\n"]
    total_diverge = 0
    for r in rows:
        key = (r["species"], r["mode"], r["variant"])
        ref = a2_table.get(key, {})
        checks = [
            _check(r["n"],          ref.get("n",          -1), 0,      "n"),
            _check(r["mean_delta"], ref.get("mean",        99), 0.001,  "mean_delta"),
            _check(r["sd_delta"],   ref.get("sd",          99), TOL_SD, "sd"),
            _check(r["ci_lo"],      ref.get("ci_lo",       99), TOL_CI, "ci_lo"),
            _check(r["ci_hi"],      ref.get("ci_hi",       99), TOL_CI, "ci_hi"),
            _check(r["wilcoxon_p"], ref.get("wilcoxon_p",  99), TOL_P,  "wilcoxon_p"),
            _check(r["bonf_p"],     ref.get("bonf_p",      99), TOL_P,  "bonf_p"),
            _check(r["fdr_p"],      ref.get("fdr_p",       99), TOL_P,  "fdr_p"),
        ]
        row_diverges = sum(1 for _, ok in checks if not ok)
        total_diverge += row_diverges > 0
        lines.append(
            f"\n### {r['species']} / {r['mode']} / {r['variant']}"
            + (" ← DIVERGE" if row_diverges else "")
        )
        lines += [f"    {line}" for line, _ in checks]

    lines += [
        "",
        "## Result\n",
        f"Per-row divergences: **{total_diverge} of 24** combinations "
        "had at least one diverging field.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here            = Path(__file__).parent.parent
    per_species_dir = here / "outputs" / "per_species"
    notes_dir       = here.parent / "notes" / "notes_paper" / "T0_audit_outputs"
    a2_path         = notes_dir / "A2_signal_vs_noise_2026-05-05.md"

    print(f"Parsing A2 reference: {a2_path.name}")
    a2_table, a2_binomial, a2_counts = parse_a2_markdown(a2_path)
    print(f"  {len(a2_table)} combination rows  "
          f"{len(a2_binomial)} binomial entries  counts={a2_counts}\n")

    print("Computing paired statistics ...")
    rows, binomial_results = compute_all(per_species_dir)
    print()

    # Console summary
    n_uncorr = sum(1 for r in rows if r["wilcoxon_p"] < 0.05)
    n_bonf   = sum(1 for r in rows if r["bonf_p"]    < 0.05)
    n_fdr    = sum(1 for r in rows if r["fdr_p"]     < 0.05)

    print("Aggregate counts (computed vs A2):")
    for label, c, rep in [
        ("uncorrected p<0.05", n_uncorr, a2_counts.get("uncorrected_p05")),
        ("bonferroni_sig",     n_bonf,   a2_counts.get("bonferroni_sig")),
        ("fdr_sig",            n_fdr,    a2_counts.get("fdr_sig")),
    ]:
        ok = "MATCH" if c == rep else "DIVERGE"
        print(f"  {label:22s}: {c}  (A2: {rep})  {ok}")
    print()

    TOL_P = 0.005
    print("Binomial tests:")
    for key, br in binomial_results.items():
        ref    = a2_binomial.get(key, {})
        ok_k   = "MATCH  " if br["k"] == ref.get("k") else "DIVERGE"
        ok_one = "MATCH  " if abs(br["one_sided_p"] - ref.get("one_sided_p", 99)) <= TOL_P else "DIVERGE"
        ok_two = "MATCH  " if abs(br["two_sided_p"] - ref.get("two_sided_p", 99)) <= TOL_P else "DIVERGE"
        print(f"  {key:15s}  k={br['k']}/{br['n']} {ok_k}"
              f"  one={br['one_sided_p']:.4f} {ok_one}"
              f"  two={br['two_sided_p']:.4f} {ok_two}")
    print()

    # Write audit markdown
    audit_path = Path(__file__).parent / "audit_verification.md"
    audit_path.write_text(
        build_audit_markdown(rows, binomial_results, a2_table, a2_binomial, a2_counts)
    )
    print(f"Audit written: scripts/audit_verification.md")


if __name__ == "__main__":
    main()
