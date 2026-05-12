#!/usr/bin/env python3
"""Print collision counts from raw NVR data by species and accident type.

Reads every yearly file in ``data/Collisions/`` (semicolon-separated,
latin-1 encoded), restricts to the four thesis species (moose, roe deer,
wild boar, fallow deer), and prints summary tallies. CLI flags narrow
the slice further: a single year, a single species, or both.

Run from code/:

    .venv/bin/python scripts/count_animals.py
    .venv/bin/python scripts/count_animals.py --year 2024
    .venv/bin/python scripts/count_animals.py --species moose
    .venv/bin/python scripts/count_animals.py --by-year
    .venv/bin/python scripts/count_animals.py --by-month
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import COLLISION_INFRASTRUCTURE_MAP, SPECIES_MAP

COLLISIONS_DIR = REPO_ROOT / "data" / "Collisions"


def load_all(year: int | None) -> pd.DataFrame:
    """Concatenate all yearly NVR CSVs, optionally filtered to one year."""
    files = sorted(COLLISIONS_DIR.glob("R*data *.csv"))
    if year is not None:
        files = [f for f in files if str(year) in f.name]
    if not files:
        raise FileNotFoundError(
            f"No collision CSVs found in {COLLISIONS_DIR} for year={year}"
        )
    frames = [
        pd.read_csv(f, sep=";", encoding="latin1", low_memory=False) for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df["species_en"] = df["Viltslag"].str.lower().map(SPECIES_MAP)
    df = df[df["species_en"].notna()].copy()
    df["infra_en"] = df["Typ av olycka"].map(COLLISION_INFRASTRUCTURE_MAP).fillna(
        df["Typ av olycka"]
    )
    dt = pd.to_datetime(df["Datum"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    return df


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_counts(series: pd.Series) -> None:
    for label, count in series.items():
        print(f"  {label:<20} {count:>10,}")
    print(f"  {'TOTAL':<20} {int(series.sum()):>10,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="Filter to one year")
    parser.add_argument(
        "--species",
        choices=list(SPECIES_MAP.values()),
        default=None,
        help="Filter to one species (English label)",
    )
    parser.add_argument(
        "--by-year",
        action="store_true",
        help="Also print a per-year breakdown",
    )
    parser.add_argument(
        "--by-month",
        action="store_true",
        help="Also print a per-month breakdown (calendar month, summed across years in scope)",
    )
    args = parser.parse_args()

    df = load_all(args.year)
    if args.year is not None:
        df = df[df["year"] == args.year]
    if args.species:
        df = df[df["species_en"] == args.species]

    year_range = (
        f"{int(df['year'].min())}–{int(df['year'].max())}"
        if df["year"].notna().any()
        else "unknown"
    )
    scope = []
    if args.year:
        scope.append(f"year={args.year}")
    if args.species:
        scope.append(f"species={args.species}")
    scope_str = ", ".join(scope) if scope else "all data"

    print(f"NVR collision records ({scope_str}, years {year_range})")
    print(f"  rows: {len(df):,}")

    print_section("By species")
    print_counts(df["species_en"].value_counts())

    print_section("By accident type")
    print_counts(df["infra_en"].value_counts())

    print_section("Species × accident type")
    crosstab = pd.crosstab(df["species_en"], df["infra_en"], margins=True, margins_name="TOTAL")
    print(crosstab.to_string())

    if args.by_year:
        print_section("By year × species")
        by_year = pd.crosstab(df["year"], df["species_en"], margins=True, margins_name="TOTAL")
        print(by_year.to_string())

    if args.by_month:
        print_section("By month × species")
        by_month = pd.crosstab(df["month"], df["species_en"], margins=True, margins_name="TOTAL")
        print(by_month.to_string())

        print_section("By month × accident type")
        by_month_infra = pd.crosstab(df["month"], df["infra_en"], margins=True, margins_name="TOTAL")
        print(by_month_infra.to_string())


if __name__ == "__main__":
    main()
