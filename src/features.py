"""Feature engineering — WVC pipeline.

Live exports:
    HUNTING_PERIODS, RUT_PERIODS  (constants)
    month_overlap_fraction        (helper)
    build_hunting_features        (cell 12)
    build_rut_features            (cell 12)
    build_lagged_light            (cell 7  — Phase 5 row 5)
    build_lagged_species          (cell 8  — Phase 5 row 6)
    add_cyclical_month            (cell 12 — Phase 5 row 10)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import geopandas as gpd

from src.config import SPECIES_MAP


def build_lagged_light(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-cell-month lagged light-condition shares (cell 7).

    Time-of-day classification: 5–7 = dawn, 8–16 = day, 17–20 = dusk,
    else night. Per (cell_id, period_start) the function computes the
    share of each light condition, then lags one month per cell.
    Returns columns: cell_id, period_start, dawn_lag1, day_lag1,
    dusk_lag1, night_lag1.
    """
    df = joined[["cell_id", "datetime"]].copy()
    df["cell_id"] = df["cell_id"].astype(int)
    df["period_start"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    df["hour"] = df["datetime"].dt.hour

    conditions = [
        df["hour"].between(5, 7),
        df["hour"].between(8, 16),
        df["hour"].between(17, 20),
    ]
    df["light_condition"] = np.select(conditions, ["dawn", "day", "dusk"], default="night")

    light_counts = (
        df.groupby(["cell_id", "period_start", "light_condition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    light_cols = [c for c in light_counts.columns if c not in ["cell_id", "period_start"]]
    row_sums = light_counts[light_cols].sum(axis=1)
    light_counts[light_cols] = light_counts[light_cols].div(
        row_sums.where(row_sums > 0, 1), axis=0
    )

    light_counts = light_counts.sort_values(["cell_id", "period_start"])
    for col in light_cols:
        light_counts[f"{col}_lag1"] = light_counts.groupby("cell_id")[col].shift(1)

    lag_cols = [f"{c}_lag1" for c in light_cols]
    return light_counts[["cell_id", "period_start"] + lag_cols].fillna(0)


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Append `month`, `month_sin`, `month_cos` columns derived from `period_start`.

    Mutates and returns the same dataframe (in place; matches cell 12 of
    test.ipynb). Cyclical encoding maps month 1..12 onto the unit circle
    so January is adjacent to December.
    """
    df["month"] = df["period_start"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def build_lagged_species(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-cell-month lagged collision counts for the four focal species (cell 8).

    Filters to moose, roe_deer, wild_boar, fallow_deer (Swedish labels
    translated via SPECIES_MAP), pivots to per-species counts, and lags
    one month per cell. Returns columns: cell_id, period_start,
    {species}_lag1 for each focal species.
    """
    relevant = ["moose", "roe_deer", "wild_boar", "fallow_deer"]

    df = joined[["cell_id", "datetime", "species"]].copy()
    df["cell_id"] = df["cell_id"].astype(int)
    df["period_start"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    df["species"] = (
        df["species"].astype(str).str.strip().str.lower().replace(SPECIES_MAP)
    )
    df = df[df["species"].isin(relevant)]

    species_counts = (
        df.groupby(["cell_id", "period_start", "species"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(["cell_id", "period_start"])
    )

    species_cols = [c for c in species_counts.columns if c not in ["cell_id", "period_start"]]
    for col in species_cols:
        species_counts[f"{col}_lag1"] = species_counts.groupby("cell_id")[col].shift(1)

    lag_cols = [f"{c}_lag1" for c in species_cols]
    return species_counts[["cell_id", "period_start"] + lag_cols].fillna(0)


HUNTING_PERIODS = {
    # Länsstyrelsen Stockholm: älgskötsel/licensområde
    "moose": [
        ("10-08", "01-31", 1.0),   # all allowed moose hunting
    ],

    # Naturvårdsverket: vildsvin except sow with small striped/brown piglets
    # + yearling wild boar all year
    "wild_boar": [
        ("04-01", "01-31", 1.0),   # general hunting
        ("02-01", "03-31", 0.5),   # yearlings only / reduced proxy
    ],

    # Naturvårdsverket: rådjur
    "roe_deer": [
        ("10-01", "01-31", 1.0),   # all roe deer
        ("08-16", "09-30", 0.5),   # horn-bearing males only
        ("05-01", "06-15", 0.5),   # horn-bearing males only
        ("09-01", "09-30", 0.5),   # kids only, overlaps with horn-bearing period
    ],

    # Naturvårdsverket: dovhjort
    "fallow_deer": [
        ("10-01", "10-20", 1.0),   # all animals
        ("11-16", "02-28", 1.0),   # all animals
        ("09-01", "09-30", 0.5),   # horn-bearing males + calves only
        ("10-21", "11-15", 0.5),   # hind + calf only
        ("03-01", "03-31", 0.5),   # hind + calf only
    ],
}

RUT_PERIODS = {
    "moose": [
        ("09-15", "10-15"),   # peak late Sept–early Oct
    ],
    "roe_deer": [
        ("07-15", "08-15"),   # summer rut
    ],
    "wild_boar": [
        ("11-01", "02-28"),   # extended winter rut
    ],
    "fallow_deer": [
        ("10-01", "11-15"),   # autumn rut
    ],
}


def month_overlap_fraction(period_start, start_str, end_str):
    year = period_start.year

    start = pd.Timestamp(f"{year}-{start_str}")
    end   = pd.Timestamp(f"{year}-{end_str}")

    # Handle wrap-around (e.g. Oct → Jan)
    if end < start:
        if period_start.month >= start.month:
            end = pd.Timestamp(f"{year+1}-{end_str}")
        else:
            start = pd.Timestamp(f"{year-1}-{start_str}")

    month_start = period_start
    month_end   = period_start + pd.offsets.MonthEnd(1)

    overlap_start = max(start, month_start)
    overlap_end   = min(end, month_end)

    if overlap_start > overlap_end:
        return 0.0

    overlap_days = (overlap_end - overlap_start).days + 1
    month_days   = (month_end - month_start).days + 1

    return overlap_days / month_days

def build_hunting_features(df):
    df = df.copy()

    for species, periods in HUNTING_PERIODS.items():
        values = []

        for _, row in df.iterrows():
            period = row["period_start"]

            frac = 0
            for start, end, weight in periods:
                overlap = month_overlap_fraction(period, start, end)
                frac += overlap * weight

            values.append(min(frac, 1.0))

        df[f"{species}_hunting_frac"] = values

    return df

def build_rut_features(df):
    df = df.copy()

    for species, periods in RUT_PERIODS.items():
        values = []

        for _, row in df.iterrows():
            period = row["period_start"]

            frac = 0
            for start, end in periods:
                frac += month_overlap_fraction(period, start, end)

            values.append(min(frac, 1.0))

        df[f"{species}_rut_frac"] = values

    return df

