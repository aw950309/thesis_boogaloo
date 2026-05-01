"""Smoke tests for src.features lag builders — Phase 5 Rows 5, 6."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from src import features


def _toy_joined() -> gpd.GeoDataFrame:
    """Two cells, three months of observations, a mix of times and species."""
    rows = [
        # cell 0 — month 2020-01: 2 dawn, 1 day, 1 dusk; species moose, roe_deer, wild_boar, fallow_deer
        (0, "2020-01-05 06:00", "älg"),
        (0, "2020-01-15 07:00", "rådjur"),
        (0, "2020-01-20 12:00", "vildsvin"),
        (0, "2020-01-25 18:00", "dovhjort"),
        # cell 0 — month 2020-02: 2 night, 1 day; one non-focal species
        (0, "2020-02-01 02:00", "älg"),
        (0, "2020-02-15 22:00", "rådjur"),
        (0, "2020-02-20 10:00", "ren"),  # non-focal — dropped by build_lagged_species
        # cell 1 — month 2020-01: all day; one moose
        (1, "2020-01-10 10:00", "älg"),
    ]
    df = pd.DataFrame(rows, columns=["cell_id", "datetime", "species"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return gpd.GeoDataFrame(
        df, geometry=[Point(0, 0)] * len(df), crs="EPSG:3006"
    )


def test_build_lagged_light_columns_and_lag() -> None:
    joined = _toy_joined()
    out = features.build_lagged_light(joined)

    # Output schema.
    assert set(out.columns) == {"cell_id", "period_start", "dawn_lag1", "day_lag1", "dusk_lag1", "night_lag1"}

    # Cell 0, 2020-01: 4 obs with dawn=2, day=1, dusk=1, night=0 → shares 0.5/0.25/0.25/0.0.
    # The lag1 column for 2020-02 is the shifted 2020-01 values.
    cell0_feb = out[(out["cell_id"] == 0) & (out["period_start"] == pd.Timestamp("2020-02-01"))].iloc[0]
    assert cell0_feb["dawn_lag1"] == pytest.approx(0.5)
    assert cell0_feb["day_lag1"] == pytest.approx(0.25)
    assert cell0_feb["dusk_lag1"] == pytest.approx(0.25)
    assert cell0_feb["night_lag1"] == pytest.approx(0.0)

    # First-month lags filled with 0 (no prior period).
    cell0_jan = out[(out["cell_id"] == 0) & (out["period_start"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert cell0_jan[["dawn_lag1", "day_lag1", "dusk_lag1", "night_lag1"]].sum() == 0


def test_build_lagged_species_filter_and_lag() -> None:
    joined = _toy_joined()
    out = features.build_lagged_species(joined)

    # Schema: four focal-species lag columns. Order may vary.
    expected_lag_cols = {"moose_lag1", "roe_deer_lag1", "wild_boar_lag1", "fallow_deer_lag1"}
    assert expected_lag_cols.issubset(set(out.columns))

    # Cell 0, 2020-02: prior month had 1 of each focal species (the non-focal "ren" in Feb dropped).
    cell0_feb = out[(out["cell_id"] == 0) & (out["period_start"] == pd.Timestamp("2020-02-01"))].iloc[0]
    for col in expected_lag_cols:
        assert cell0_feb[col] == 1, f"{col} expected 1; got {cell0_feb[col]}"

    # Non-focal species removed entirely from the pivot.
    assert "ren_lag1" not in out.columns


def test_add_cyclical_month_unit_circle() -> None:
    """add_cyclical_month maps 1-12 onto the unit circle (sin^2 + cos^2 = 1)."""
    df = pd.DataFrame(
        {"period_start": pd.to_datetime([f"2020-{m:02d}-01" for m in range(1, 13)])}
    )
    out = features.add_cyclical_month(df)

    assert "month" in out.columns
    assert "month_sin" in out.columns
    assert "month_cos" in out.columns
    assert out["month"].tolist() == list(range(1, 13))

    # Unit-circle invariant.
    radii = out["month_sin"] ** 2 + out["month_cos"] ** 2
    assert np.allclose(radii, 1.0, atol=1e-12)

    # December should be adjacent to January (cosine close).
    cos_dec = out.loc[out["month"] == 12, "month_cos"].iloc[0]
    cos_jan = out.loc[out["month"] == 1, "month_cos"].iloc[0]
    assert abs(cos_dec - cos_jan) < 0.5  # both near sqrt(3)/2 region


def test_build_lagged_species_drops_non_focal() -> None:
    """A frame containing only non-focal species returns an empty result."""
    joined = gpd.GeoDataFrame(
        {
            "cell_id": [0, 0],
            "datetime": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "species": ["ren", "ren"],
        },
        geometry=[Point(0, 0)] * 2,
        crs="EPSG:3006",
    )
    out = features.build_lagged_species(joined)
    assert len(out) == 0
