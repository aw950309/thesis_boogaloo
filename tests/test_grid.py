"""Smoke tests for src.grid — Phase 5 Row 3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from src import grid as grid_mod


def _toy_points_gdf() -> gpd.GeoDataFrame:
    """A handful of WGS84 points around Sweden, projected to EPSG:3006."""
    rows = [
        ("2020-01-01", 59.33, 18.07),
        ("2020-02-01", 59.34, 18.08),
        ("2020-03-01", 55.70, 13.19),
        ("2020-04-01", 57.71, 11.97),
    ]
    df = pd.DataFrame(rows, columns=["datetime", "lat", "lon"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lat, lon in zip(df["lat"], df["lon"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3006")


def test_create_grid_covers_bounds() -> None:
    gdf = _toy_points_gdf()
    grid = grid_mod.create_grid(gdf, cell_size=10000)
    assert "cell_id" in grid.columns
    assert len(grid) > 0
    minx, miny, maxx, maxy = gdf.total_bounds
    gminx, gminy, gmaxx, gmaxy = grid.total_bounds
    assert gminx <= minx and gminy <= miny
    assert gmaxx >= minx and gmaxy >= miny


def test_compute_grid_risk_threshold_quantile() -> None:
    """compute_grid_risk parameterises the threshold (AD-03)."""
    joined = pd.DataFrame({"cell_id": [0] * 10 + [1] * 5 + [2] * 1})
    out_75 = grid_mod.compute_grid_risk(joined, threshold_quantile=0.75)
    # Per-cell counts: 0->10, 1->5, 2->1; quantile 0.75 = 7.5 → cells with count>=7.5 are risky
    assert out_75.set_index("cell_id")["risk"].to_dict() == {0: 1, 1: 0, 2: 0}

    out_50 = grid_mod.compute_grid_risk(joined, threshold_quantile=0.5)
    # Median = 5 → cells with count>=5 are risky
    assert out_50.set_index("cell_id")["risk"].to_dict() == {0: 1, 1: 1, 2: 0}


def test_build_cell_month_panel_full_panel_shape() -> None:
    gdf = _toy_points_gdf()
    grid, joined, cell_month = grid_mod.build_cell_month_panel(gdf, cell_size=10000)
    # Full panel = unique cells × unique months
    n_cells = grid["cell_id"].nunique()
    n_months = pd.date_range(
        gdf["datetime"].min().to_period("M").to_timestamp(),
        gdf["datetime"].max().to_period("M").to_timestamp(),
        freq="MS",
    ).size
    assert len(cell_month) == n_cells * n_months
    assert set(cell_month.columns) >= {"cell_id", "period_start", "collision_count", "risk"}
    assert cell_month["collision_count"].min() == 0
    assert cell_month["risk"].isin({0, 1}).all()
