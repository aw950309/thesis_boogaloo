"""Spatial grid construction and per-cell-month panel.

Public API:
    create_grid(gdf, cell_size)
    spatial_join_points_to_grid(gdf_points, grid)
    compute_grid_risk(gdf_joined, threshold_quantile=0.75)
    build_cell_month_panel(gdf_points, cell_size, threshold_quantile=0.75)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def create_grid(gdf: gpd.GeoDataFrame, cell_size: int = 5000) -> gpd.GeoDataFrame:
    """Build a square grid over the bounding box of ``gdf``.

    ``gdf`` must be in a projected CRS (metres). ``cell_size`` is the
    grid resolution in metres.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    grid_cells = [box(x, y, x + cell_size, y + cell_size) for x in x_coords for y in y_coords]

    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=gdf.crs)
    grid["cell_id"] = range(len(grid))
    return grid


def spatial_join_points_to_grid(
    gdf_points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Assign each collision point to its grid cell via a left within-join."""
    return gpd.sjoin(gdf_points, grid, how="left", predicate="within")


def compute_grid_risk(
    gdf_joined: gpd.GeoDataFrame, threshold_quantile: float = 0.75
) -> pd.DataFrame:
    """Aggregate collision counts per cell and label high-risk cells.

    Threshold is the ``threshold_quantile`` quantile of per-cell counts.
    Default 0.75 (per AD-03 refinement). Set to 0.5 to recover the
    legacy median behaviour.
    """
    cell_counts = (
        gdf_joined.groupby("cell_id")
        .size()
        .reset_index(name="collision_count")
    )

    threshold = cell_counts["collision_count"].quantile(threshold_quantile)
    cell_counts["risk"] = (cell_counts["collision_count"] >= threshold).astype(int)
    return cell_counts


def build_cell_month_panel(
    gdf_points: gpd.GeoDataFrame,
    cell_size: int = 10000,
    threshold_quantile: float = 0.75,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Build the (cell × month) panel used by the modelling pipeline.

    Mirrors cell 2 of test.ipynb. Returns ``(grid, joined, cell_month)``:

    - ``grid`` — the full grid GeoDataFrame.
    - ``joined`` — points spatially joined to grid cells, with cleaned
      ``cell_id`` (int) and ``period_start`` (month start) columns.
    - ``cell_month`` — full grid × month panel with ``collision_count``
      (filled with 0 where no observations) and a binary ``risk`` label
      from the ``threshold_quantile`` of *non-zero* counts.
    """
    grid = create_grid(gdf_points, cell_size=cell_size)
    joined = spatial_join_points_to_grid(gdf_points, grid)
    joined = joined.dropna(subset=["cell_id"]).copy()
    joined["cell_id"] = joined["cell_id"].astype(int)
    joined["period_start"] = joined["datetime"].dt.to_period("M").dt.to_timestamp()

    observed = (
        joined.groupby(["cell_id", "period_start"])
        .size()
        .reset_index(name="collision_count")
    )

    periods = gdf_points["datetime"].dt.to_period("M")
    min_month = periods.min().to_timestamp()
    max_month = periods.max().to_timestamp()
    all_months = pd.date_range(start=min_month, end=max_month, freq="MS")

    full_index = pd.MultiIndex.from_product(
        [grid["cell_id"].astype(int).unique(), all_months],
        names=["cell_id", "period_start"],
    )
    cell_month = (
        full_index.to_frame(index=False)
        .merge(observed, on=["cell_id", "period_start"], how="left")
    )
    cell_month["collision_count"] = cell_month["collision_count"].fillna(0).astype(int)

    nonzero = cell_month.loc[cell_month["collision_count"] > 0, "collision_count"]
    if nonzero.empty:
        raise ValueError("No non-zero collision counts; cannot define risk threshold.")
    threshold = nonzero.quantile(threshold_quantile)
    cell_month["risk"] = (cell_month["collision_count"] >= threshold).astype(int)

    return grid, joined, cell_month
