"""Spatial grid construction and per-cell-month panel.

Public API:
    create_grid(gdf, cell_size)
    spatial_join_points_to_grid(gdf_points, grid)
    compute_grid_risk(gdf_joined, threshold_quantile=0.75)
    build_cell_month_panel(gdf_points, cell_size, threshold_quantile=0.75)
    build_species_panel(joined, grid, species_name, threshold_quantile=0.75)
    build_species_model_df(species_name, joined, grid, model_df)
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


def build_species_panel(
    joined: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    species_name: str,
    threshold_quantile: float = 0.75,
    collision_infrastructure_filter: str | None = None,
) -> pd.DataFrame:
    """Per-species cell-month panel with a per-species risk label.

    Filters ``joined`` to ``species_name``, builds the full grid × month panel,
    then applies ``threshold_quantile`` of non-zero counts as the risk threshold.
    Month range is taken from ``joined`` (consistent with build_cell_month_panel).

    ``collision_infrastructure_filter`` (T10): if set to "road" or "rail", further filters
    ``joined`` to collisions of that infrastructure type via the ``collision_infrastructure``
    column. Default ``None`` includes both. Used by per-species mode runs.

    Returns a DataFrame with columns: cell_id, period_start, collision_count, risk.
    """
    from src.config import SPECIES_MAP

    cols = ["cell_id", "datetime", "species"]
    if collision_infrastructure_filter is not None:
        if "collision_infrastructure" not in joined.columns:
            raise ValueError(
                "collision_infrastructure_filter requested but joined frame has no 'collision_infrastructure' column. "
                "Ensure data_prep.load_collision_data preserved the column."
            )
        cols = cols + ["collision_infrastructure"]

    df = joined[cols].copy()
    df["species"] = df["species"].astype(str).str.strip().str.lower().replace(SPECIES_MAP)
    df = df[df["species"] == species_name].copy()

    if collision_infrastructure_filter is not None:
        df = df[df["collision_infrastructure"] == collision_infrastructure_filter].copy()

    df["cell_id"] = df["cell_id"].astype(int)
    df["period_start"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

    observed = (
        df.groupby(["cell_id", "period_start"])
        .size()
        .reset_index(name="collision_count")
    )

    all_months = pd.date_range(
        start=joined["datetime"].dt.to_period("M").min().to_timestamp(),
        end=joined["datetime"].dt.to_period("M").max().to_timestamp(),
        freq="MS",
    )

    full_index = pd.MultiIndex.from_product(
        [grid["cell_id"].astype(int).unique(), all_months],
        names=["cell_id", "period_start"],
    )
    species_cm = (
        full_index.to_frame(index=False)
        .merge(observed, on=["cell_id", "period_start"], how="left")
    )
    species_cm["collision_count"] = species_cm["collision_count"].fillna(0).astype(int)

    nonzero = species_cm.loc[species_cm["collision_count"] > 0, "collision_count"]
    if nonzero.empty:
        raise ValueError(f"No non-zero collisions for species '{species_name}'.")
    threshold = nonzero.quantile(threshold_quantile)
    species_cm["risk"] = (species_cm["collision_count"] >= threshold).astype(int)

    return species_cm


def build_species_model_df(
    species_name: str,
    joined: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    model_df: pd.DataFrame,
    collision_infrastructure_filter: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Swap the pooled risk label in model_df for a per-species one.

    All feature columns (infrastructure, weather, lags) are reused from
    ``model_df`` — no re-loading. Returns ``(df, features)`` where
    ``features = get_species_features(species_name)`` (22 features: 19 base
    environmental + 3 species-specific lag/hunting/rut).

    ``collision_infrastructure_filter`` (T10): passed through to ``build_species_panel`` to
    restrict the target collision count to a single infrastructure type
    ("road" or "rail"). Default ``None`` counts all collisions.
    """
    from src.config import get_species_features

    species_cm = build_species_panel(
        joined, grid, species_name, collision_infrastructure_filter=collision_infrastructure_filter
    )

    feature_cols = [c for c in model_df.columns if c not in ["risk", "collision_count"]]
    df = model_df[feature_cols].merge(
        species_cm[["cell_id", "period_start", "collision_count", "risk"]],
        on=["cell_id", "period_start"],
        how="inner",
    )

    species_f = get_species_features(species_name)
    missing = [f for f in species_f if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features for '{species_name}': {missing}")

    df = df.dropna(subset=species_f).copy()
    return df, species_f
