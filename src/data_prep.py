"""Data preparation — NVR collision CSV loaders.

Public API:
    load_collision_data(path)
    load_collision_data_multi_year(directory, year_range=None)

The NVR CSVs are semicolon-separated, latin-1 encoded, with Swedish
decimal commas in the lat/lon columns. The loader normalises all of
that and projects EPSG:4326 → EPSG:3006 (SWEREF99 TM).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd

from src.config import NVR_COLUMN_RENAME, NVR_SOURCE_CRS, NVR_TARGET_CRS


# Plausible Sweden WGS84 bounding box; any row outside is dropped as a
# coordinate-entry error rather than a real observation.
_SWEDEN_LAT_MIN, _SWEDEN_LAT_MAX = 55, 70
_SWEDEN_LON_MIN, _SWEDEN_LON_MAX = 10, 25


def load_collision_data(path: Path | str) -> gpd.GeoDataFrame:
    """Load one yearly NVR CSV and return a projected GeoDataFrame.

    Steps: read with `;` separator and latin-1 encoding, rename columns
    per NVR_COLUMN_RENAME, fix Swedish decimal commas in lat/lon, parse
    datetimes, drop rows missing datetime/lat/lon, filter to Sweden bbox,
    build geometry, project to EPSG:3006.
    """
    df = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)
    df = df.rename(columns=NVR_COLUMN_RENAME)

    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["datetime", "lat", "lon"])

    df = df[
        df["lat"].between(_SWEDEN_LAT_MIN, _SWEDEN_LAT_MAX)
        & df["lon"].between(_SWEDEN_LON_MIN, _SWEDEN_LON_MAX)
    ]

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=NVR_SOURCE_CRS,
    ).to_crs(NVR_TARGET_CRS)


def load_collision_data_multi_year(
    directory: Path | str,
    year_range: tuple[int | None, int | None] | None = None,
) -> gpd.GeoDataFrame:
    """Load every NVR CSV in a directory and concatenate.

    `year_range` is an inclusive (lower, upper) bound on the parsed
    `datetime`'s year; either side can be None for an open bound. Default
    `None` applies no year filter. The notebook's de-facto behaviour is
    `(None, 2025)` (drops the incomplete 2026 partial).
    """
    folder = Path(directory)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {folder}")

    parts = [load_collision_data(f) for f in files]
    gdf = pd.concat(parts, ignore_index=True)

    if year_range is not None:
        lo, hi = year_range
        years = gdf["datetime"].dt.year
        if lo is not None:
            gdf = gdf[years >= lo]
        if hi is not None:
            gdf = gdf[years <= hi]

    return gdf
