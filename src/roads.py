from __future__ import annotations

import re
import numpy as np
import pandas as pd
import geopandas as gpd


def validate_projected_crs(gdf: gpd.GeoDataFrame, name: str = "GeoDataFrame") -> None:
    if gdf.crs is None:
        raise ValueError(f"{name} must have a CRS")

    if getattr(gdf.crs, "is_geographic", False):
        raise ValueError(
            f"{name} must use a projected CRS, not geographic coordinates. "
            f"Use EPSG:3006 for Sweden."
        )


def require_columns(df: pd.DataFrame, required: list[str], name: str = "DataFrame") -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def make_safe_column_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-zA-Z0-9_åäöÅÄÖ]", "", value)
    return value


def load_linear_layer(
    path: str,
    bbox=None,
    rows=None,
) -> gpd.GeoDataFrame:

    gdf = gpd.read_file(path, bbox=bbox, rows=rows)

    if gdf.empty:
        raise ValueError(f"Loaded layer is empty: {path}")

    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS: {path}")

    return gdf


def clean_linear_layer(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:3006",
) -> gpd.GeoDataFrame:

    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    if gdf.empty:
        raise ValueError("No valid line geometries found in layer")

    return gdf.reset_index(drop=True)


def load_linear_layer_for_study_area(
    path: str,
    gdf_points: gpd.GeoDataFrame,
    target_crs: str = "EPSG:3006",
    buffer_m: float = 0,
) -> gpd.GeoDataFrame:
    """
    Load any line-based layer clipped to the study area extent.
    """
    if gdf_points.crs is None:
        raise ValueError("gdf_points must have a CRS")

    points = gdf_points.to_crs(target_crs).copy()
    minx, miny, maxx, maxy = points.total_bounds

    bbox = (
        minx - buffer_m,
        miny - buffer_m,
        maxx + buffer_m,
        maxy + buffer_m,
    )

    gdf = load_linear_layer(path=path, bbox=bbox)
    gdf = clean_linear_layer(gdf, target_crs=target_crs)

    return gdf


def load_roads(path: str, bbox=None, rows=None) -> gpd.GeoDataFrame:
    roads = gpd.read_file(path, bbox=bbox, rows=rows)

    if roads.empty:
        raise ValueError("Loaded roads dataset is empty")

    if roads.crs is None:
        raise ValueError("Road dataset has no CRS")

    return roads


def clean_roads(
    roads: gpd.GeoDataFrame,
    target_crs: str = "EPSG:3006",
    road_class_col: str = "Nattyp",
    exclude_classes: list[str] | None = None,
    keep_only_classes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    if str(roads.crs) != target_crs:
        roads = roads.to_crs(target_crs)

    roads = roads[roads.geometry.notna()].copy()
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    if roads.empty:
        raise ValueError("No valid line geometries found in roads dataset")

    if road_class_col in roads.columns:
        roads[road_class_col] = (
            roads[road_class_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if keep_only_classes is not None:
            keep_only_classes = [str(x).strip().lower() for x in keep_only_classes]
            roads = roads[roads[road_class_col].isin(keep_only_classes)].copy()

        if exclude_classes is not None:
            exclude_classes = [str(x).strip().lower() for x in exclude_classes]
            roads = roads[~roads[road_class_col].isin(exclude_classes)].copy()

    roads = roads.reset_index(drop=True)

    if roads.empty:
        raise ValueError("All roads were filtered out during cleaning")

    return roads


def inspect_road_classes(
    roads: gpd.GeoDataFrame,
    road_class_col: str = "Nattyp",
    top_n: int = 30
) -> pd.Series:
    require_columns(roads, [road_class_col], "roads")

    return (
        roads[road_class_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(dropna=False)
        .head(top_n)
    )


def compute_segments_by_cell(
    grid: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    require_columns(grid, ["cell_id"], "grid")
    validate_projected_crs(grid, "grid")
    validate_projected_crs(lines, "lines")

    if grid.crs != lines.crs:
        lines = lines.to_crs(grid.crs)

    segments = gpd.overlay(
        lines,
        grid[["cell_id", "geometry"]],
        how="intersection"
    )

    if segments.empty:
        return gpd.GeoDataFrame(
            {"cell_id": [], "segment_length_m": []},
            geometry=[],
            crs=grid.crs
        )

    segments["segment_length_m"] = segments.geometry.length
    return segments


def add_basic_line_exposure(
    grid: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    length_col_name: str = "line_length_m",
    density_col_name: str = "line_density",
) -> gpd.GeoDataFrame:
    segments = compute_segments_by_cell(grid, lines)

    if segments.empty:
        out = grid.copy()
        out[length_col_name] = 0.0
        out["cell_area_m2"] = out.geometry.area
        out[density_col_name] = 0.0
        return out

    exposure = (
        segments.groupby("cell_id", as_index=False)["segment_length_m"]
        .sum()
        .rename(columns={"segment_length_m": length_col_name})
    )

    out = grid.merge(exposure, on="cell_id", how="left").copy()
    out[length_col_name] = out[length_col_name].fillna(0)

    out["cell_area_m2"] = out.geometry.area
    out[density_col_name] = np.where(
        out["cell_area_m2"] > 0,
        out[length_col_name] / out["cell_area_m2"],
        0
    )

    return out


def add_nearest_line_distance(
    grid: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    distance_col_name: str = "nearest_line_distance_m",
) -> gpd.GeoDataFrame:
    validate_projected_crs(grid, "grid")
    validate_projected_crs(lines, "lines")

    if grid.crs != lines.crs:
        lines = lines.to_crs(grid.crs)

    centroids = gpd.GeoDataFrame(
        grid[["cell_id"]].copy(),
        geometry=grid.geometry.centroid,
        crs=grid.crs
    )

    nearest = gpd.sjoin_nearest(
        centroids,
        lines[["geometry"]],
        how="left",
        distance_col=distance_col_name
    )

    out = grid.merge(
        nearest[["cell_id", distance_col_name]],
        on="cell_id",
        how="left"
    ).copy()

    out[distance_col_name] = out[distance_col_name].fillna(np.inf)
    return out


def build_linear_features(
    grid: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    prefix: str,
) -> gpd.GeoDataFrame:

    require_columns(grid, ["cell_id"], "grid")
    validate_projected_crs(grid, "grid")
    validate_projected_crs(lines, "lines")

    if grid.crs != lines.crs:
        lines = lines.to_crs(grid.crs)

    length_col = f"{prefix}_length_m"
    density_col = f"{prefix}_density"
    distance_col = f"nearest_{prefix}_distance_m"

    out = add_basic_line_exposure(
        grid=grid,
        lines=lines,
        length_col_name=length_col,
        density_col_name=density_col,
    )

    out = add_nearest_line_distance(
        grid=out,
        lines=lines,
        distance_col_name=distance_col,
    )

    return out


def add_road_class_exposure(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    road_class_col: str = "Nattyp",
    selected_classes: list[str] | None = None,
    prefix: str = "road_class",
) -> gpd.GeoDataFrame:
    require_columns(roads, [road_class_col], "roads")

    segments = compute_segments_by_cell(grid, roads)

    if segments.empty:
        return grid.copy()

    segments[road_class_col] = (
        segments[road_class_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if selected_classes is not None:
        selected_classes = [str(x).strip().lower() for x in selected_classes]
        segments = segments[segments[road_class_col].isin(selected_classes)].copy()

    if segments.empty:
        return grid.copy()

    class_lengths = (
        segments.groupby(["cell_id", road_class_col])["segment_length_m"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    rename_map = {}
    for col in class_lengths.columns:
        if col == "cell_id":
            continue
        safe = make_safe_column_name(col)
        rename_map[col] = f"{prefix}_{safe}_length_m"

    class_lengths = class_lengths.rename(columns=rename_map)

    out = grid.merge(class_lengths, on="cell_id", how="left").copy()

    class_cols = [
        c for c in out.columns
        if c.startswith(f"{prefix}_") and c.endswith("_length_m")
    ]
    for col in class_cols:
        out[col] = out[col].fillna(0)

    return out



def build_road_features(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    road_class_col: str = "Nattyp",
    exclude_classes: list[str] | None = None,
    keep_only_classes: list[str] | None = None,
    include_class_lengths: bool = True,
    selected_classes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    require_columns(grid, ["cell_id"], "grid")
    validate_projected_crs(grid, "grid")

    roads = clean_roads(
        roads=roads,
        target_crs=str(grid.crs),
        road_class_col=road_class_col,
        exclude_classes=exclude_classes,
        keep_only_classes=keep_only_classes,
    )

    out = build_linear_features(
        grid=grid,
        lines=roads,
        prefix="road",
    )

    if include_class_lengths and road_class_col in roads.columns:
        class_df = add_road_class_exposure(
            grid=grid,
            roads=roads,
            road_class_col=road_class_col,
            selected_classes=selected_classes,
            prefix="road_class",
        )

        class_cols = [
            c for c in class_df.columns
            if c.startswith("road_class_") and c.endswith("_length_m")
        ]

        if class_cols:
            out = out.merge(class_df[["cell_id"] + class_cols], on="cell_id", how="left")
            for col in class_cols:
                out[col] = out[col].fillna(0)

    return out


def load_roads_for_study_area(
    path: str,
    gdf_points: gpd.GeoDataFrame,
    target_crs: str = "EPSG:3006",
    buffer_m: float = 0,
) -> gpd.GeoDataFrame:
    if gdf_points.crs is None:
        raise ValueError("gdf_points must have a CRS")

    points = gdf_points.to_crs(target_crs).copy()
    minx, miny, maxx, maxy = points.total_bounds

    bbox = (
        minx - buffer_m,
        miny - buffer_m,
        maxx + buffer_m,
        maxy + buffer_m,
    )

    roads = load_roads(path=path, bbox=bbox)
    roads = clean_roads(roads, target_crs=target_crs)

    return roads