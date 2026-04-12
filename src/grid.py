import numpy as np
import geopandas as gpd
from shapely.geometry import box


def create_grid(gdf, cell_size=5000):
    """
    Creates a square grid over the bounding box of a GeoDataFrame.

    Parameters:
        gdf: GeoDataFrame in projected CRS (e.g. EPSG:3006)
        cell_size: grid resolution in meters

    Returns:
        GeoDataFrame with grid cells
    """

    minx, miny, maxx, maxy = gdf.total_bounds

    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    grid_cells = []

    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=gdf.crs)
    grid["cell_id"] = range(len(grid))

    return grid


def spatial_join_points_to_grid(gdf_points, grid):
    """
    Assigns each collision point to a grid cell.
    """

    return gpd.sjoin(
        gdf_points,
        grid,
        how="left",
        predicate="within"
    )


def compute_grid_risk(gdf_joined):
    """
    Aggregates collision counts per grid cell and creates risk label.
    """

    cell_counts = (
        gdf_joined
        .groupby("cell_id")
        .size()
        .reset_index(name="collision_count")
    )

    threshold = cell_counts["collision_count"].median()

    cell_counts["risk"] = (cell_counts["collision_count"] >= threshold).astype(int)

    return cell_counts



def build_spatial_grid_pipeline(gdf_points, cell_size=5000):
    """
    End-to-end spatial grid pipeline.
    """

    grid = create_grid(gdf_points, cell_size)

    joined = spatial_join_points_to_grid(gdf_points, grid)

    risk_table = compute_grid_risk(joined)

    return grid, joined, risk_table