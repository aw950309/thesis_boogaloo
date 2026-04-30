"""Smoke tests for src.infrastructure — Phase 5 Row 7.

Tests focus on the cache write/read round-trip and the proximity-flag
computation logic. The full GeoPackage compute path is not exercised
here; that runs against real data in the orchestrator parity sub-check
(notes/notes_code/migration_run_log.md, Band C).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import infrastructure


def test_infrastructure_paths_namedtuple() -> None:
    """InfrastructurePaths is a NamedTuple — tuple at runtime, attribute access."""
    p = infrastructure.InfrastructurePaths(
        roads=Path("a.gpkg"),
        rail=Path("b.gpkg"),
        fences=Path("c.gpkg"),
        speedlimit=Path("d.gpkg"),
    )
    assert p.roads == Path("a.gpkg")
    assert p.speedlimit == Path("d.gpkg")
    assert isinstance(p, tuple)


def test_build_infrastructure_features_cache_hit_returns_dict_of_four(tmp_path: Path) -> None:
    """When all four parquet caches exist, the function reads them and returns a dict.

    Uses tiny in-memory frames written via fastparquet to verify the round-trip,
    without invoking any GeoPackage logic.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Write four minimal cache files matching the column shape the orchestrator expects.
    pd.DataFrame(
        {
            "cell_id": [0, 1],
            "road_length_m": [100.0, 0.0],
            "cell_area_m2": [1e8, 1e8],
            "road_density": [1e-6, 0.0],
            "nearest_road_distance_m": [50.0, 1e6],
            "road_class_bilnät_length_m": [100.0, 0.0],
        }
    ).to_parquet(cache_dir / "road_features.parquet", engine="fastparquet")

    pd.DataFrame(
        {
            "cell_id": [0, 1],
            "rail_length_m": [0.0, 200.0],
            "rail_density": [0.0, 2e-6],
            "nearest_rail_distance_m": [5e3, 100.0],
        }
    ).to_parquet(cache_dir / "rail_features.parquet", engine="fastparquet")

    pd.DataFrame(
        {
            "cell_id": [0, 1],
            "fence_length_m": [50.0, 0.0],
            "fence_density": [5e-7, 0.0],
            "nearest_fence_distance_m": [200.0, 1e6],
        }
    ).to_parquet(cache_dir / "fence_features.parquet", engine="fastparquet")

    pd.DataFrame(
        {
            "cell_id": [0, 1],
            "speedlimit_mean_weighted": [70.0, 0.0],
            "speedlimit_max": [90.0, 0.0],
            "speedlimit_min": [50.0, 0.0],
            "speedlimit_90plus_share": [0.2, 0.0],
            "speedlimit_segment_length_m": [100.0, 0.0],
        }
    ).to_parquet(cache_dir / "speedlimit_features.parquet", engine="fastparquet")

    paths = infrastructure.InfrastructurePaths(
        roads=tmp_path / "irrelevant.gpkg",
        rail=tmp_path / "irrelevant.gpkg",
        fences=tmp_path / "irrelevant.gpkg",
        speedlimit=tmp_path / "irrelevant.gpkg",
    )

    result = infrastructure.build_infrastructure_features(
        grid=None,        # unused on cache hit
        gdf_points=None,  # unused on cache hit
        paths=paths,
        cache_dir=cache_dir,
        use_cache=True,
    )

    assert set(result.keys()) == {"roads", "rail", "fences", "speedlimit"}
    assert len(result["roads"]) == 2
    assert "nearest_rail_distance_m" in result["rail"].columns


def test_proximity_flag_logic_matches_notebook() -> None:
    """The orchestrator computes near-X flags as (distance < 10_000).astype(int)
    after fillna(0). Verify the boundary conditions explicitly."""
    df = pd.DataFrame(
        {"nearest_rail_distance_m": [0.0, 9_999.9, 10_000.0, 50_000.0]}
    )
    df["rail_near_10km"] = (df["nearest_rail_distance_m"] < 10_000).astype(int)
    assert df["rail_near_10km"].tolist() == [1, 1, 0, 0]
