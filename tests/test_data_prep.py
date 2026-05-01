"""Smoke tests for src.data_prep — Phase 5 Row 2.

Round-trip on a small in-memory NVR-shaped CSV: column rename, decimal-comma
fix, datetime parsing, Sweden-bbox filter, EPSG conversion.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import data_prep


@pytest.fixture
def nvr_csv(tmp_path: Path) -> Path:
    """Write a 6-row NVR-shaped CSV with one out-of-bbox row and one bad row."""
    rows = [
        ("2020-01-15", "Älg",       "Stockholms län", "Stockholm",  "59,33", "18,07"),  # in bbox
        ("2020-02-20", "Rådjur",    "Skåne län",      "Lund",       "55,70", "13,19"),  # in bbox
        ("2020-03-10", "Vildsvin",  "Västra Götaland","Göteborg",   "57,71", "11,97"),  # in bbox
        ("2020-04-05", "Dovhjort",  "Uppsala län",    "Uppsala",    "59,86", "17,64"),  # in bbox
        ("2020-05-12", "Älg",       "Murmansk",       "Foreign",    "68,97", "33,08"),  # OUT (lon > 25)
        ("not-a-date", "Älg",       "Stockholm",      "Stockholm",  "59,33", "18,07"),  # bad datetime
    ]
    df = pd.DataFrame(rows, columns=["Datum", "Viltslag", "Län", "Kommun", "Lat WGS84", "Long WGS84"])
    csv_path = tmp_path / "nvr_sample.csv"
    df.to_csv(csv_path, sep=";", encoding="latin1", index=False)
    return csv_path


def test_load_collision_data_round_trip(nvr_csv: Path) -> None:
    gdf = data_prep.load_collision_data(nvr_csv)

    # 4 valid rows survive (out-of-bbox dropped, bad-datetime dropped).
    assert len(gdf) == 4

    # Column rename applied.
    for col in ("datetime", "species", "lan", "kommun", "lat", "lon"):
        assert col in gdf.columns
    for original in ("Datum", "Viltslag", "Län", "Kommun", "Lat WGS84", "Long WGS84"):
        assert original not in gdf.columns

    # Decimal-comma fix worked.
    assert gdf["lat"].dtype.kind == "f"
    assert gdf["lon"].dtype.kind == "f"
    assert 55 <= gdf["lat"].min() <= gdf["lat"].max() <= 70
    assert 10 <= gdf["lon"].min() <= gdf["lon"].max() <= 25

    # Projection landed in EPSG:3006.
    assert str(gdf.crs) == "EPSG:3006"


def test_load_collision_data_multi_year_year_range(tmp_path: Path) -> None:
    """Two yearly CSVs; year_range filter keeps only the requested span."""
    for year, lat, lon in [(2019, "59,33", "18,07"), (2020, "55,70", "13,19"), (2021, "57,71", "11,97")]:
        df = pd.DataFrame(
            [(f"{year}-06-01", "Älg", "Stockholm", "Stockholm", lat, lon)],
            columns=["Datum", "Viltslag", "Län", "Kommun", "Lat WGS84", "Long WGS84"],
        )
        df.to_csv(tmp_path / f"nvr_{year}.csv", sep=";", encoding="latin1", index=False)

    gdf_all = data_prep.load_collision_data_multi_year(tmp_path)
    assert len(gdf_all) == 3

    gdf_filtered = data_prep.load_collision_data_multi_year(tmp_path, year_range=(None, 2020))
    assert len(gdf_filtered) == 2  # 2019 and 2020 retained; 2021 dropped
    assert gdf_filtered["datetime"].dt.year.max() == 2020


def test_load_collision_data_multi_year_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No CSV files"):
        data_prep.load_collision_data_multi_year(tmp_path)
