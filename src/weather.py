"""
Weather pipeline for Wildlife Collision thesis (fixed version)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from io import StringIO
from pathlib import Path


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))
def load_temperature_stations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, engine="python")
    df = df.iloc[1:].reset_index(drop=True)

    def clean_num(x):
        if pd.isna(x):
            return np.nan
        x = str(x).replace(",", ".")
        try:
            return float(x)
        except ValueError:
            return np.nan

    stations = pd.DataFrame({
        "station_id": df[0].astype(str).str.strip(),
        "station_name": df[1].astype(str).str.strip(),
        "lat": df[4].apply(clean_num),
        "lon": df[5].apply(clean_num),
    })

    stations = stations.dropna(subset=["lat", "lon"]).drop_duplicates("station_id")
    return stations.reset_index(drop=True)

def find_nearest_station(lat, lon, stations):
    s = stations.copy()
    s["dist"] = haversine_distance(lat, lon, s.lat, s.lon)
    return s.sort_values("dist").iloc[0]


def get_station_weather(station_id):
    url = (
        f"https://opendata-download-metobs.smhi.se/api/"
        f"version/1.0/parameter/1/station/{station_id}/period/corrected-archive.json"
    )

    r = requests.get(url)
    r.raise_for_status()
    meta = r.json()

    csv_url = meta["data"][0]["link"][0]["href"]

    r = requests.get(csv_url)
    r.raise_for_status()

    lines = r.text.splitlines()

    start = 0
    for i, line in enumerate(lines):
        if line.startswith("Datum;Tid"):
            start = i
            break

    df = pd.read_csv(
        StringIO("\n".join(lines[start:])),
        sep=";",
        decimal=",",
        on_bad_lines="skip"
    )

    df["time"] = pd.to_datetime(df["Datum"] + " " + df["Tid (UTC)"], errors="coerce")
    df["temp"] = pd.to_numeric(df["Lufttemperatur"], errors="coerce")

    return df[["time", "temp"]].dropna()
def fetch_temperature_stations_from_api() -> pd.DataFrame:
    """
    Fetch all SMHI temperature stations (parameter 1).
    """
    url = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/1.json"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    stations = []
    for s in data.get("station", []):
        lat = s.get("latitude")
        lon = s.get("longitude")

        try:
            lat = float(str(lat).replace(",", "."))
            lon = float(str(lon).replace(",", "."))
        except:
            continue

        stations.append({
            "station_id": str(s.get("id")),
            "station_name": s.get("name"),
            "lat": lat,
            "lon": lon,
            "height": s.get("height"),
            "country": s.get("country"),
            "active": s.get("active"),
        })

    return pd.DataFrame(stations)

def assign_nearest_temperature_station(
    grid: gpd.GeoDataFrame,
    stations: pd.DataFrame,
    cell_id_col: str = "cell_id"
) -> pd.DataFrame:
    if grid.crs is None:
        raise ValueError("grid must have a CRS")

    grid_proj = grid.to_crs("EPSG:3006")
    centroids = grid_proj[[cell_id_col, "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids["lat"] = centroids.geometry.y
    centroids["lon"] = centroids.geometry.x

    out = []

    for _, row in centroids[[cell_id_col, "lat", "lon"]].iterrows():
        d = haversine_distance(
            row["lat"],
            row["lon"],
            stations["lat"].values,
            stations["lon"].values
        )
        idx = int(np.argmin(d))
        nearest = stations.iloc[idx]

        out.append({
            "cell_id": row[cell_id_col],
            "station_id": nearest["station_id"],
            "station_name": nearest["station_name"],
            "station_distance_km": float(d[idx]),
        })

    return pd.DataFrame(out)
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_station_temperature_history(
    station_id: str,
    cache_dir: str = "cache/temperature",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Fetch corrected archive temperature data for one station from SMHI.
    Returns columns: time, temp
    """
    cache_dir = ensure_dir(cache_dir)
    cache_file = cache_dir / f"temperature_station_{station_id}.csv"

    if cache_file.exists() and not force_download:
        df_cached = pd.read_csv(cache_file)
        df_cached["time"] = pd.to_datetime(df_cached["time"], errors="coerce", utc=True)
        df_cached["temp"] = pd.to_numeric(df_cached["temp"], errors="coerce")
        return df_cached.dropna(subset=["time", "temp"]).reset_index(drop=True)

    url = (
        f"https://opendata-download-metobs.smhi.se/api/"
        f"version/1.0/parameter/1/station/{station_id}/period/corrected-archive.json"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"Metadata request failed for station {station_id}: {r.status_code}")
            return pd.DataFrame(columns=["time", "temp"])

        meta = r.json()
        csv_url = meta["data"][0]["link"][0]["href"]

        r2 = requests.get(csv_url, timeout=60)
        if r2.status_code != 200:
            print(f"CSV request failed for station {station_id}: {r2.status_code}")
            return pd.DataFrame(columns=["time", "temp"])

        lines = r2.text.splitlines()

        # Find the real table header row
        start = None
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if (
                "Datum" in line_clean
                and ";" in line_clean
                and ("Tid" in line_clean or "UTC" in line_clean)
            ):
                start = i
                break

        if start is None:
            print(f"Could not find data header for station {station_id}")
            print("First 30 lines:")
            for line in lines[:30]:
                print(line)
            return pd.DataFrame(columns=["time", "temp"])

        # Read only from the actual table header
        df = pd.read_csv(
            StringIO("\n".join(lines[start:])),
            sep=";",
            decimal=",",
            on_bad_lines="skip",
            low_memory=False
        )

        print(f"Parsed columns for station {station_id}: {df.columns.tolist()}")

        # Find columns flexibly
        date_col = next((c for c in df.columns if "Datum" in str(c) and "Tid" not in str(c)), None)
        time_col = next((c for c in df.columns if "Tid" in str(c) and "UTC" in str(c)), None)
        temp_col = next((c for c in df.columns if "temperatur" in str(c).lower()), None)

        print("date_col:", date_col)
        print("time_col:", time_col)
        print("temp_col:", temp_col)

        if date_col is None or time_col is None or temp_col is None:
            print(f"Missing expected columns for station {station_id}")
            return pd.DataFrame(columns=["time", "temp"])

        df["time"] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",
            utc=True
        )
        df["temp"] = pd.to_numeric(df[temp_col], errors="coerce")

        out = (
            df[["time", "temp"]]
            .dropna()
            .sort_values("time")
            .reset_index(drop=True)
        )

        if not out.empty:
            out.to_csv(cache_file, index=False)

        return out

    except Exception as e:
        print(f"Error for station {station_id}: {e}")
        return pd.DataFrame(columns=["time", "temp"])


def aggregate_monthly_temperature(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return pd.DataFrame(columns=["period_start", "temp_mean", "temp_min", "temp_max"])

    df = weather_df.copy()
    df = df.set_index("time").sort_index()

    monthly = df.resample("MS").agg(
        temp_mean=("temp", "mean"),
        temp_min=("temp", "min"),
        temp_max=("temp", "max"),
    ).reset_index()

    monthly = monthly.rename(columns={"time": "period_start"})
    monthly["period_start"] = pd.to_datetime(monthly["period_start"]).dt.tz_localize(None)

    return monthly

def build_cell_month_temperature(
    grid: gpd.GeoDataFrame,
    cache_dir: str = "cache/temperature"
) -> pd.DataFrame:
    stations = fetch_temperature_stations_from_api()
    cell_station = assign_nearest_temperature_station(grid, stations)

    parts = []

    for station_id in cell_station["station_id"].dropna().unique():
        station_weather = get_station_temperature_history(
            station_id=station_id,
            cache_dir=cache_dir,
            force_download=False
        )

        monthly = aggregate_monthly_temperature(station_weather)
        if monthly.empty:
            continue

        monthly["station_id"] = station_id
        parts.append(monthly)

    if not parts:
        return pd.DataFrame(columns=[
            "cell_id", "period_start", "temp_mean", "temp_min", "temp_max"
        ])

    station_month = pd.concat(parts, ignore_index=True)

    out = cell_station.merge(
        station_month,
        on="station_id",
        how="left"
    )

    return out[[
        "cell_id",
        "period_start",
        "temp_mean",
        "temp_min",
        "temp_max",
        "station_id",
        "station_name",
        "station_distance_km",
    ]]

def temp_at_time(weather_df, when):
    weather_df = weather_df.copy()
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    when = pd.to_datetime(when)

    weather_df["diff"] = (weather_df["time"] - when).abs()

    row = weather_df.loc[weather_df["diff"].idxmin()]

    return row["temp"]


def add_weather_feature(df, stations):
    df = df.copy()
    weather_cache = {}
    station_ids = []
    temps = []

    for _, row in df.iterrows():
        station = find_nearest_station(row["lat"], row["lon"], stations)
        station_id = station["id"]
        station_ids.append(station_id)

        if station_id not in weather_cache:
            weather_cache[station_id] = get_station_weather(station_id)

        weather = weather_cache[station_id]
        temp = temp_at_time(weather, row["datetime"])
        temps.append(temp)

    df["nearest_station_id"] = station_ids
    df["temperature"] = temps
    return df

def fetch_precipitation_stations_from_api() -> pd.DataFrame:

    url = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/23.json"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    stations = []
    for s in data.get("station", []):
        lat = s.get("latitude")
        lon = s.get("longitude")

        try:
            lat = float(str(lat).replace(",", "."))
            lon = float(str(lon).replace(",", "."))
        except Exception:
            continue

        stations.append({
            "station_id": str(s.get("id")),
            "station_name": s.get("name"),
            "lat": lat,
            "lon": lon,
            "height": s.get("height"),
            "country": s.get("country"),
            "active": s.get("active"),
        })

    return pd.DataFrame(stations)

def get_station_precipitation_history(
    station_id: str,
    cache_dir: str = "cache/precipitation",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Fetch monthly precipitation totals for one station from SMHI.
    Returns columns: period_start, precip_total
    """
    cache_dir = ensure_dir(cache_dir)
    cache_file = cache_dir / f"precipitation_station_{station_id}.csv"

    if cache_file.exists() and not force_download:
        df_cached = pd.read_csv(cache_file)
        df_cached["period_start"] = pd.to_datetime(df_cached["period_start"], errors="coerce")
        df_cached["precip_total"] = pd.to_numeric(df_cached["precip_total"], errors="coerce")
        return df_cached.dropna(subset=["period_start", "precip_total"]).reset_index(drop=True)

    url = (
        f"https://opendata-download-metobs.smhi.se/api/"
        f"version/1.0/parameter/23/station/{station_id}/period/corrected-archive.json"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"Metadata request failed for precipitation station {station_id}: {r.status_code}")
            return pd.DataFrame(columns=["period_start", "precip_total"])

        meta = r.json()
        csv_url = meta["data"][0]["link"][0]["href"]

        r2 = requests.get(csv_url, timeout=60)
        if r2.status_code != 200:
            print(f"CSV request failed for precipitation station {station_id}: {r2.status_code}")
            return pd.DataFrame(columns=["period_start", "precip_total"])

        lines = r2.text.splitlines()

        start = None
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if "Datum" in line_clean and ";" in line_clean:
                start = i
                break

        if start is None:
            print(f"Could not find data header for precipitation station {station_id}")
            return pd.DataFrame(columns=["period_start", "precip_total"])

        df = pd.read_csv(
            StringIO("\n".join(lines[start:])),
            sep=";",
            decimal=",",
            on_bad_lines="skip",
            low_memory=False
        )

        print(f"Parsed precipitation columns for station {station_id}: {df.columns.tolist()}")

        date_col = next((c for c in df.columns if "Datum" in str(c)), None)
        precip_col = next(
            (
                c for c in df.columns
                if "nederb" in str(c).lower()
                or "precip" in str(c).lower()
            ),
            None
        )

        print("date_col:", date_col)
        print("precip_col:", precip_col)

        if date_col is None or precip_col is None:
            print(f"Missing expected precipitation columns for station {station_id}")
            return pd.DataFrame(columns=["period_start", "precip_total"])

        df["period_start"] = pd.to_datetime(df[date_col], errors="coerce")
        df["period_start"] = df["period_start"].dt.to_period("M").dt.to_timestamp()
        df["precip_total"] = pd.to_numeric(df[precip_col], errors="coerce")

        out = (
            df[["period_start", "precip_total"]]
            .dropna()
            .sort_values("period_start")
            .reset_index(drop=True)
        )

        if not out.empty:
            out.to_csv(cache_file, index=False)

        return out

    except Exception as e:
        print(f"Error for precipitation station {station_id}: {e}")
        return pd.DataFrame(columns=["period_start", "precip_total"])


def build_cell_month_precipitation(
    grid: gpd.GeoDataFrame,
    cache_dir: str = "cache/precipitation"
) -> pd.DataFrame:
    stations = fetch_precipitation_stations_from_api()
    cell_station = assign_nearest_temperature_station(grid, stations)

    parts = []

    for station_id in cell_station["station_id"].dropna().unique():
        monthly = get_station_precipitation_history(
            station_id=station_id,
            cache_dir=cache_dir,
            force_download=False
        )

        if monthly.empty:
            continue

        monthly["station_id"] = station_id
        parts.append(monthly)

    if not parts:
        return pd.DataFrame(columns=[
            "cell_id", "period_start", "precip_total"
        ])

    station_month = pd.concat(parts, ignore_index=True)

    out = cell_station.merge(
        station_month,
        on="station_id",
        how="left"
    )

    return out[[
        "cell_id",
        "period_start",
        "precip_total",
        "station_id",
        "station_name",
        "station_distance_km",
    ]]