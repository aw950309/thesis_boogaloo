"""
Weather pipeline for Wildlife Collision thesis (fixed version)
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def load_stations(path):
    df = pd.read_csv(path, sep=";", dtype=str, engine="python")
    df = df.iloc[1:].reset_index(drop=True)

    def clean(x):
        if pd.isna(x):
            return None
        x = x.replace(",", ".")
        try:
            return float(x)
        except:
            return None

    stations = pd.DataFrame({
        "id": df[0],
        "name": df[1],
        "lat": df[4].apply(clean),
        "lon": df[5].apply(clean)
    })

    return stations.dropna(subset=["lat", "lon"])


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