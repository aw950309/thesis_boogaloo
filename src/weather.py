# src/weather.py
"""
Weather data functions for wildlife collision thesis.
Handles SMHI station data, distance calculations, and temperature retrieval.
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in kilometers between two geographic points
    using the Haversine formula.

    lat1, lon1 = start point (can be a single point)
    lat2, lon2 = end point(s) (can be arrays/Series from pandas)
    """
    R = 6371  # Earth's radius in km

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Difference in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def load_stations(path):
    """
    Load station CSV file and return DataFrame with columns:
    id, name, lat, lon

    Removes invalid coordinates and converts lat/lon to float.
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        dtype=str,
        engine="python"
    )

    # Remove first row (often headers or metadata)
    df = df.iloc[1:].reset_index(drop=True)

    def to_float(x):
        """Convert string with comma decimal to float."""
        if not isinstance(x, str):
            return None
        x = x.strip()
        if x in ["NaN", "undefined", "", "Latitud", "Longitud"]:
            return None
        x = x.replace(",", ".")
        try:
            return float(x)
        except:
            return None

    stations = pd.DataFrame({
        "id": df[0],
        "name": df[1],
        "lat": df[4].apply(to_float),
        "lon": df[5].apply(to_float)
    })

    stations = stations.dropna(subset=["lat", "lon"])
    return stations


def find_nearest_station(lat, lon, stations, n=1):
    """
    Find the n nearest weather stations to a given point.

    lat, lon  = coordinates (decimal degrees)
    stations  = DataFrame with columns: id, name, lat, lon
    n         = number of nearest stations to return (default = 1)

    Returns DataFrame with n nearest stations including 'dist' column.
    """
    s = stations.copy()
    s["dist"] = haversine_distance(lat, lon, s.lat, s.lon)
    return s.nsmallest(n, "dist")


def get_temperatures_from_api(station_id):
    """
    Fetch temperature data from SMHI API for a given station.

    Returns DataFrame with columns: time, temp
    """
    meta_url = (
        f"https://opendata-download-metobs.smhi.se/api/"
        f"version/1.0/parameter/1/station/{station_id}/period/corrected-archive.json"
    )
    r = requests.get(meta_url)
    r.raise_for_status()
    meta = r.json()

    csv_url = meta['data'][0]['link'][0]['href']
    r_csv = requests.get(csv_url)
    r_csv.raise_for_status()

    # Find header row
    lines = r_csv.text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Datum;Tid (UTC);Lufttemperatur"):
            start_idx = i
            break

    df = pd.read_csv(
        StringIO("\n".join(lines[start_idx:])),
        sep=';',
        decimal=',',
        on_bad_lines='skip',
        encoding='utf-8'
    )

    df['time'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'], errors='coerce')
    df['temp'] = pd.to_numeric(df['Lufttemperatur'], errors='coerce')
    df = df.dropna(subset=['time', 'temp'])

    return df[['time', 'temp']]


def temp_at_time(df, when):
    """
    Find the temperature closest to a given time.

    df   = DataFrame with 'time' and 'temp' columns
    when = target datetime (str or datetime)

    Returns tuple: (temperature, actual_measurement_time)
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    when = pd.to_datetime(when)

    df["time_diff"] = (df["time"] - when).abs()
    row = df.loc[df["time_diff"].idxmin()]

    return row["temp"], row["time"]


def get_temperature(lat, lon, time, stations):
    """
    Get temperature for a given location at a specific time.

    lat, lon  = coordinates in decimal degrees
    time      = target time (str or datetime)
    stations  = DataFrame with stations (id, name, lat, lon)

    Returns dictionary with temperature, station, and metadata.
    """
    # Find nearest station
    nearest = find_nearest_station(lat, lon, stations, 1).iloc[0]

    # Fetch all temperatures for that station
    df = get_temperatures_from_api(nearest.id)

    # Find temperature closest to requested time
    temp, actual_time = temp_at_time(df, time)

    return {
        "temperature": temp,
        "requested_time": str(time),
        "measured_time": str(actual_time),
        "station": nearest.id,
        "name": nearest["name"],
        "distance_km": round(nearest.dist, 2)
    }
