import pandas as pd
import requests
from io import StringIO


def get_temperatures(station_id):
    # Hämta metadata
    meta_url = (
        f"https://opendata-download-metobs.smhi.se/api/"
        f"version/1.0/parameter/1/station/{station_id}/period/corrected-archive.json"
    )
    r = requests.get(meta_url)
    r.raise_for_status()
    meta = r.json()

    # Hämta CSV-länk
    csv_url = meta['data'][0]['link'][0]['href']
    r_csv = requests.get(csv_url)
    r_csv.raise_for_status()

    # Hitta rubrikraden (Datum;Tid (UTC);Lufttemperatur;Kvalitet)
    lines = r_csv.text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Datum;Tid (UTC);Lufttemperatur"):
            start_idx = i
            break

    # Läs CSV från rubrikraden och hoppa över dåliga rader
    df = pd.read_csv(
        StringIO("\n".join(lines[start_idx:])),
        sep=';',
        decimal=',',
        on_bad_lines='skip',
        encoding='utf-8'
    )

    # Kombinera Datum + Tid till en datetime-kolumn
    df['time'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'], errors='coerce')
    df['temp'] = pd.to_numeric(df['Lufttemperatur'], errors='coerce')

    # Håll bara de rader med giltiga värden
    df = df.dropna(subset=['time', 'temp'])

    return df[['time', 'temp']]
