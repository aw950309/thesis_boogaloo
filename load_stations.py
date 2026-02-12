import pandas as pd


def load_stations(path):
    """
    Läser in en CSV-fil med stationsinformation och returnerar
    en DataFrame med kolumnerna:
    id, name, lat, lon

    Rensar bort ogiltiga koordinater och konverterar lat/lon till float.
    """

    # Läs in hela CSV-filen som strängar (för att undvika typ-problem)
    df = pd.read_csv(
        path,
        sep=";",  # SMHI använder semikolon som separator
        header=None,  # Ingen header-rad i korrekt format
        dtype=str,  # Läs allt som sträng först
        engine="python"  # Mer tolerant parser
    )

    # Ta bort första raden (ofta rubriker eller metadata)
    df = df.iloc[1:].reset_index(drop=True)

    # Hjälpfunktion för att konvertera lat/lon till float
    def to_float(x):
        """
        Konverterar en sträng med kommatecken-decimal till float.
        Returnerar None om värdet är ogiltigt.
        """
        if not isinstance(x, str):
            return None

        x = x.strip()  # Ta bort whitespace

        # Ignorera ogiltiga värden
        if x in ["NaN", "undefined", "", "Latitud", "Longitud"]:
            return None

        # Byt svensk decimal (,) till punkt (.)
        x = x.replace(",", ".")

        try:
            return float(x)
        except:
            return None

    # Skapa ny DataFrame med endast relevanta kolumner
    stations = pd.DataFrame({
        "id": df[0],  # Stations-ID
        "name": df[1],  # Stationsnamn
        "lat": df[4].apply(to_float),  # Latitud konverterad till float
        "lon": df[5].apply(to_float)  # Longitud konverterad till float
    })

    # Ta bort stationer som saknar giltiga koordinater
    stations = stations.dropna(subset=["lat", "lon"])

    return stations
