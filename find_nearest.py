from distance_km import distance_km


def find_nearest(lat, lon, stations, n=1):
    """
    Hittar de n närmaste väderstationerna till en given punkt.

    lat, lon  = koordinater (decimalgrader) för platsen du vill undersöka
    stations  = DataFrame med kolumnerna: id, name, lat, lon
    n         = hur många närmaste stationer som ska returneras (default = 1)

    Returnerar en DataFrame med de n närmaste stationerna,
    inklusive en ny kolumn 'dist' (avstånd i km).
    """

    # Skapa en kopia så att vi inte ändrar original-DataFrame
    s = stations.copy()

    # Beräkna avstånd från given punkt till alla stationer
    # distance_km fungerar med både skalärer och pandas Series
    s["dist"] = distance_km(
        lat,        # din latitud
        lon,        # din longitud
        s.lat,      # alla stationers latituder
        s.lon       # alla stationers longituder
    )

    # Returnera de n stationer med minst avstånd
    return s.nsmallest(n, "dist")
