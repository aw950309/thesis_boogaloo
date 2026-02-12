import numpy as np

def distance_km(lat1, lon1, lat2, lon2):
    """
    Beräknar avståndet i kilometer mellan två geografiska punkter
    med hjälp av Haversine-formeln.

    lat1, lon1 = startpunkt (kan vara en enskild punkt)
    lat2, lon2 = målpunkt(er) (kan vara arrays/Series från pandas)
    """

    # Jordens radie i kilometer
    R = 6371

    # Konvertera grader till radianer (krävs för trigonometriska funktioner)
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    # Skillnad i latitud och longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine-formeln
    # a är ett mellanvärde som används för att räkna ut sfäriskt avstånd
    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    # Slutlig distans i kilometer
    return 2 * R * np.arcsin(np.sqrt(a))
