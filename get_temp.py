from find_nearest import find_nearest
from get_temperatures import get_temperatures
from temp_at_time import temp_at_time


def get_temp(lat, lon, time, stations):
    """
    Hämtar temperaturen för en given plats (lat, lon)
    vid en specifik tidpunkt.

    lat, lon  = koordinater i decimalgrader
    time      = önskad tidpunkt (str eller datetime)
    stations  = DataFrame med stationer (id, name, lat, lon)

    Returnerar en dictionary med temperatur, station och metadata.
    """

    #Hitta närmaste station
    nearest = find_nearest(lat, lon, stations, 1).iloc[0]

    #Hämta alla temperaturer för den stationen
    df = get_temperatures(nearest.id)

    #Hitta temperatur närmast den önskade tiden
    temp, actual_time = temp_at_time(df, time)

    #Returnera resultat som dictionary
    return {
        "temperature": temp,                 # Temperaturvärdet
        "requested_time": str(time),         # Tiden användaren bad om
        "measured_time": str(actual_time),   # Faktisk mättid (kan skilja lite)
        "station": nearest.id,               # Stations-ID
        "name": nearest["name"],             # Stationsnamn
        "distance_km": round(nearest.dist, 2) # Avstånd i km (avrundat)
    }
