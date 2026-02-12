import pandas as pd

def temp_at_time(df, when):
    df = df.copy()

    # Konvertera till datetime
    df["time"] = pd.to_datetime(df["time"])
    when = pd.to_datetime(when)

    # skillnad i tid
    df["time_diff"] = (df["time"] - when).abs()

    # hitta närmaste tid
    row = df.loc[df["time_diff"].idxmin()]

    return row["temp"], row["time"]
