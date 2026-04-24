import pandas as pd
import numpy as np


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features:
    - hour, month, dayofweek
    - seasonal encoding
    - cyclic hour encoding (important for ML models)
    """

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime"])

    # basic time features
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek

    # seasons
    df["season"] = df["month"] % 12 // 3

    # cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def create_species_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds species-specific ecological features row by row.
    Assumes df contains a column named 'species'.
    """

    df = df.copy()

    if "species" not in df.columns:
        raise ValueError("DataFrame must contain a 'species' column")


    s = df["species"].astype(str).str.strip().str.lower()


    df["is_rutting_season"] = 0

    # Lol dessa behöver vi dubbelkolla lite ai varning

    # Moose (älg): Sept–Oct
    df.loc[(s.isin(["moose", "älg"])) & (df["month"].isin([9, 10])), "is_rutting_season"] = 1

    # Roe deer (rådjur): July–Aug
    df.loc[(s.isin(["roe deer", "rådjur"])) & (df["month"].isin([7, 8])), "is_rutting_season"] = 1

    # Wild boar (vildsvin): higher winter activity proxy
    df.loc[(s.isin(["wild boar", "vildsvin"])) & (df["month"].isin([11, 12, 1, 2])), "is_rutting_season"] = 1

    # Fallow deer (dovhjort): Oct–Nov
    df.loc[(s.isin(["fallow deer", "dovhjort"])) & (df["month"].isin([10, 11])), "is_rutting_season"] = 1

    return df



def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables into ML-ready format.
    """

    df = df.copy()


    for col in ["species", "Län", "Kommun"]:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df

HUNTING_PERIODS = {
    # Länsstyrelsen Stockholm: älgskötsel/licensområde
    "moose": [
        ("10-08", "01-31", 1.0),   # all allowed moose hunting
    ],

    # Naturvårdsverket: vildsvin except sow with small striped/brown piglets
    # + yearling wild boar all year
    "wild_boar": [
        ("04-01", "01-31", 1.0),   # general hunting
        ("02-01", "03-31", 0.5),   # yearlings only / reduced proxy
    ],

    # Naturvårdsverket: rådjur
    "roe_deer": [
        ("10-01", "01-31", 1.0),   # all roe deer
        ("08-16", "09-30", 0.5),   # horn-bearing males only
        ("05-01", "06-15", 0.5),   # horn-bearing males only
        ("09-01", "09-30", 0.5),   # kids only, overlaps with horn-bearing period
    ],

    # Naturvårdsverket: dovhjort
    "fallow_deer": [
        ("10-01", "10-20", 1.0),   # all animals
        ("11-16", "02-28", 1.0),   # all animals
        ("09-01", "09-30", 0.5),   # horn-bearing males + calves only
        ("10-21", "11-15", 0.5),   # hind + calf only
        ("03-01", "03-31", 0.5),   # hind + calf only
    ],
}

RUT_PERIODS = {
    "moose": [
        ("09-15", "10-15"),   # peak late Sept–early Oct
    ],
    "roe_deer": [
        ("07-15", "08-15"),   # summer rut
    ],
    "wild_boar": [
        ("11-01", "02-28"),   # extended winter rut
    ],
    "fallow_deer": [
        ("10-01", "11-15"),   # autumn rut
    ],
}


def month_overlap_fraction(period_start, start_str, end_str):
    year = period_start.year

    start = pd.Timestamp(f"{year}-{start_str}")
    end   = pd.Timestamp(f"{year}-{end_str}")

    # Handle wrap-around (e.g. Oct → Jan)
    if end < start:
        if period_start.month >= start.month:
            end = pd.Timestamp(f"{year+1}-{end_str}")
        else:
            start = pd.Timestamp(f"{year-1}-{start_str}")

    month_start = period_start
    month_end   = period_start + pd.offsets.MonthEnd(1)

    overlap_start = max(start, month_start)
    overlap_end   = min(end, month_end)

    if overlap_start > overlap_end:
        return 0.0

    overlap_days = (overlap_end - overlap_start).days + 1
    month_days   = (month_end - month_start).days + 1

    return overlap_days / month_days

def build_hunting_features(df):
    df = df.copy()

    for species, periods in HUNTING_PERIODS.items():
        values = []

        for _, row in df.iterrows():
            period = row["period_start"]

            frac = 0
            for start, end, weight in periods:
                overlap = month_overlap_fraction(period, start, end)
                frac += overlap * weight

            values.append(min(frac, 1.0))

        df[f"{species}_hunting_frac"] = values

    return df

def build_rut_features(df):
    df = df.copy()

    for species, periods in RUT_PERIODS.items():
        values = []

        for _, row in df.iterrows():
            period = row["period_start"]

            frac = 0
            for start, end in periods:
                frac += month_overlap_fraction(period, start, end)

            values.append(min(frac, 1.0))

        df[f"{species}_rut_frac"] = values

    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline:
    1. temporal features
    2. species behaviour
    3. encoding
    """
    df = create_temporal_features(df)
    df = create_species_features(df)
    df = encode_features(df)

    return df