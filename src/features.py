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