import pandas as pd
from typing import Dict


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features (e.g., season, time of day, dusk/dawn based on sun position)."""
    pass

def create_species_features(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Create features specific to species behaviour applying the correct calendar for the given species."""
    pass

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables and create necessary binary flags for the models."""
    pass

def build_all_features(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """
    Orchestrator function for feature engineering.
    Applies all feature transformations to a single species subset.
    """
    df = create_temporal_features(df)
    df = create_species_features(df, species)
    df = encode_features(df)
    return df
