import pandas as pd
from typing import Dict

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    pass

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all necessary cleaning steps (e.g., filtering, standardizing, imputing)."""
    pass

def split_by_species(df: pd.DataFrame, species_col: str = 'species') -> Dict[str, pd.DataFrame]:
    """
    Split the cleaned dataset into subsets based on the species.
    Returns a dictionary mapping species names to their respective DataFrames.
    """
    return {species: group for species, group in df.groupby(species_col)}

def run_data_prep(filepath: str) -> pd.DataFrame:
    """
    Orchestrator function for data preparation.
    Executes the high-level data preparation pipeline sequentially.
    """
    df = load_raw_data(filepath)
    df = clean_dataset(df)
    return df

