import pandas as pd

def load_infrastructure_data(filepath: str) -> pd.DataFrame:
    """Loads the raw infrastructure data."""
    return pd.read_csv(filepath)

def process_infrastructure(df_infra: pd.DataFrame) -> pd.DataFrame:
    #BOILERPLATE AMANDA VAR FÖRSIKTIG.
    """
    Cleans and selects relevant infrastructure features.
    Target variables: Speed limits, Presence of wildlife fencing.
    """
    # Adjust these column names to match your actual dataset
    cols_to_keep = ['road_id', 'speed_limit', 'has_wildlife_fencing']

    # Filter only the relevant columns for predicting collision risk
    existing_cols = [col for col in cols_to_keep if col in df_infra.columns]
    df_processed = df_infra[existing_cols].copy()

    # Handle missing values for fencing (assuming missing means no fence)
    if 'has_wildlife_fencing' in df_processed.columns:
        df_processed['has_wildlife_fencing'] = df_processed['has_wildlife_fencing'].fillna(0).astype(int)

    return df_processed

def merge_infrastructure(df_collisions: pd.DataFrame, df_infra: pd.DataFrame, join_key: str = 'road_id') -> pd.DataFrame:
    """
    Merges infrastructure data (speed limits, fencing) into the main collision dataset.
    """
    return pd.merge(df_collisions, df_infra, on=join_key, how='left')

def run_infrastructure_pipeline(filepath: str, df_collisions: pd.DataFrame, join_key: str = 'road_id') -> pd.DataFrame:
    """
    Orchestrates the infrastructure data pipeline:
    1. Loads the raw infrastructure data from the given filepath
    2. Processes and cleans the infrastructure features
    3. Merges the cleaned infrastructure data into the main collision dataset
    """
    pass
