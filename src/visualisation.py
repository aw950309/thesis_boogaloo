import pandas as pd
from typing import Any

def export_figure(fig, filename: str, species: str) -> None:
    """
    Helper function to save generated plots consistently to the outputs/figures/ directory.
    """
    pass

def plot_model_evaluations(model: Any, X_test: pd.DataFrame, y_test: pd.Series, species: str) -> None:
    """
    Generates performance evaluation plots for the trained model
    (e.g., feature importance, confusion matrix).
    Calls export_figure() to save them.
    """
    pass

def plot_analytical_figures(df: pd.DataFrame, species: str) -> None:
    """
    Generates analytical and exploratory figures based on the dataset
    (e.g., temporal distributions, spatial heatmaps).
    Calls export_figure() to save them.
    """
    pass

def generate_visualisations(df: pd.DataFrame, model: Any, X_test: pd.DataFrame, y_test: pd.Series, species: str) -> None:
    """
    Orchestrator function for visualization.
    Generates both analytical plots and model evaluation metrics for a specific species.
    All outputs will be saved to the outputs/figures/ directory.
    """
    plot_analytical_figures(df, species)
    if model is not None:
        plot_model_evaluations(model, X_test, y_test, species)


