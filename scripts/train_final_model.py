"""Train the final WVC collision-risk model.

Run from the code/ repo root:

    python scripts/train_final_model.py

Phase 5 work-in-progress. The orchestrator is wired band-by-band per
notes/notes_code/phase5_plan.md.
"""
from __future__ import annotations

import random as _random
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from src import data_prep
from src import grid as grid_mod
from src import features
from src import infrastructure
from src import weather
from src import models
from src import visualisation
from src.config import FEATURES, GROUPS, SPECIES_LIST, SPECIES_LABELS, get_species_features
from src.grid import build_species_model_df as _build_species_model_df
from src.exports import export_artefacts


# ─── Kawaii progress reporter ✿◕ ‿ ◕✿ ───────────────────────────────────
# Maximally over-the-top cute step counter for Amanda. The pipeline has 23
# logical steps. Each step prints a sparkly header on entry and a celebratory
# completion line with a rotating kaomoji on exit; the start and end banners
# are extra extravagant (˶ᵔ ᵕ ᵔ˶)♡

_TOTAL_STEPS = 23

_STEP_EMOJI = [
    "🐱", "🌸", "🌷", "🌼", "🌻", "🌹", "🌺", "💖", "✨", "🦄",
    "🎀", "💕", "🍓", "🧁", "🌈", "🦋", "🪻", "🌙", "⭐", "🐰",
    "💗", "🍡", "🪐",
]

_TRAIL_EMOJI = [
    "🌷✨", "💕🎀", "🌸💖", "🦋✨", "🌟💗", "🍓💞", "🧁💕", "🪻💖",
    "🌼✨", "🐱💕", "🦄🌈", "🌹💖", "💝🌸", "🌺✨", "🎀💗",
]

_KAOMOJI = [
    "(◕‿◕✿)", "( ˘͈ ᵕ ˘͈♡)", "(✿◠‿◠)", "(｡♥‿♥｡)", "ヾ(＾∇＾)",
    "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "(✿ ♡‿♡)", "(*˘︶˘*).｡.:*♡", "٩(◕‿◕)۶",
    "(づ｡◕‿‿◕｡)づ", "ʕ•́ᴥ•̀ʔっ♡", "(ᵔᴥᵔ)", "(っ◔◡◔)っ ♥", "(♡˙︶˙♡)",
    "ヾ(＾-＾)ノ", "(˶ᵔ ᵕ ᵔ˶)♡", "ʚ♡⃛ɞ", "(っ˘ω˘ς )", "(/^▽^)/",
    "(✯◡✯)", "(◍•ᴗ•◍)❤", "(*✧×✧*)",
]

_VERBS = [
    "✨ done in", "🌸 finished in", "💕 wrapped up in", "🎀 all done in",
    "💖 completed in", "🌷 ready in", "✨ baked in", "💗 prepped in",
]

_RNG = _random.Random(42)  # deterministic — same kawaii order every run

_STATE = {"step": 0, "t0": 0.0}


def _step_start(label: str) -> float:
    """Print a sparkly kawaii step header and bump the step counter."""
    _STATE["step"] += 1
    n = _STATE["step"]
    emoji = _STEP_EMOJI[(n - 1) % len(_STEP_EMOJI)]
    trail = _TRAIL_EMOJI[(n - 1) % len(_TRAIL_EMOJI)]
    print(f"\n  ╭─ ✿ ─ {emoji} ─ ✿ ─ ✿ ─ {trail} ─ ✿ ─ ✿ ─ ✿ ─╮")
    print(f"  │ [{n:2d}/{_TOTAL_STEPS}] {emoji} {label}… {trail}")
    print(f"  ╰─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─╯", flush=True)
    return _time.time()


def _step_end(t_start: float, message: str) -> None:
    """Print a cute completion line with rotating kaomoji + cute verb."""
    elapsed = _time.time() - t_start
    kao = _KAOMOJI[_STATE["step"] % len(_KAOMOJI)]
    verb = _VERBS[_STATE["step"] % len(_VERBS)]
    print(f"        🌷 {message}")
    print(f"        💕 {kao}  {verb} {elapsed:5.1f}s  ✧･ﾟ:*", flush=True)


def _banner_start() -> None:
    _STATE["step"] = 0
    _STATE["t0"] = _time.time()
    print("")
    print("  ✿*ﾟ‘ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ‘ﾟ･✿.｡  ")
    print("  ✿  ٩(♡ε♡ )۶  ❀  ٩(♡ε♡ )۶  ❀  ٩(♡ε♡ )۶  ❀  ٩(♡ε♡ )۶  ✿  ")
    print("  ✿*ﾟ‘ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ‘ﾟ･✿.｡  ")
    print("                                                                   ")
    print("       🌸💖 ✧ W I L D L I F E   C O L L I S I O N ✧ 💖🌸          ")
    print("       🌷✨ ✧     P R E D I C T I O N   ⋆ M O D E L     ✧ ✨🌷    ")
    print("                                                                   ")
    print("  ✿*ﾟ‘ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ‘ﾟ･✿.｡  ")
    print("                                                                   ")
    print(f"        🦄✨   starting up the kawaii train   🚂🎀✨               ")
    print(f"        🌸💕   {_TOTAL_STEPS} cute little steps to chug through   💕🌸  ")
    print(f"        🐱♡    please be patient, lots of moose to count    ♡🐱   ")
    print(f"        🌷    ٩(◕‿◕✿)۶  ✧･ﾟ:*  here we go!!  *:･ﾟ✧  (◕‿◕✿)۶    🌷")
    print("                                                                   ")
    print("  ✿*ﾟ‘ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ‘ﾟ･✿.｡  ", flush=True)


def _banner_end(output_dir: Path, models_dir: Path, figures_dir: Path) -> None:
    total = _time.time() - _STATE["t0"]
    mins, secs = divmod(int(total), 60)
    print("")
    print("  🌈✨💖🎀🌸💕  ✧･ﾟ:*✧･ﾟ:*  P I P E L I N E   C O M P L E T E !!  *:･ﾟ✧*:･ﾟ✧  💕🌸🎀💖✨🌈")
    print("                                                                                          ")
    print("        ⋆｡˚ ⋆｡˚ ⋆｡˚    ٩( ๑•̀o•́๑ )و    ⋆｡˚ ⋆｡˚ ⋆｡˚                                       ")
    print(f"             🐱   total time     ➜   {mins:>2}m {secs:>2}s   ( ´ ▽ ` )ﾉ ♡                ")
    print(f"             🌸   steps done     ➜   {_STATE['step']}/{_TOTAL_STEPS}  ✓✓✓ ✧･ﾟ:*           ")
    print(f"             🌷   CSVs           ➜   {output_dir}                                          ")
    print(f"             🦄   models         ➜   {models_dir}                                          ")
    print(f"             🎀   figures        ➜   {figures_dir}                                         ")
    print("                                                                                          ")
    print("        ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿                   ")
    print("                                                                                          ")
    print("        🌸💖💕   thank you for running, Amanda!!   💕💖🌸                                 ")
    print("        🌷🎀✨   you are absolutely loved   ✿(◍•ᴗ•◍)❤   ✨🎀🌷                          ")
    print("        🦄💗💞   have a wonderful, sparkly day   💞💗🦄                                   ")
    print("                                                                                          ")
    print("              ╭─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─╮                                              ")
    print("              │   ⋆｡˚    ٩(◕‿◕✿)۶    ˚｡⋆     │                                          ")
    print("              ╰─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─╯                                              ")
    print("                                                                                          ")
    print("  🌸✿*ﾟ‘ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ’ﾟ･✿.｡.:* *.:｡✿*ﾟ‘ﾟ･✿.｡✿🌸  ")
    print("")


def _dump_parity_arrays(
    dump_dir: Path,
    *,
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    results_df: pd.DataFrame,
    mean_importance: pd.Series,
    calibration_xy: pd.DataFrame,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    pr_thresholds: np.ndarray,
    ap: float,
    cell_risk: pd.DataFrame,
    group_importance_df: pd.DataFrame,
    rf_final_preds: np.ndarray,
    rf_calibrated_preds: np.ndarray,
) -> None:
    """Write Phase 6 parity-verification artefacts to dump_dir.

    Only called when --dump-parity-arrays is supplied. Default run is a no-op
    (dump_dir is None). All artefact names and formats mirror the Phase 1
    baseline in notes/notes_code/parity_baseline/.
    """
    arrays_dir = dump_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    np.save(arrays_dir / "rf_final_predictions.npy", rf_final_preds)
    np.save(arrays_dir / "rf_calibrated_predictions.npy", rf_calibrated_preds)
    np.save(arrays_dir / "oof_probs.npy", oof_probs)
    np.save(arrays_dir / "oof_labels.npy", oof_labels)

    results_df.to_csv(arrays_dir / "results_df.csv", index=False)
    mean_importance.to_csv(arrays_dir / "mean_importance.csv")

    calibration_xy.to_csv(arrays_dir / "calibration_curve.csv", index=False)

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}).to_csv(
        arrays_dir / "roc_curve.csv", index=False
    )

    # pr_thresholds is length N-1; pad with NaN to align with precision/recall
    # length-N arrays (matches sklearn's precision_recall_curve convention and
    # the Phase 1 baseline capture).
    pr_thresh_padded = np.append(pr_thresholds, np.nan)
    pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "thresholds": pr_thresh_padded,
    }).to_csv(arrays_dir / "pr_curve.csv", index=False)

    (arrays_dir / "pr_average_precision.txt").write_text(
        f"{ap:.18f}\n", encoding="utf-8"
    )

    cell_risk.to_csv(arrays_dir / "cell_risk.csv", index=False)
    group_importance_df.to_csv(arrays_dir / "group_importance.csv", index=False)

    print(f"        📋 parity arrays written to {arrays_dir}", flush=True)


def main(
    data_dir: Path = Path("data"),
    parquet_cache_dir: Path = Path("data/processed/cache"),
    # Pinned per-station SMHI cache root. Subdirs `temperature/` and
    # `precipitation/` are passed to weather.py functions explicitly, NOT
    # left to weather.py's CWD-relative defaults (which resolve unpredictably
    # depending on launch directory). Phase 5 row 8 / FLAG_012-adjacent
    # decision; weather.py module itself is unmodified per H8.
    weather_cache_dir: Path = Path("notebooks/cache"),
    output_dir: Path = Path("data/processed"),
    models_dir: Path = Path("outputs/models"),
    figures_dir: Path = Path("outputs/figures"),
    seed: int = 42,
    species_filter: str | None = None,
    use_cache: bool = True,
    hyperparameters_path: Path = Path("config/hyperparameters.yaml"),
    dump_parity_arrays: Path | None = None,
) -> None:
    """Run the WVC pipeline end-to-end.

    Phase 5 wires this band-by-band. Each band is commented in or out
    as the corresponding breakup_map rows land.
    """
    _banner_start()

    # === Band A — Foundation (Rows 2, 3) ===
    t = _step_start("Loading NVR collision CSVs")
    gdf = data_prep.load_collision_data_multi_year(data_dir / "Collisions", year_range=(None, 2025))
    _step_end(t, f"loaded {len(gdf):,} collisions")

    t = _step_start("Building cell × month panel")
    grid_full, joined, cell_month = grid_mod.build_cell_month_panel(gdf, cell_size=10000)
    _step_end(t, f"{len(grid_full):,} cells × {cell_month['period_start'].nunique()} months")

    # === Band B — Per-cell features (Rows 5, 6) ===
    t = _step_start("Building lagged light-condition features")
    lagged_light = features.build_lagged_light(joined)
    _step_end(t, f"{len(lagged_light):,} cell-month rows")

    t = _step_start("Building lagged species features (4 species)")
    lagged_species = features.build_lagged_species(joined)
    _step_end(t, f"{len(lagged_species):,} cell-month rows")

    # === Band C — Infrastructure overlays (Row 7) ===
    t = _step_start("Loading infrastructure features (roads/rail/fences/speed)")
    paths = infrastructure.InfrastructurePaths(
        roads=data_dir / "Sverige_Vägtrafiknät_GeoPackage" / "Sverige_Vägtrafiknät_194602.gpkg",
        rail=data_dir / "Järnvägnät_grundegenskaper" / "Järnvägsnät_grundegenskaper3_0_GeoPackage.gpkg",
        fences=data_dir / "Barrairanalys" / "Barriaranalys.gpkg",
        speedlimit=data_dir / "Speedlimit" / "ISA.gpkg",
    )
    infra = infrastructure.build_infrastructure_features(
        grid=grid_full,
        gdf_points=gdf,
        paths=paths,
        cache_dir=parquet_cache_dir,
        use_cache=use_cache,
    )
    _step_end(t, f"4 feature sets ready ({'cache' if use_cache else 'fresh compute'})")

    t = _step_start("Merging roads + filtering road_length_m > 0")
    model_df = (
        cell_month
        .merge(infra["roads"].drop(columns="geometry", errors="ignore"), on="cell_id", how="left")
        .query("road_length_m > 0")
    )
    _step_end(t, f"model_df now {len(model_df):,} rows")

    t = _step_start("Merging rail + computing near-10km flag")
    model_df = model_df.merge(
        infra["rail"].drop(columns=["geometry", "cell_area_m2"], errors="ignore"),
        on="cell_id", how="left",
    )
    for col in ["rail_length_m", "rail_density", "nearest_rail_distance_m"]:
        model_df[col] = model_df[col].fillna(0)
    model_df["rail_near_10km"] = (model_df["nearest_rail_distance_m"] < 10_000).astype(int)
    _step_end(t, f"{int(model_df['rail_near_10km'].sum()):,} cell-months near rail")

    t = _step_start("Merging fences + computing near-10km flag")
    model_df = model_df.merge(
        infra["fences"].drop(columns=["geometry", "cell_area_m2"], errors="ignore"),
        on="cell_id", how="left",
    )
    for col in ["fence_length_m", "fence_density", "nearest_fence_distance_m"]:
        model_df[col] = model_df[col].fillna(0)
    model_df["fence_near_10km"] = (model_df["nearest_fence_distance_m"] < 10_000).astype(int)
    _step_end(t, f"{int(model_df['fence_near_10km'].sum()):,} cell-months near a fence")

    t = _step_start("Merging speed-limit features")
    model_df = model_df.merge(
        infra["speedlimit"].drop(columns="geometry", errors="ignore"),
        on="cell_id", how="left",
    )
    for col in [
        "speedlimit_mean_weighted",
        "speedlimit_max",
        "speedlimit_min",
        "speedlimit_90plus_share",
        "speedlimit_segment_length_m",
    ]:
        model_df[col] = model_df[col].fillna(0)
    _step_end(t, f"model_df now {len(model_df):,} × {len(model_df.columns)}")

    # === Band D — Orchestrator merges (Rows 8, 9) ===
    t = _step_start("Merging weather (temperature + precipitation, then dropna)")
    relevant_cell_ids = model_df["cell_id"].unique()
    grid_small = grid_full[grid_full["cell_id"].isin(relevant_cell_ids)]

    temperature_features = weather.build_cell_month_temperature(
        grid=grid_small,
        cache_dir=str(weather_cache_dir / "temperature"),
    )
    model_df = model_df.merge(
        temperature_features[["cell_id", "period_start", "temp_mean", "temp_min", "temp_max"]],
        on=["cell_id", "period_start"],
        how="left",
    )

    precip_features = weather.build_cell_month_precipitation(
        grid=grid_small,
        cache_dir=str(weather_cache_dir / "precipitation"),
    )
    model_df = model_df.merge(
        precip_features[["cell_id", "period_start", "precip_total"]],
        on=["cell_id", "period_start"],
        how="left",
    )
    model_df = model_df.dropna(
        subset=["temp_mean", "temp_min", "temp_max", "precip_total"]
    ).copy()
    _step_end(t, f"{len(model_df):,} cell-months × {model_df['cell_id'].nunique():,} cells survive")

    t = _step_start("Merging lag features (species + light)")
    model_df = model_df.merge(lagged_species, on=["cell_id", "period_start"], how="left")
    species_lag_cols = ["moose_lag1", "roe_deer_lag1", "wild_boar_lag1", "fallow_deer_lag1"]
    for col in species_lag_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(0)

    model_df = model_df.merge(lagged_light, on=["cell_id", "period_start"], how="left")
    light_lag_cols = [c for c in lagged_light.columns if c not in ["cell_id", "period_start"]]
    for col in light_lag_cols:
        model_df[col] = model_df[col].fillna(0)
    _step_end(t, f"+8 lag columns; model_df now {len(model_df):,} × {len(model_df.columns)}")

    # === Band E — Features assembly (Row 10) — cell 12 verbatim ===
    t = _step_start("Cyclical month + hunting + rut features (slow, ~30s)")
    model_df = features.add_cyclical_month(model_df)
    model_df = features.build_hunting_features(model_df)
    model_df = features.build_rut_features(model_df)
    model_df_clean = model_df.dropna(subset=FEATURES).copy()
    _step_end(t, f"model_df_clean: {len(model_df_clean):,} × {len(model_df_clean.columns)}")

    # === Band F — Modelling (Rows 11, 12, 14) ===
    hyperparameters = models.load_hyperparameters(hyperparameters_path)

    t = _step_start("Building expanding-window time splits (12-month train, 1-month test)")
    months = sorted(model_df_clean["period_start"].unique())
    splits = models.make_expanding_time_splits(months, min_train_months=12, test_horizon=1)
    _step_end(t, f"{len(splits)} folds")

    t = _step_start("Evaluating LR + RF across all folds — go grab a tea! 🍵 (~2 min)")
    results_df, oof_probs, oof_labels, mean_importance = models.evaluate_time_splits(
        model_df_clean, FEATURES, "risk", splits, hyperparameters,
    )
    rf_auc = results_df.query("model == 'rf'")["auc"].mean()
    lr_auc = results_df.query("model == 'logreg'")["auc"].mean()
    _step_end(t, f"LR mean AUC = {lr_auc:.4f} | RF mean AUC = {rf_auc:.4f}")

    t = _step_start("Fitting final calibrated RF on all data (isotonic, cv=3)")
    rf_final, rf_calibrated = models.fit_final_model(
        model_df_clean, FEATURES, "risk", hyperparameters,
    )
    _step_end(t, "rf_final + rf_calibrated ready ✨")

    # === Band G — Visualisations (Rows 13, 15, 17-20) ===
    t = _step_start("Plotting calibration curve")
    fig_calib, calibration_xy = visualisation.plot_calibration(oof_probs, oof_labels)
    _step_end(t, "calibration plot done")

    t = _step_start("Plotting top-15 feature importances")
    fig_top, top_features_df = visualisation.plot_top_features(mean_importance, top_n=15)
    top_feat = top_features_df.iloc[0]["feature"]
    _step_end(t, f"top feature: {top_feat}")

    t = _step_start("Plotting spatial risk maps (heatmaps)")
    fig_spatial, cell_risk = visualisation.plot_spatial_risk_maps(
        rf_final, model_df_clean, FEATURES, grid_full, joined,
    )
    _step_end(t, f"per-cell risk for {len(cell_risk):,} cells")

    t = _step_start("Plotting ROC curve")
    fig_roc, (fpr, tpr, roc_thresholds) = visualisation.plot_roc(oof_probs, oof_labels)
    _step_end(t, f"{len(fpr)} points")

    t = _step_start("Plotting Precision–Recall curve")
    fig_pr, (precision, recall, pr_thresholds, ap) = visualisation.plot_precision_recall(
        oof_probs, oof_labels,
    )
    _step_end(t, f"average precision = {ap:.4f}")

    t = _step_start("Plotting feature importance by group")
    fig_groups, group_importance_df = visualisation.plot_feature_importance_by_group(
        mean_importance, GROUPS,
    )
    _step_end(t, f"{len(group_importance_df)} feature groups")

    # Cell 19 mutation: add per-row risk_prob to model_df_clean before export.
    model_df_clean["risk_prob"] = rf_final.predict_proba(model_df_clean[FEATURES])[:, 1]

    # === Band H — Export (Row 22) — cell 24 verbatim. FLAG_017 contract surface. ===
    t = _step_start("Exporting CSVs (model_df_clean, feature_importance, model_summary)")
    export_artefacts(model_df_clean, mean_importance, results_df, output_dir)
    _step_end(t, f"3 CSVs written to {output_dir.name}/")

    # === Phase 6 parity dump (opt-in; --dump-parity-arrays only) ===
    if dump_parity_arrays is not None:
        rf_calibrated_preds = rf_calibrated.predict_proba(model_df_clean[FEATURES])[:, 1]
        _dump_parity_arrays(
            dump_parity_arrays,
            oof_probs=oof_probs,
            oof_labels=oof_labels,
            results_df=results_df,
            mean_importance=mean_importance,
            calibration_xy=calibration_xy,
            fpr=fpr,
            tpr=tpr,
            roc_thresholds=roc_thresholds,
            precision=precision,
            recall=recall,
            pr_thresholds=pr_thresholds,
            ap=ap,
            cell_risk=cell_risk,
            group_importance_df=group_importance_df,
            rf_final_preds=model_df_clean["risk_prob"].to_numpy(),
            rf_calibrated_preds=rf_calibrated_preds,
        )

    # === Save joblib models + figures (orchestrator polish; Row 23) ===
    t = _step_start("Saving joblib models + figure PNGs")
    import joblib
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, models_dir / "rf_final.joblib")
    joblib.dump(rf_calibrated, models_dir / "rf_calibrated.joblib")
    fig_calib.savefig(figures_dir / "calibration.png", bbox_inches="tight")
    fig_top.savefig(figures_dir / "top_features.png", bbox_inches="tight")
    fig_spatial.savefig(figures_dir / "spatial_risk_maps.png", bbox_inches="tight")
    fig_roc.savefig(figures_dir / "roc.png", bbox_inches="tight")
    fig_pr.savefig(figures_dir / "precision_recall.png", bbox_inches="tight")
    fig_groups.savefig(figures_dir / "feature_importance_by_group.png", bbox_inches="tight")
    plt.close("all")
    _step_end(t, "2 joblib models + 6 figure PNGs saved")

    # === Per-species pipeline (opt-in; --species flag required) ===
    if species_filter is not None:
        species_to_run = SPECIES_LIST if species_filter == "all" else [species_filter]
        species_output_dir = _REPO_ROOT / "outputs/per_species"
        species_comparison_rows = []

        for sp in species_to_run:
            print(f"\n{'=' * 70}\n  {sp}\n{'=' * 70}")
            sp_label = SPECIES_LABELS[sp]

            df_s, features_s = _build_species_model_df(sp, joined, grid_full, model_df_clean)
            print(f"  threshold printed during build; shape={df_s.shape}, "
                  f"positive_rate={df_s['risk'].mean():.3f}")

            out_dir = species_output_dir / sp
            out_dir.mkdir(parents=True, exist_ok=True)

            months_s = sorted(df_s["period_start"].unique())
            splits_s = models.make_expanding_time_splits(
                months_s, min_train_months=12, test_horizon=1
            )

            res_df, oof_rf, oof_lr, oof_lbl, mean_imp = models.evaluate_time_splits(
                df_s, features_s, "risk", splits_s, hyperparameters,
                return_lr_probs=True,
            )

            summary = (
                res_df.groupby("model")[["auc", "precision", "recall", "f1", "accuracy"]]
                .agg(["mean", "std"])
                .round(3)
            )
            print(summary)

            res_df.to_csv(out_dir / f"cv_results_{sp}.csv", index=False)
            mean_imp.to_csv(out_dir / f"feature_importance_{sp}.csv")
            df_s.to_csv(out_dir / f"model_df_{sp}.csv", index=False)

            # Fit final per-species RF for the risk map
            rf_s = RandomForestClassifier(
                **hyperparameters["random_forest"]
            )
            rf_s.fit(df_s[features_s], df_s["risk"])

            fig, _ = visualisation.plot_species_feature_importance(mean_imp, sp_label)
            fig.savefig(out_dir / f"feature_importance_{sp}.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, _ = visualisation.plot_roc(
                oof_rf, oof_lbl, oof_lr_probs=oof_lr,
                title=f"ROC Curve — {sp_label}",
            )
            fig.savefig(out_dir / f"roc_{sp}.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, _ = visualisation.plot_precision_recall(
                oof_rf, oof_lbl, oof_lr_probs=oof_lr,
                title=f"Precision–Recall — {sp_label}",
            )
            fig.savefig(out_dir / f"pr_{sp}.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, _ = visualisation.plot_calibration(
                oof_rf, oof_lbl, oof_lr_probs=oof_lr,
                title=f"Calibration — {sp_label}",
            )
            fig.savefig(out_dir / f"calibration_{sp}.pdf", bbox_inches="tight")
            plt.close(fig)

            fig, _ = visualisation.plot_species_risk_map(
                grid_full, joined, sp, sp_label, df_s, rf_s, features_s,
            )
            fig.savefig(out_dir / f"risk_map_{sp}.pdf", bbox_inches="tight")
            plt.close(fig)

            for model_name in ["logreg", "rf"]:
                species_comparison_rows.append({
                    "species":      sp,
                    "model":        model_name,
                    "auc_mean":     summary.loc[model_name, ("auc",    "mean")],
                    "auc_std":      summary.loc[model_name, ("auc",    "std")],
                    "f1_mean":      summary.loc[model_name, ("f1",     "mean")],
                    "recall_mean":  summary.loc[model_name, ("recall", "mean")],
                })

        if species_comparison_rows:
            comparison_df = pd.DataFrame(species_comparison_rows)
            comparison_df.to_csv(species_output_dir / "species_comparison.csv", index=False)
            print("\n=== Species comparison ===")
            print(comparison_df.to_string(index=False))

    _banner_end(output_dir, models_dir, figures_dir)


# Resolve all default paths relative to the code/ repo root so the script
# works no matter what the CWD is (including PyCharm's default run-config
# working directory). `__file__` is .../code/scripts/train_final_model.py;
# parent.parent = .../code/.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_argparser() -> "argparse.ArgumentParser":
    """CLI argument parser mirroring main()'s keyword signature.

    Closes FLAG_013 (CLI portability) and FLAG_014 (no Windows path inherited):
    every path is a CLI argument with a sensible repo-root-relative default;
    nothing is hardcoded. Defaults resolve against the code/ repo root via
    ``_REPO_ROOT``, so the script is invocable from any CWD.
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Train the WVC collision-risk model. Defaults resolve against the code/ repo root.",
    )
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data",
                   help=f"Root of the data tree (default: {_REPO_ROOT / 'data'})")
    p.add_argument("--parquet-cache-dir", type=Path, default=_REPO_ROOT / "data/processed/cache",
                   help=f"Parquet feature cache directory (default: {_REPO_ROOT / 'data/processed/cache'})")
    p.add_argument("--weather-cache-dir", type=Path, default=_REPO_ROOT / "notebooks/cache",
                   help=f"Per-station SMHI cache root with subdirs temperature/ and precipitation/ "
                        f"(default: {_REPO_ROOT / 'notebooks/cache'}; matches the Phase 1 baseline)")
    p.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "data/processed",
                   help=f"Where to write CSV exports (default: {_REPO_ROOT / 'data/processed'})")
    p.add_argument("--models-dir", type=Path, default=_REPO_ROOT / "outputs/models",
                   help=f"Where to dump joblib models (default: {_REPO_ROOT / 'outputs/models'})")
    p.add_argument("--figures-dir", type=Path, default=_REPO_ROOT / "outputs/figures",
                   help=f"Where to save figure PNGs (default: {_REPO_ROOT / 'outputs/figures'})")
    p.add_argument("--seed", type=int, default=42, help="Reserved; RF seed comes from hyperparameters.yaml")
    p.add_argument("--species", type=str, default=None, dest="species_filter",
                   metavar="SPECIES",
                   help="Run per-species models. 'all' = all four species; "
                        "or one of: roe_deer, moose, wild_boar, fallow_deer. "
                        "Default: pooled pipeline only.")
    cache_grp = p.add_mutually_exclusive_group()
    cache_grp.add_argument("--use-cache", dest="use_cache", action="store_true", default=True,
                           help="Read parquet feature caches if present (default: True)")
    cache_grp.add_argument("--no-cache", dest="use_cache", action="store_false",
                           help="Recompute parquet feature caches from scratch")
    p.add_argument("--hyperparameters-path", type=Path, default=_REPO_ROOT / "config/hyperparameters.yaml",
                   help=f"YAML hyperparameters file (default: {_REPO_ROOT / 'config/hyperparameters.yaml'})")
    p.add_argument("--dump-parity-arrays", type=Path, default=None, metavar="DIR",
                   help="Phase 6 only: write parity-verification artefacts to DIR/arrays/. "
                        "Default behaviour unchanged when flag is absent.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    main(
        data_dir=args.data_dir,
        parquet_cache_dir=args.parquet_cache_dir,
        weather_cache_dir=args.weather_cache_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
        species_filter=args.species_filter,
        use_cache=args.use_cache,
        hyperparameters_path=args.hyperparameters_path,
        dump_parity_arrays=args.dump_parity_arrays,
    )
