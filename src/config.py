"""Configuration constants for the WVC pipeline.

Holds the NVR column-rename mapping (cell 1 of test.ipynb) and CRS constants.
Functions live in the modules that consume these constants; this file is
intentionally constants-only.
"""
from __future__ import annotations

# Maps raw NVR CSV column names to the standardised lower-case names used
# downstream. Source columns confirmed by the runnability audit
# (notes/notes_code/runnability_audit.md).
NVR_COLUMN_RENAME: dict[str, str] = {
    "Datum":          "datetime",
    "Viltslag":       "species",
    "Län":            "lan",
    "Kommun":         "kommun",
    "Lat WGS84":      "lat",
    "Long WGS84":     "lon",
    "Typ av olycka":  "collision_infrastructure",
}

# Maps Swedish NVR infrastructure-type values to lower-case English used
# downstream. Added 2026-05-03 (T10) — drives the collision-type target
# filter in build_species_panel/build_species_model_df.
COLLISION_INFRASTRUCTURE_MAP: dict[str, str] = {
    "Väg":     "road",
    "Järnväg": "rail",
}

# NVR coordinates are WGS84; downstream spatial work runs in SWEREF99 TM.
NVR_SOURCE_CRS: str = "EPSG:4326"
NVR_TARGET_CRS: str = "EPSG:3006"

# Swedish-to-English species labels. Originally defined inline in cell 3 of
# test.ipynb (a DROPPED EDA cell per FLAG_008); migrated here because
# build_lagged_species (cell 8) is the live model path that needs the mapping.
SPECIES_MAP: dict[str, str] = {
    "älg":      "moose",
    "rådjur":   "roe_deer",
    "vildsvin": "wild_boar",
    "dovhjort": "fallow_deer",
}

# 31-element FEATURES list — the model feature set.
# Includes `speedlimit_max` (the 31st feature identified by Phase 1 baseline
# diff against the breakup_map's enumerated 30). Single source of truth per
# FLAG_016 (resolved Phase 2). Imported by scripts/train_final_model.py and
# notebooks/test.ipynb.
FEATURES: list[str] = [
    "road_density", "nearest_road_distance_m",
    "rail_density", "rail_near_10km",
    "fence_density", "fence_near_10km",
    "temp_mean", "temp_min", "temp_max",
    "precip_total",
    "moose_lag1", "roe_deer_lag1", "wild_boar_lag1", "fallow_deer_lag1",
    "night_lag1", "dawn_lag1", "day_lag1", "dusk_lag1",
    "month_sin", "month_cos",
    "moose_hunting_frac", "wild_boar_hunting_frac", "roe_deer_hunting_frac", "fallow_deer_hunting_frac",
    "speedlimit_mean_weighted", "speedlimit_max", "speedlimit_90plus_share",
    "moose_rut_frac", "roe_deer_rut_frac", "wild_boar_rut_frac", "fallow_deer_rut_frac",
]

# Feature-group dict for the per-group importance plot (cell 22; Row 20).
# T5 (2026-05-03): speedlimit_max added to "speed" group. Previously excluded
# to preserve Phase 1 parity baseline; that baseline was regenerated in T10.
# Per-species constants — used by src/grid.py and scripts/train_final_model.py.
SPECIES_LIST: list[str] = ["roe_deer", "moose", "wild_boar", "fallow_deer"]

SPECIES_LABELS: dict[str, str] = {
    "roe_deer":    "Roe deer",
    "moose":       "Moose",
    "wild_boar":   "Wild boar",
    "fallow_deer": "Fallow deer",
}

# Environmental features shared across all per-species models.
# Derived from FEATURES by removing species-prefixed columns (cross-species
# lag1, hunting_frac, rut_frac). Rail and road_density retained — P3 decision
# (wrongly excluded in Amanda's prototype). Result: 19 features.
_SPECIES_PREFIXES = ("moose_", "roe_deer_", "wild_boar_", "fallow_deer_")
BASE_FEATURES_SPECIES: list[str] = [
    f for f in FEATURES if not f.startswith(_SPECIES_PREFIXES)
]


def get_species_features(species_name: str) -> list[str]:
    """Return full feature list for one species: 19 base + 3 species-specific.

    Species-specific features are {species}_lag1, {species}_hunting_frac,
    {species}_rut_frac — derived from the species name string.
    """
    return BASE_FEATURES_SPECIES + [
        f"{species_name}_lag1",
        f"{species_name}_hunting_frac",
        f"{species_name}_rut_frac",
    ]


# No-lag variant: BASE_FEATURES_SPECIES minus all *_lag1 features (15 features).
BASE_FEATURES_SPECIES_NO_LAG: list[str] = [
    f for f in BASE_FEATURES_SPECIES if not f.endswith("_lag1")
]


def get_species_features_no_lag(species_name: str) -> list[str]:
    """Per-species feature list without lag features: 15 base + 2 species-specific.

    Drops night_lag1/dawn_lag1/day_lag1/dusk_lag1 from base and {species}_lag1
    from species-specific. Used for the no-lag variant to expose environmental
    determinants without autocorrelation signal.
    """
    return BASE_FEATURES_SPECIES_NO_LAG + [
        f"{species_name}_hunting_frac",
        f"{species_name}_rut_frac",
    ]


GROUPS: dict[str, list[str]] = {
    "roads":   ["road_density", "nearest_road_distance_m"],
    "fences":  ["fence_density", "fence_near_10km"],
    "weather": ["temp_mean", "temp_min", "temp_max", "precip_total"],
    "species": ["roe_deer_lag1", "moose_lag1", "wild_boar_lag1", "fallow_deer_lag1"],
    "light":   ["night_lag1", "day_lag1", "dawn_lag1", "dusk_lag1"],
    "hunting": [
        "moose_hunting_frac", "wild_boar_hunting_frac",
        "roe_deer_hunting_frac", "fallow_deer_hunting_frac",
    ],
    "rutting": [
        "moose_rut_frac", "roe_deer_rut_frac",
        "wild_boar_rut_frac", "fallow_deer_rut_frac",
    ],
    "speed":   ["speedlimit_mean_weighted", "speedlimit_max", "speedlimit_90plus_share"],
    "rail":    ["rail_density", "rail_near_10km"],
}
