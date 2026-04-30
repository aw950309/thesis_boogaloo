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
    "Datum":      "datetime",
    "Viltslag":   "species",
    "Län":        "lan",
    "Kommun":     "kommun",
    "Lat WGS84":  "lat",
    "Long WGS84": "lon",
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
