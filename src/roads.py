"""
Compatibility shim — re-exports from src.infrastructure.

Background: the real module was renamed `src/roads.py` → `src/infrastructure.py` in Phase 4
(AD-01) because the file handles roads, rail, fences, and speed limits — `infrastructure`
is the accurate name. The notebook `code/notebooks/test.ipynb` cell 0 still does
`from roads import (...)`; this shim keeps that import working so the notebook stays
runnable through Phase 9 and the re-launch (H8 strengthened). All real logic lives in
`src/infrastructure.py`.

Disposition: retire this shim only after the notebook is itself disposed of (Phase 8 /
post-Phase-9 decision per FLAG_004) and no other consumer remains.
"""
from src.infrastructure import (
    build_road_features,
    load_roads_for_study_area,
    build_linear_features,
    load_linear_layer_for_study_area,
    build_speedlimit_features,
)

__all__ = [
    "build_road_features",
    "load_roads_for_study_area",
    "build_linear_features",
    "load_linear_layer_for_study_area",
    "build_speedlimit_features",
]
