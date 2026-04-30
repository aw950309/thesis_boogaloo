<repo_identity>
Bachelor's thesis codebase. Stockholm University, computer and systems science. ML-based prediction of wildlife-vehicle collisions in Sweden using NVR data from 2015 onwards. Species: moose, roe deer, wild boar, fallow deer.

GitHub: thesis_boogaloo. Co-author: Amanda. Be conservative with changes that affect shared paths or imports.
</repo_identity>

<pipeline>
Orchestrated by scripts/train_final_model.py. Six sequential steps:

1. Data Preparation — src/data_prep.py
2. Infrastructure Enrichment — src/infrastructure.py
3. Weather Fetching (SMHI) — src/weather.py
4. Feature Engineering — src/features.py
5. Model Training — src/models.py
6. Visualisation — src/visualisation.py

Configuration in src/config.py. Tests in tests/.
</pipeline>

<conventions>
- Functions only. No classes.
- Parquet for intermediate data caching.
- Joblib for model serialisation.
- Per-station SMHI weather caching at data/processed/weather_cache/.
- Hunting seasons computed as binary flags from date/species/county (län-specific). Do not import hunting calendars from external datasets.
- Species labels standardised Swedish-to-English in data_prep.py.
</conventions>

<open_items>
NVR CSV column names in src/config.py (latitude, longitude, datetime, species, road_id) are assumptions. Verify against actual NVR CSV headers before any pipeline run.
</open_items>

<workflow>
Pull from GitHub before starting work. Co-authored repo.

Run tests in tests/ before pushing changes that touch src/.

Population density and forest-edge features were removed because the source data was not obtainable from Naturvårdsverket / Jägareförbundet / Viltdata. Do not reintroduce them without confirming the data is now available.
</workflow>

<out_of_scope>
- Do not modify the LaTeX thesis. That lives in the sibling paper/ repo, not this one.
- Do not write code outside the established module structure unless explicitly asked.
- Do not introduce classes to refactor existing functions.
</out_of_scope>
