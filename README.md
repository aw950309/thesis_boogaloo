# 🦌 Wildlife-Vehicle Collision Prediction

Bachelor thesis project analyzing wildlife-vehicle collision (WVC) risk zones in Sweden using machine learning approaches.

**Authors:** Amanda Stephenson & Alex Wagner

## 📖 Project Overview

This research project aims to predict high-risk wildlife collision zones in Sweden by analyzing temporal patterns, meteorological conditions, and species-specific behavior. The methodology focuses on four main species (moose, roe deer, wild boar, fallow deer) which account for 91% of wildlife collisions.

## 🔄 For Existing Collaborators (Amanda & Alex)

Since you already have the repository cloned, skip the fresh install section.

### One-Time Setup (After Pulling New Structure)

**Step 1: Pull the latest changes (GitHub Desktop)**
1. Open GitHub Desktop
2. Click "Fetch origin" (top right)
3. If changes available, click "Pull origin"

**Step 2: Set up the environment (Terminal in PyCharm)**
```bash
# Activate your existing venv (if not already active)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install the project as an editable package (ONE TIME ONLY)
pip install -e .

# Verify it works
python -c "from src.config import SPECIES; print('Setup successful!')"
```

### Daily Workflow

**Pulling changes (GitHub Desktop):**
1. Open GitHub Desktop
2. Click "Fetch origin"
3. If changes available, click "Pull origin"
4. Changes are immediately available - no reinstall needed!

**Pushing changes (GitHub Desktop):**
1. Your changes appear in the "Changes" tab
2. Write a descriptive commit message
3. Click "Commit to main"
4. Click "Push origin"

> **Note:** The `pip install -e .` command only needs to run once per computer. After that, code changes sync through git automatically.

---

## 🚀 Fresh Install (New Computer or New Collaborator)

### Prerequisites
- Python 3.8+
- Git or GitHub Desktop

### Setup

1. **Clone the repository**
   
   **Using GitHub Desktop:**
   - File → Clone Repository → URL tab
   - Paste the repository URL
   - Choose local path and click "Clone"
   
   **Or using terminal:**
   ```bash
   git clone [repository-url]
   cd thesis_boogaloo
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install the project in editable mode**
   ```bash
   pip install -e .
   ```

4. **Verify installation**
   ```bash
   python -c "from src.config import SPECIES; print('Setup successful!')"
   ```

## 📁 Project Structure

```
thesis_boogaloo/
├── data/                    # Data files (excluded from git)
│   ├── raw/                 # Original NVR and weather data
│   └── processed/           # Cleaned data ready for ML models
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_data_cleaning.ipynb       # NVR cleaning + weather integration
│   ├── 02_exploratory_analysis.ipynb # Temporal patterns, species distributions
│   ├── 03_model_training.ipynb       # Feature engineering + ML models
│   └── 04_results_visualisation.ipynb # Figures for thesis
├── src/                     # Core reusable logic
│   ├── __init__.py
│   ├── config.py            # Project constants and parameters
│   ├── data_prep.py         # Data cleaning and preprocessing
│   ├── weather.py           # SMHI station matching and temperature (This is seperate because it calls its own API)
│   ├── features.py          # Feature engineering (seasons, time)
│   ├── models.py            # ML model definitions
│   └── visualization.py     # Plotting and visualization
├── outputs/                 # Generated results
│   ├── models/              # Trained model files (.pkl)
│   └── figures/             # Charts and plots for thesis
├── scripts/                 # Automation scripts
│   └── train_final_model.py
├── config/                  # Configuration files
│   └── hyperparameters.yaml # ML hyperparameters (tree depth, learning rate
├── tests/                   # Unit tests (optional)
│   └── test_data_prep.py
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project config & dependencies
├── README.md                # This file
└── venv/                    # Virtual environment (excluded from git)
```

## 🔬 Methodology

- **Species Focus**: Moose, roe deer, wild boar, fallow deer
- **Temporal Analysis**: Day/night/dawn/dusk patterns, seasonal variations, rutting periods
- **Environmental Factors**: Temperature, weather conditions
- **ML Approaches**: Random Forest and linear models
- **Framework**: Design Science Research methodology

## 🛠️ Development Workflow

### For Notebooks
```python
# Import from src package
from src.data_prep import load_nvr_data
from src.config import SPECIES, RUTTING_PERIODS
from src.features import encode_temporal_features
```

### For Scripts
- Place heavy computation scripts in `scripts/`
- Use proper `main()` functions with `if __name__ == "__main__":`

### Code Organization
- **`src/`**: Reusable functions and classes
- **`notebooks/`**: Data exploration and visualization
- **`outputs/`**: Generated models and figures (figures tracked in git, large models ignored)

## 📊 Data Sources

- **NVR (Nationella Viltolycksrådet)**: Wildlife collision data (2015+)
- **SMHI**: Meteorological data
- **Spatial Data**: Swedish road network and geographical boundaries

## 🤝 Collaboration

This project uses an editable install approach for seamless collaboration:

1. **Pull latest changes**: Use GitHub Desktop or `git pull`
2. **Code changes sync automatically** - no need to reinstall after pulling
3. **Consistent imports** - `from src.module import function` works on all computers

## 📈 Current Status

- [x] Project structure established
- [ ] Data preprocessing pipeline
- [ ] Feature engineering implementation
- [ ] Model training and evaluation
- [ ] Spatial visualization
- [ ] Results analysis

## 📝 Notes

- Data files are excluded from version control (see `.gitignore`)
- Large model files (`.pkl`) are not tracked in git
- Use `notebooks/` for exploration, `src/` for reusable code
