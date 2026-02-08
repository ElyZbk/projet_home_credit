from pathlib import Path

# Racine du projet = dossier parent de src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dossiers d'artefacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Dossier modèles exportés (joblib + metadata)
MODELS_DIR = ARTIFACTS_DIR / "models"
CHAMPION_PIPELINE_PATH = MODELS_DIR / "champion_pipeline.joblib"
THRESHOLD_METADATA_PATH = MODELS_DIR / "champion_metadata.json"

# -------------------- Paramètres métier --------------------

C_FN = 10.0  # faux négatif = mauvais client accepté
C_FP = 1.0   # faux positif = bon client refusé
DEFAULT_THRESHOLD = 0.54

# -------------------- MLflow --------------------
MLFLOW_TRACKING_URI = "file:///Users/ely/Developer/mlruns"
MLFLOW_EXPERIMENT_NAME = "home_credit_scoring"
REGISTERED_MODEL_NAME = "home_credit_xgb_champion"

MODEL_STAGE = "None"
