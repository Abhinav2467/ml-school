from pathlib import Path

# Paths assume you run from student_resource/ or set BER_ROOT.
ROOT = Path(__file__).resolve().parents[3]  # ber_starter/
DATA_ROOT = ROOT / "dataset"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "code" / "business_entity_resolution" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Blocking
MAX_CANDIDATES_PER_S1 = 80
TFIDF_TOPK = 30
SAME_COUNTRY_ONLY_IN_BLOCK = True  # still keep country as a *feature*, not a frozen vocab

# Matching
POS_NEG_RATIO = 8
RANDOM_SEED = 42
VAL_FRACTION = 0.2
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 400,
    "reg_lambda": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
}

# Decision: start conservative because F0.5 punishes false merges 2x
DEFAULT_THRESHOLD = 0.72
