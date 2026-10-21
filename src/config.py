"""Central configuration for the Fraud Detection Pipeline.

Contains reproducibility seeds, directory paths, train/val/test split ratios,
and business cost parameters as specified in PRD.md.
"""

from pathlib import Path

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------
# File System Paths
# ---------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
DOCS_DIR: Path = BASE_DIR / "docs"

# ---------------------------------------------------------
# Dataset Specifics (IEEE-CIS Fraud Detection)
# ---------------------------------------------------------
ID_COL: str = "TransactionID"
TIME_COL: str = "TransactionDT"
TARGET_COL: str = "isFraud"

# Raw file names expected from Kaggle
TRAIN_TRANSACTION_FILE: str = "train_transaction.csv"
TRAIN_IDENTITY_FILE: str = "train_identity.csv"

# ---------------------------------------------------------
# Validation Strategy [CONFIRM]
# Time-based split: Earliest 70% train, next 15% val, latest 15% test
# ---------------------------------------------------------
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ---------------------------------------------------------
# Business Cost Parameters [CONFIRM]
# Total Cost = (FN * COST_FN) + (FP * COST_FP)
# Missed fraud (FN) is assumed 10x costlier than a false alarm (FP).
# ---------------------------------------------------------
COST_FN: float = 10.0
COST_FP: float = 1.0

# Fixed precision target for secondary evaluation
FIXED_PRECISION_TARGET: float = 0.80
