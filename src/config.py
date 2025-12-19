"""
Configuration module for Fraud Decisioning Platform.

Centralizes all paths, constants, and hyperparameters.
"""

from pathlib import Path
from typing import List

# =============================================================================
# PATHS
# =============================================================================

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw data files (Kaggle IEEE-CIS)
TRAIN_TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_FILE = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_FILE = RAW_DATA_DIR / "test_identity.csv"

# Processed data files
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.parquet"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.parquet"
FEATURE_NAMES_FILE = PROCESSED_DATA_DIR / "feature_names.json"

# Model files
MODEL_FILE = MODELS_DIR / "fraud_model.joblib"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Target column
TARGET_COL = "isFraud"

# Transaction ID column
ID_COL = "TransactionID"

# Train/validation split
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Time-based split column (for proper temporal validation)
TIME_COL = "TransactionDT"

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Categorical columns to encode
CATEGORICAL_COLS: List[str] = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
]

# Numeric columns to keep
NUMERIC_COLS: List[str] = [
    "TransactionAmt",
    "dist1", "dist2",
]

# V columns (anonymous features from Vesta)
V_COLS: List[str] = [f"V{i}" for i in range(1, 340)]

# C columns (counting features)
C_COLS: List[str] = [f"C{i}" for i in range(1, 15)]

# D columns (time delta features)
D_COLS: List[str] = [f"D{i}" for i in range(1, 16)]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Threshold tuning
PRECISION_AT_K_VALUES = [100, 500, 1000, 5000]
RECALL_TARGET = 0.80

# =============================================================================
# POLICY / SIMULATION CONFIGURATION
# =============================================================================

# Fraud-ops team configuration
NUM_ANALYSTS = 10
ANALYST_SHIFT_HOURS = 8
REVIEW_TIME_MINUTES = 15

# Capacity (alerts per analyst per shift)
ALERTS_PER_ANALYST_PER_SHIFT = int(ANALYST_SHIFT_HOURS * 60 / REVIEW_TIME_MINUTES)

# Expected value parameters
RECOVERY_RATE = 0.80  # % of fraud amount recovered if caught
FALSE_POSITIVE_COST = 50.0  # $ cost of reviewing non-fraud
ANALYST_HOURLY_RATE = 35.0  # $ per hour

# Risk tier thresholds
RISK_TIERS = {
    "CRITICAL": 0.80,
    "HIGH": 0.50,
    "MEDIUM": 0.20,
    "LOW": 0.05,
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Fraud Decisioning Platform API"
API_VERSION = "1.0.0"

# Latency SLA (milliseconds)
LATENCY_SLA_MS = 100

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

# Drift detection
DRIFT_THRESHOLD = 0.05  # p-value threshold for statistical tests
DRIFT_REPORT_FILE = REPORTS_DIR / "drift_report.html"

# Reference window (for drift comparison)
REFERENCE_WINDOW_DAYS = 7

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for dir_path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_feature_cols() -> List[str]:
    """Return list of all feature columns."""
    return NUMERIC_COLS + CATEGORICAL_COLS + V_COLS + C_COLS + D_COLS
