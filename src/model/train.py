"""
LightGBM model training for fraud detection.

Includes:
- Model training with early stopping
- Evaluation metrics (AUC, Precision@K, Recall@K)
- SHAP feature importance
- Model serialization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import (
    ID_COL,
    LIGHTGBM_PARAMS,
    MODEL_FILE,
    MODELS_DIR,
    PRECISION_AT_K_VALUES,
    PROCESSED_DATA_DIR,
    RECALL_TARGET,
    TARGET_COL,
    THRESHOLD_FILE,
    TRAIN_FEATURES_FILE,
    ensure_dirs,
)
from src.features.build_features import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed training and validation data."""
    train_df = pd.read_parquet(TRAIN_FEATURES_FILE)
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val_features.parquet")
    return train_df, val_df


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Calculate precision at K (top-K predictions).
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Precision at K
    """
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true[top_k_idx].mean()


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Calculate recall at K (fraction of frauds in top-K).
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        k: Number of top predictions to consider
        
    Returns:
        Recall at K
    """
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true[top_k_idx].sum() / y_true.sum()


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_recall: float,
) -> float:
    """
    Find probability threshold to achieve target recall.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        target_recall: Desired recall level
        
    Returns:
        Probability threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Find threshold closest to target recall
    idx = np.argmin(np.abs(recall - target_recall))
    if idx < len(thresholds):
        return thresholds[idx]
    return 0.5


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = PRECISION_AT_K_VALUES,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        k_values: List of K values for Precision@K and Recall@K
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_pred),
        "auc_pr": average_precision_score(y_true, y_pred),
    }
    
    for k in k_values:
        if k <= len(y_true):
            metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_pred, k)
            metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_pred, k)
    
    return metrics


def train_model(
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    save_model: bool = True,
) -> Tuple[lgb.LGBMClassifier, FeatureEngineer, Dict[str, float]]:
    """
    Train LightGBM fraud detection model.
    
    Args:
        train_df: Training data (loads from disk if None)
        val_df: Validation data (loads from disk if None)
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (model, feature_engineer, metrics)
    """
    logger.info("=" * 60)
    logger.info("TRAINING FRAUD DETECTION MODEL")
    logger.info("=" * 60)
    
    ensure_dirs()
    
    # Load data if not provided
    if train_df is None or val_df is None:
        logger.info("Loading processed data...")
        train_df, val_df = load_processed_data()
    
    logger.info(f"Training set: {len(train_df):,} samples")
    logger.info(f"Validation set: {len(val_df):,} samples")
    
    # Feature engineering
    logger.info("Engineering features...")
    fe = FeatureEngineer()
    train_df = fe.fit_transform(train_df)
    val_df = fe.transform(val_df)
    
    # Prepare features and target
    feature_cols = [
        c for c in train_df.columns 
        if c not in [ID_COL, TARGET_COL] and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Training fraud rate: {y_train.mean():.2%}")
    logger.info(f"Validation fraud rate: {y_val.mean():.2%}")
    
    # Train model
    logger.info("Training LightGBM model...")
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )
    
    # Predictions
    y_pred = model.predict_proba(X_val)[:, 1]
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(y_val.values, y_pred)
    
    logger.info("-" * 40)
    logger.info("METRICS:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    for k in PRECISION_AT_K_VALUES:
        if f"precision_at_{k}" in metrics:
            logger.info(f"  Precision@{k}: {metrics[f'precision_at_{k}']:.4f}")
            logger.info(f"  Recall@{k}:    {metrics[f'recall_at_{k}']:.4f}")
    logger.info("-" * 40)
    
    # Find optimal threshold
    threshold = find_threshold_for_recall(y_val.values, y_pred, RECALL_TARGET)
    logger.info(f"Threshold for {RECALL_TARGET:.0%} recall: {threshold:.4f}")
    
    # Save model
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {MODEL_FILE}...")
        joblib.dump(
            {
                "model": model,
                "feature_engineer": fe,
                "feature_cols": feature_cols,
            },
            MODEL_FILE,
        )
        
        with open(THRESHOLD_FILE, "w") as f:
            json.dump({"threshold": threshold, "target_recall": RECALL_TARGET}, f)
        
        logger.info("Model saved successfully!")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return model, fe, metrics


if __name__ == "__main__":
    train_model()
