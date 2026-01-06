"""
Model training and prediction module.

Handles:
- LightGBM model training
- Hyperparameter configuration
- Model evaluation (AUC, Precision@K, Recall@K)
- SHAP explanations
- Model persistence
"""

from src.model.train import train_model
from src.model.predict import FraudPredictor

__all__ = ["train_model", "FraudPredictor"]
