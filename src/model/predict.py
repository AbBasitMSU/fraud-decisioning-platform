"""
Prediction module for fraud scoring.

Provides:
- Model loading
- Real-time prediction
- Batch prediction
- Risk tier classification
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, RISK_TIERS, THRESHOLD_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Real-time fraud prediction service."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model (uses default if None)
        """
        self.model_path = model_path or MODEL_FILE
        self.model = None
        self.feature_engineer = None
        self.feature_cols = None
        self.threshold = 0.5
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and feature engineer from disk."""
        logger.info(f"Loading model from {self.model_path}...")
        
        artifacts = joblib.load(self.model_path)
        self.model = artifacts["model"]
        self.feature_engineer = artifacts["feature_engineer"]
        self.feature_cols = artifacts["feature_cols"]
        
        # Load threshold
        if THRESHOLD_FILE.exists():
            with open(THRESHOLD_FILE) as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data.get("threshold", 0.5)
        
        logger.info("Model loaded successfully!")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability.
        
        Args:
            df: Input DataFrame with transaction features
            
        Returns:
            Array of fraud probabilities
        """
        # Apply feature engineering
        df_transformed = self.feature_engineer.transform(df)
        
        # Ensure all required columns exist
        for col in self.feature_cols:
            if col not in df_transformed.columns:
                df_transformed[col] = -999
        
        X = df_transformed[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud label (0/1).
        
        Args:
            df: Input DataFrame with transaction features
            
        Returns:
            Array of predictions (0 = legitimate, 1 = fraud)
        """
        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)
    
    def get_risk_tier(self, probability: float) -> str:
        """
        Map probability to risk tier.
        
        Args:
            probability: Fraud probability
            
        Returns:
            Risk tier string (CRITICAL, HIGH, MEDIUM, LOW)
        """
        for tier, threshold in RISK_TIERS.items():
            if probability >= threshold:
                return tier
        return "MINIMAL"
    
    def get_recommended_action(self, probability: float) -> str:
        """
        Get recommended action based on probability.
        
        Args:
            probability: Fraud probability
            
        Returns:
            Action string (BLOCK, REVIEW, CHALLENGE, APPROVE)
        """
        tier = self.get_risk_tier(probability)
        actions = {
            "CRITICAL": "BLOCK",
            "HIGH": "REVIEW",
            "MEDIUM": "CHALLENGE",
            "LOW": "APPROVE",
            "MINIMAL": "APPROVE",
        }
        return actions.get(tier, "REVIEW")
    
    def score_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Scoring result with probability, tier, and action
        """
        df = pd.DataFrame([transaction])
        proba = self.predict_proba(df)[0]
        
        return {
            "fraud_probability": round(float(proba), 6),
            "risk_tier": self.get_risk_tier(proba),
            "recommended_action": self.get_recommended_action(proba),
            "threshold": self.threshold,
        }
    
    def score_batch(
        self,
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of scoring results
        """
        df = pd.DataFrame(transactions)
        probas = self.predict_proba(df)
        
        results = []
        for i, proba in enumerate(probas):
            results.append({
                "index": i,
                "fraud_probability": round(float(proba), 6),
                "risk_tier": self.get_risk_tier(proba),
                "recommended_action": self.get_recommended_action(proba),
            })
        
        return results
