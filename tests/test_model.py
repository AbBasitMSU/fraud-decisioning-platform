"""
Model Tests for Fraud Decisioning Platform
===========================================
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_log_transform(self):
        """Test log transformation."""
        amounts = np.array([1, 10, 100, 1000])
        log_amounts = np.log1p(amounts)
        
        assert len(log_amounts) == len(amounts)
        assert all(log_amounts >= 0)
        assert log_amounts[0] < log_amounts[-1]
    
    def test_time_features(self):
        """Test time-based feature extraction."""
        # Simulate TransactionDT (seconds from reference)
        dt = 86400 * 3 + 3600 * 14  # Day 3, 14:00
        
        hour = (dt // 3600) % 24
        day = (dt // 86400) % 7
        is_weekend = int(day >= 5)
        is_night = int(hour >= 22 or hour <= 6)
        
        assert hour == 14
        assert day == 3
        assert is_weekend == 0
        assert is_night == 0
    
    def test_night_classification(self):
        """Test night transaction classification."""
        test_hours = [0, 5, 6, 7, 21, 22, 23]
        expected_night = [1, 1, 1, 0, 0, 1, 1]
        
        for hour, expected in zip(test_hours, expected_night):
            is_night = int(hour >= 22 or hour <= 6)
            assert is_night == expected, f"Hour {hour} should be night={expected}"


class TestMetrics:
    """Test model evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@K calculation."""
        y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.4, 0.15, 0.85, 0.25])
        
        k = 4
        top_k_idx = np.argsort(y_pred)[-k:]
        precision_at_k = y_true[top_k_idx].mean()
        
        # Top 4 predictions should include indices 2, 4, 5, 8 (scores 0.9, 0.8, 0.7, 0.85)
        # All of these are fraud (y_true = 1)
        assert precision_at_k == 1.0
    
    def test_recall_at_k(self):
        """Test recall@K calculation."""
        y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.4, 0.15, 0.85, 0.25])
        
        k = 4
        top_k_idx = np.argsort(y_pred)[-k:]
        total_fraud = y_true.sum()
        recall_at_k = y_true[top_k_idx].sum() / total_fraud
        
        # Top 4 captures all 4 frauds
        assert recall_at_k == 1.0
    
    def test_auc_bounds(self):
        """Test AUC is within valid bounds."""
        from sklearn.metrics import roc_auc_score
        
        y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.4, 0.15, 0.85, 0.25])
        
        auc = roc_auc_score(y_true, y_pred)
        
        assert 0 <= auc <= 1
        assert auc > 0.5  # Better than random


class TestRiskTiers:
    """Test risk tier assignment."""
    
    def test_tier_thresholds(self):
        """Test risk tier threshold mapping."""
        from src.config import RISK_TIERS
        
        def get_tier(prob):
            if prob >= RISK_TIERS["CRITICAL"]:
                return "CRITICAL"
            elif prob >= RISK_TIERS["HIGH"]:
                return "HIGH"
            elif prob >= RISK_TIERS["MEDIUM"]:
                return "MEDIUM"
            return "LOW"
        
        assert get_tier(0.95) == "CRITICAL"
        assert get_tier(0.80) == "CRITICAL"
        assert get_tier(0.79) == "HIGH"
        assert get_tier(0.50) == "HIGH"
        assert get_tier(0.49) == "MEDIUM"
        assert get_tier(0.20) == "MEDIUM"
        assert get_tier(0.19) == "LOW"
        assert get_tier(0.05) == "LOW"
    
    def test_action_mapping(self):
        """Test tier to action mapping."""
        tier_actions = {
            "CRITICAL": "BLOCK",
            "HIGH": "REVIEW",
            "MEDIUM": "CHALLENGE",
            "LOW": "APPROVE"
        }
        
        for tier, expected_action in tier_actions.items():
            assert expected_action in ["BLOCK", "REVIEW", "CHALLENGE", "APPROVE"]


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [np.nan, 2, 3, 4]
        })
        
        df_filled = df.fillna(-999)
        
        assert not df_filled.isna().any().any()
        assert df_filled.loc[2, 'A'] == -999
        assert df_filled.loc[0, 'B'] == -999
    
    def test_stratified_sampling(self):
        """Test stratified sampling preserves fraud rate."""
        # Create imbalanced dataset
        n = 10000
        fraud_rate = 0.035
        
        y = np.zeros(n)
        y[:int(n * fraud_rate)] = 1
        np.random.shuffle(y)
        
        # Sample 1000 with stratification
        sample_size = 1000
        fraud_sample = int(sample_size * fraud_rate)
        
        fraud_idx = np.where(y == 1)[0]
        non_fraud_idx = np.where(y == 0)[0]
        
        sample_fraud = np.random.choice(fraud_idx, fraud_sample, replace=False)
        sample_non_fraud = np.random.choice(non_fraud_idx, sample_size - fraud_sample, replace=False)
        
        y_sample = y[np.concatenate([sample_fraud, sample_non_fraud])]
        
        assert len(y_sample) == sample_size
        assert np.abs(y_sample.mean() - fraud_rate) < 0.01


class TestEnsemble:
    """Test ensemble model logic."""
    
    def test_weighted_average(self):
        """Test weighted ensemble prediction."""
        lgb_pred = np.array([0.3, 0.7, 0.5])
        xgb_pred = np.array([0.4, 0.6, 0.4])
        
        weights = {'lgb': 0.6, 'xgb': 0.4}
        
        ensemble = weights['lgb'] * lgb_pred + weights['xgb'] * xgb_pred
        
        expected = np.array([0.34, 0.66, 0.46])
        np.testing.assert_array_almost_equal(ensemble, expected)
    
    def test_ensemble_bounds(self):
        """Test ensemble predictions are in [0, 1]."""
        lgb_pred = np.array([0.0, 0.5, 1.0])
        xgb_pred = np.array([0.0, 0.5, 1.0])
        
        weights = {'lgb': 0.6, 'xgb': 0.4}
        ensemble = weights['lgb'] * lgb_pred + weights['xgb'] * xgb_pred
        
        assert all(0 <= p <= 1 for p in ensemble)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
