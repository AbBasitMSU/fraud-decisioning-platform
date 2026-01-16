"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.build_features import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample transaction data."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "TransactionID": range(n),
        "TransactionAmt": np.random.exponential(100, n),
        "TransactionDT": np.random.randint(0, 86400 * 30, n),
        "card1": np.random.randint(1000, 9999, n),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", None], n),
        "addr1": np.random.choice([100, 200, 300, None], n),
        "isFraud": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_init(self):
        """FeatureEngineer should initialize correctly."""
        fe = FeatureEngineer()
        assert fe.is_fitted is False
        assert isinstance(fe.label_encoders, dict)
        assert isinstance(fe.frequency_maps, dict)
    
    def test_fit(self, sample_data):
        """Fit should populate encoders."""
        fe = FeatureEngineer()
        fe.fit(sample_data)
        
        assert fe.is_fitted is True
    
    def test_transform_without_fit_raises(self, sample_data):
        """Transform without fit should raise error."""
        fe = FeatureEngineer()
        
        with pytest.raises(ValueError, match="must be fitted"):
            fe.transform(sample_data)
    
    def test_fit_transform(self, sample_data):
        """fit_transform should work correctly."""
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert fe.is_fitted is True
    
    def test_creates_amount_features(self, sample_data):
        """Should create transaction amount features."""
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        
        assert "TransactionAmt_log" in result.columns
        assert "TransactionAmt_is_round" in result.columns
    
    def test_creates_time_features(self, sample_data):
        """Should create time-based features."""
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        
        assert "hour" in result.columns
        assert "day" in result.columns
        assert "is_weekend" in result.columns
    
    def test_handles_missing_values(self, sample_data):
        """Should handle missing values."""
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        
        # Should have no NaN in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not result[col].isna().any(), f"Column {col} has NaN values"
    
    def test_get_feature_names(self, sample_data):
        """Should return feature column names."""
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        
        feature_names = fe.get_feature_names(result)
        
        assert isinstance(feature_names, list)
        assert "TransactionID" not in feature_names
        assert "isFraud" not in feature_names
