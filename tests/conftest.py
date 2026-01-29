"""
Pytest Configuration and Fixtures
==================================
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    n = 1000
    
    data = {
        'TransactionID': range(1, n + 1),
        'TransactionAmt': np.random.lognormal(4, 1, n),
        'TransactionDT': np.sort(np.random.randint(0, 86400 * 180, n)),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n),
        'card4': np.random.choice(['visa', 'mastercard', 'discover'], n),
        'card6': np.random.choice(['debit', 'credit'], n),
        'DeviceType': np.random.choice(['desktop', 'mobile'], n),
        'isFraud': np.random.choice([0, 1], n, p=[0.965, 0.035])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'TransactionAmt': np.random.lognormal(4, 1, n),
        'amt_log': np.log1p(np.random.lognormal(4, 1, n)),
        'hour': np.random.randint(0, 24, n),
        'day': np.random.randint(0, 7, n),
        'is_night': np.random.randint(0, 2, n),
        'is_weekend': np.random.randint(0, 2, n),
    })


@pytest.fixture  
def sample_predictions():
    """Generate sample predictions and labels."""
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.choice([0, 1], n, p=[0.965, 0.035])
    y_pred = np.clip(
        np.random.beta(2, 5, n) + 0.3 * y_true + np.random.normal(0, 0.1, n),
        0, 1
    )
    
    return y_true, y_pred
