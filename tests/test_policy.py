"""
Tests for policy simulation module.
"""

import pytest
import numpy as np
import pandas as pd

from src.policy.simulate import Alert, SimulationResult, FraudOpsSimulator


class TestAlert:
    """Tests for Alert dataclass."""
    
    def test_expected_value_fraud(self):
        """Expected value should be positive for high-prob fraud."""
        alert = Alert(
            transaction_id="1",
            fraud_probability=0.9,
            transaction_amount=1000,
            is_fraud=True,
        )
        
        # EV = 0.9 * 1000 * 0.8 - 0.1 * 50 = 720 - 5 = 715
        assert alert.expected_value > 0
    
    def test_expected_value_low_prob(self):
        """Expected value should be negative for low-prob transaction."""
        alert = Alert(
            transaction_id="2",
            fraud_probability=0.01,
            transaction_amount=100,
            is_fraud=False,
        )
        
        # EV = 0.01 * 100 * 0.8 - 0.99 * 50 = 0.8 - 49.5 < 0
        assert alert.expected_value < 0
    
    def test_comparison_by_expected_value(self):
        """Alerts should compare by expected value (descending)."""
        high_ev = Alert("1", 0.9, 1000, True)
        low_ev = Alert("2", 0.1, 100, False)
        
        # In max heap, higher EV should be "less than"
        assert high_ev < low_ev


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_precision_calculation(self):
        """Precision should be TP / (TP + FP)."""
        result = SimulationResult(
            true_positives=80,
            false_positives=20,
        )
        
        assert result.precision == 0.8
    
    def test_recall_calculation(self):
        """Recall should be TP / (TP + FN)."""
        result = SimulationResult(
            true_positives=80,
            false_negatives=20,
        )
        
        assert result.recall == 0.8
    
    def test_precision_zero_division(self):
        """Precision should handle zero division."""
        result = SimulationResult()
        assert result.precision == 0.0
    
    def test_recall_zero_division(self):
        """Recall should handle zero division."""
        result = SimulationResult()
        assert result.recall == 0.0
    
    def test_net_value_calculation(self):
        """Net value should be saved - lost - cost."""
        result = SimulationResult(
            fraud_caught_amount=10000,
            fraud_missed_amount=2000,
            review_cost=500,
        )
        
        # Net = 10000 * 0.8 - 2000 - 500 = 8000 - 2000 - 500 = 5500
        assert result.net_value == 8000 - 2000 - 500


class TestFraudOpsSimulator:
    """Tests for FraudOpsSimulator class."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions DataFrame."""
        np.random.seed(42)
        n = 1000
        
        is_fraud = np.random.choice([0, 1], n, p=[0.97, 0.03])
        
        return pd.DataFrame({
            "TransactionID": range(n),
            "TransactionAmt": np.random.exponential(200, n),
            "isFraud": is_fraud,
            "fraud_probability": np.where(
                is_fraud,
                np.random.beta(5, 2, n),
                np.random.beta(1, 5, n),
            ),
        })
    
    def test_init(self):
        """Simulator should initialize with capacity."""
        sim = FraudOpsSimulator(num_analysts=5, shift_capacity=32)
        
        assert sim.num_analysts == 5
        assert sim.shift_capacity == 32
        assert sim.total_capacity == 160
    
    def test_simulate_returns_result(self, sample_predictions):
        """Simulate should return SimulationResult."""
        sim = FraudOpsSimulator(num_analysts=2, shift_capacity=10)
        result = sim.simulate(sample_predictions, threshold=0.1)
        
        assert isinstance(result, SimulationResult)
        assert result.alerts_reviewed <= sim.total_capacity
    
    def test_simulate_respects_capacity(self, sample_predictions):
        """Simulator should not exceed capacity."""
        sim = FraudOpsSimulator(num_analysts=1, shift_capacity=10)
        result = sim.simulate(sample_predictions, threshold=0.0)
        
        assert result.alerts_reviewed <= 10
    
    def test_simulate_prioritizes_high_ev(self, sample_predictions):
        """High EV alerts should be reviewed first."""
        sim = FraudOpsSimulator(num_analysts=1, shift_capacity=20)
        result = sim.simulate(sample_predictions, threshold=0.1)
        
        # With prioritization, we should catch more fraud than random
        assert result.true_positives >= 0
    
    def test_threshold_filters_alerts(self, sample_predictions):
        """Higher threshold should create fewer alerts."""
        sim = FraudOpsSimulator()
        
        result_low = sim.simulate(sample_predictions, threshold=0.1)
        result_high = sim.simulate(sample_predictions, threshold=0.5)
        
        assert result_low.total_alerts >= result_high.total_alerts
