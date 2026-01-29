"""
API Tests for Fraud Decisioning Platform
=========================================
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.service.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "operational"
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
    
    def test_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "avg_latency_ms" in data
        assert "risk_distribution" in data


class TestScoringEndpoints:
    """Test fraud scoring endpoints."""
    
    def test_score_basic(self, client):
        """Test basic transaction scoring."""
        payload = {
            "TransactionID": "TEST_001",
            "TransactionAmt": 100.00,
            "ProductCD": "W",
            "card4": "visa",
            "DeviceType": "desktop"
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["transaction_id"] == "TEST_001"
        assert 0 <= data["fraud_probability"] <= 1
        assert data["risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert data["recommended_action"] in ["BLOCK", "REVIEW", "CHALLENGE", "APPROVE"]
        assert data["latency_ms"] > 0
    
    def test_score_high_amount(self, client):
        """Test scoring high amount transaction."""
        payload = {
            "TransactionID": "TEST_002",
            "TransactionAmt": 5000.00,
            "hour": 3  # Night transaction
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # High amount + night should have elevated risk
        assert len(data["risk_factors"]) >= 1
    
    def test_score_low_amount(self, client):
        """Test scoring potential card testing."""
        payload = {
            "TransactionID": "TEST_003",
            "TransactionAmt": 1.00
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # Very low amount should be flagged
        assert any("low" in f.lower() or "test" in f.lower() for f in data["risk_factors"])
    
    def test_batch_scoring(self, client):
        """Test batch scoring endpoint."""
        payload = {
            "transactions": [
                {"TransactionID": "BATCH_001", "TransactionAmt": 50.00},
                {"TransactionID": "BATCH_002", "TransactionAmt": 150.00},
                {"TransactionID": "BATCH_003", "TransactionAmt": 500.00}
            ]
        }
        
        response = client.post("/score/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_transactions"] == 3
        assert len(data["results"]) == 3
        assert data["total_latency_ms"] > 0
        assert "high_risk_count" in data


class TestValidation:
    """Test input validation."""
    
    def test_missing_required_field(self, client):
        """Test missing TransactionID."""
        payload = {
            "TransactionAmt": 100.00
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_negative_amount(self, client):
        """Test negative transaction amount."""
        payload = {
            "TransactionID": "TEST_NEG",
            "TransactionAmt": -50.00
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_hour(self, client):
        """Test invalid hour value."""
        payload = {
            "TransactionID": "TEST_HOUR",
            "TransactionAmt": 100.00,
            "hour": 25  # Invalid
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 422


class TestRiskTiers:
    """Test risk tier assignment."""
    
    def test_risk_tier_critical(self, client):
        """Test critical risk classification."""
        # This test assumes the model/heuristics work correctly
        payload = {
            "TransactionID": "RISK_CRIT",
            "TransactionAmt": 100.00
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # Just verify tier is valid
        assert data["risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    def test_action_mapping(self, client):
        """Test risk tier to action mapping."""
        payload = {
            "TransactionID": "RISK_ACTION",
            "TransactionAmt": 100.00
        }
        
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        tier = data["risk_tier"]
        action = data["recommended_action"]
        
        expected = {
            "CRITICAL": "BLOCK",
            "HIGH": "REVIEW",
            "MEDIUM": "CHALLENGE",
            "LOW": "APPROVE"
        }
        
        assert action == expected[tier]


class TestExplainability:
    """Test explanation endpoint."""
    
    def test_explain_prediction(self, client):
        """Test SHAP-based explanation."""
        payload = {
            "TransactionID": "EXPLAIN_001",
            "TransactionAmt": 250.00,
            "hour": 14
        }
        
        response = client.post("/explain/EXPLAIN_001", json=payload)
        
        # May be 503 if model not loaded, 200 if model available
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "fraud_probability" in data
            assert "top_contributors" in data
            assert "explanation" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
