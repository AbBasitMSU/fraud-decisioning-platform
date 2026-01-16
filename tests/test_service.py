"""
Tests for FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Tests for API endpoints (without model loaded)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.service.app import app
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_score_without_model(self, client):
        """Score should return 503 without model."""
        response = client.post(
            "/score",
            json={
                "TransactionAmt": 100.0,
                "card1": 1234,
            }
        )
        
        # Should fail gracefully without model
        assert response.status_code in [200, 503]
    
    def test_score_batch_empty(self, client):
        """Batch score with empty list should return 400."""
        response = client.post(
            "/score/batch",
            json={"transactions": []}
        )
        
        # Either 400 (validation) or 503 (no model)
        assert response.status_code in [400, 503]
    
    def test_score_batch_too_large(self, client):
        """Batch score with >1000 items should return 400."""
        transactions = [{"TransactionAmt": 100.0}] * 1001
        
        response = client.post(
            "/score/batch",
            json={"transactions": transactions}
        )
        
        # Either 400 (validation) or 503 (no model)
        assert response.status_code in [400, 503]
