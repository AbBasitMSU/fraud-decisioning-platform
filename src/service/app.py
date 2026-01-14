"""
FastAPI application for real-time fraud scoring.

Endpoints:
- GET /health - Health check
- POST /score - Score single transaction
- POST /score/batch - Score multiple transactions
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import API_TITLE, API_VERSION, LATENCY_SLA_MS, MODEL_FILE

# Global predictor (loaded on startup)
predictor = None


class TransactionRequest(BaseModel):
    """Request schema for single transaction scoring."""
    
    TransactionID: Optional[str] = Field(None, description="Transaction identifier")
    TransactionAmt: float = Field(..., description="Transaction amount", gt=0)
    ProductCD: Optional[str] = Field(None, description="Product category")
    card1: Optional[float] = Field(None, description="Card feature 1")
    card2: Optional[float] = Field(None, description="Card feature 2")
    card3: Optional[float] = Field(None, description="Card feature 3")
    card4: Optional[str] = Field(None, description="Card feature 4")
    card5: Optional[float] = Field(None, description="Card feature 5")
    card6: Optional[str] = Field(None, description="Card feature 6")
    addr1: Optional[float] = Field(None, description="Address feature 1")
    addr2: Optional[float] = Field(None, description="Address feature 2")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    DeviceType: Optional[str] = Field(None, description="Device type")
    DeviceInfo: Optional[str] = Field(None, description="Device info")
    
    class Config:
        extra = "allow"  # Allow additional fields


class BatchRequest(BaseModel):
    """Request schema for batch scoring."""
    
    transactions: List[Dict[str, Any]] = Field(..., description="List of transactions")


class ScoreResponse(BaseModel):
    """Response schema for scoring."""
    
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: str
    recommended_action: str
    latency_ms: float


class BatchScoreResponse(BaseModel):
    """Response schema for batch scoring."""
    
    results: List[Dict[str, Any]]
    total_transactions: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str
    model_loaded: bool
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    
    if MODEL_FILE.exists():
        from src.model.predict import FraudPredictor
        predictor = FraudPredictor()
        print(f"✅ Model loaded from {MODEL_FILE}")
    else:
        print(f"⚠️  Model not found at {MODEL_FILE}")
        print("   Run 'make train' to train the model first")
    
    yield
    
    # Cleanup
    predictor = None


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Real-time fraud scoring API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version=API_VERSION,
    )


@app.post("/score", response_model=ScoreResponse)
async def score_transaction(request: TransactionRequest):
    """
    Score a single transaction for fraud.
    
    Returns fraud probability, risk tier, and recommended action.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' first.",
        )
    
    start_time = time.perf_counter()
    
    # Convert request to dict for scoring
    transaction = request.model_dump(exclude_none=True)
    
    # Score transaction
    result = predictor.score_transaction(transaction)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Log if latency exceeds SLA
    if latency_ms > LATENCY_SLA_MS:
        print(f"⚠️  Latency SLA breach: {latency_ms:.1f}ms > {LATENCY_SLA_MS}ms")
    
    return ScoreResponse(
        transaction_id=request.TransactionID,
        fraud_probability=result["fraud_probability"],
        risk_tier=result["risk_tier"],
        recommended_action=result["recommended_action"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchRequest):
    """
    Score multiple transactions in batch.
    
    More efficient than individual calls for high-volume scoring.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' first.",
        )
    
    if len(request.transactions) == 0:
        raise HTTPException(
            status_code=400,
            detail="No transactions provided",
        )
    
    if len(request.transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 transactions per batch",
        )
    
    start_time = time.perf_counter()
    
    results = predictor.score_batch(request.transactions)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return BatchScoreResponse(
        results=results,
        total_transactions=len(request.transactions),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
