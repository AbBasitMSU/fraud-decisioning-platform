"""
Fraud-ops alert triage simulator.

Simulates a fraud operations team with capacity constraints,
prioritizing alerts by expected value to maximize fraud recovery.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq

import numpy as np
import pandas as pd

from src.config import (
    ALERTS_PER_ANALYST_PER_SHIFT,
    ANALYST_HOURLY_RATE,
    FALSE_POSITIVE_COST,
    NUM_ANALYSTS,
    PROCESSED_DATA_DIR,
    RECOVERY_RATE,
    REPORTS_DIR,
    REVIEW_TIME_MINUTES,
    TARGET_COL,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a fraud alert for triage."""
    
    transaction_id: str
    fraud_probability: float
    transaction_amount: float
    is_fraud: bool
    timestamp: int = 0
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of reviewing this alert."""
        return (
            self.fraud_probability 
            * self.transaction_amount 
            * RECOVERY_RATE
            - (1 - self.fraud_probability) * FALSE_POSITIVE_COST
        )
    
    def __lt__(self, other: "Alert") -> bool:
        """For priority queue (max heap by expected value)."""
        return self.expected_value > other.expected_value


@dataclass
class SimulationResult:
    """Results from a fraud-ops simulation."""
    
    total_alerts: int = 0
    alerts_reviewed: int = 0
    alerts_skipped: int = 0
    
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    fraud_caught_amount: float = 0.0
    fraud_missed_amount: float = 0.0
    review_cost: float = 0.0
    
    @property
    def precision(self) -> float:
        """Precision of reviewed alerts."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall (fraud capture rate)."""
        total_fraud = self.true_positives + self.false_negatives
        if total_fraud == 0:
            return 0.0
        return self.true_positives / total_fraud
    
    @property
    def dollars_saved(self) -> float:
        """Total fraud amount caught and recovered."""
        return self.fraud_caught_amount * RECOVERY_RATE
    
    @property
    def dollars_lost(self) -> float:
        """Fraud amount that slipped through."""
        return self.fraud_missed_amount
    
    @property
    def net_value(self) -> float:
        """Net value = Saved - Lost - Cost."""
        return self.dollars_saved - self.dollars_lost - self.review_cost
    
    def summary(self) -> Dict[str, float]:
        """Return summary metrics."""
        return {
            "total_alerts": self.total_alerts,
            "alerts_reviewed": self.alerts_reviewed,
            "alerts_skipped": self.alerts_skipped,
            "review_rate": self.alerts_reviewed / max(self.total_alerts, 1),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "precision": self.precision,
            "recall": self.recall,
            "fraud_caught_amount": self.fraud_caught_amount,
            "fraud_missed_amount": self.fraud_missed_amount,
            "review_cost": self.review_cost,
            "dollars_saved": self.dollars_saved,
            "dollars_lost": self.dollars_lost,
            "net_value": self.net_value,
        }


class FraudOpsSimulator:
    """
    Simulates fraud operations with capacity constraints.
    
    Prioritizes alerts by expected value and processes them
    up to the available analyst capacity.
    """
    
    def __init__(
        self,
        num_analysts: int = NUM_ANALYSTS,
        shift_capacity: int = ALERTS_PER_ANALYST_PER_SHIFT,
    ):
        self.num_analysts = num_analysts
        self.shift_capacity = shift_capacity
        self.total_capacity = num_analysts * shift_capacity
        
        logger.info(f"FraudOpsSimulator initialized:")
        logger.info(f"  Analysts: {num_analysts}")
        logger.info(f"  Capacity per analyst: {shift_capacity}")
        logger.info(f"  Total daily capacity: {self.total_capacity}")
    
    def simulate(
        self,
        predictions: pd.DataFrame,
        threshold: float = 0.0,
    ) -> SimulationResult:
        """
        Run simulation on predictions.
        
        Args:
            predictions: DataFrame with columns:
                - transaction_id
                - fraud_probability
                - TransactionAmt
                - isFraud (ground truth)
            threshold: Minimum probability to create alert
            
        Returns:
            SimulationResult with metrics
        """
        result = SimulationResult()
        
        # Create alerts from predictions above threshold
        alerts = []
        for _, row in predictions.iterrows():
            if row["fraud_probability"] >= threshold:
                alert = Alert(
                    transaction_id=str(row.get("TransactionID", row.name)),
                    fraud_probability=row["fraud_probability"],
                    transaction_amount=row["TransactionAmt"],
                    is_fraud=bool(row[TARGET_COL]),
                )
                alerts.append(alert)
        
        result.total_alerts = len(alerts)
        logger.info(f"Generated {result.total_alerts} alerts (threshold={threshold:.2f})")
        
        # Priority queue (sorted by expected value)
        heapq.heapify(alerts)
        
        # Process alerts up to capacity
        reviewed = 0
        while alerts and reviewed < self.total_capacity:
            alert = heapq.heappop(alerts)
            reviewed += 1
            
            if alert.is_fraud:
                result.true_positives += 1
                result.fraud_caught_amount += alert.transaction_amount
            else:
                result.false_positives += 1
        
        result.alerts_reviewed = reviewed
        result.alerts_skipped = len(alerts)
        
        # Count missed frauds (skipped alerts that were actually fraud)
        for alert in alerts:
            if alert.is_fraud:
                result.false_negatives += 1
                result.fraud_missed_amount += alert.transaction_amount
            else:
                result.true_negatives += 1
        
        # Calculate review cost
        review_hours = reviewed * REVIEW_TIME_MINUTES / 60
        result.review_cost = review_hours * ANALYST_HOURLY_RATE
        
        return result
    
    def simulate_with_varying_capacity(
        self,
        predictions: pd.DataFrame,
        capacity_levels: List[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    ) -> pd.DataFrame:
        """
        Simulate at different capacity levels.
        
        Args:
            predictions: Prediction DataFrame
            capacity_levels: Multipliers of base capacity
            
        Returns:
            DataFrame with results at each capacity level
        """
        results = []
        base_capacity = self.total_capacity
        
        for multiplier in capacity_levels:
            self.total_capacity = int(base_capacity * multiplier)
            result = self.simulate(predictions)
            summary = result.summary()
            summary["capacity_multiplier"] = multiplier
            summary["total_capacity"] = self.total_capacity
            results.append(summary)
        
        self.total_capacity = base_capacity
        return pd.DataFrame(results)


def run_simulation() -> None:
    """Run fraud-ops simulation on validation data."""
    logger.info("=" * 60)
    logger.info("FRAUD-OPS SIMULATION")
    logger.info("=" * 60)
    
    ensure_dirs()
    
    # Load validation data with predictions
    val_path = PROCESSED_DATA_DIR / "val_features.parquet"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        logger.error("Run 'make make_dataset' and 'make train' first")
        return
    
    val_df = pd.read_parquet(val_path)
    
    # Check if predictions exist
    if "fraud_probability" not in val_df.columns:
        logger.info("No predictions found, generating with random scores for demo...")
        np.random.seed(42)
        # Simulate predictions (higher for actual fraud)
        val_df["fraud_probability"] = np.where(
            val_df[TARGET_COL] == 1,
            np.random.beta(5, 2, len(val_df)),  # Higher for fraud
            np.random.beta(1, 5, len(val_df)),  # Lower for non-fraud
        )
    
    # Initialize simulator
    simulator = FraudOpsSimulator()
    
    # Run simulation at different thresholds
    thresholds = [0.05, 0.10, 0.20, 0.50]
    
    logger.info("\nSimulation Results by Threshold:")
    logger.info("-" * 60)
    
    for thresh in thresholds:
        result = simulator.simulate(val_df, threshold=thresh)
        summary = result.summary()
        
        logger.info(f"\nThreshold: {thresh:.2f}")
        logger.info(f"  Alerts: {summary['total_alerts']:,}")
        logger.info(f"  Reviewed: {summary['alerts_reviewed']:,}")
        logger.info(f"  Precision: {summary['precision']:.2%}")
        logger.info(f"  Recall: {summary['recall']:.2%}")
        logger.info(f"  Fraud Caught: ${summary['fraud_caught_amount']:,.0f}")
        logger.info(f"  Fraud Missed: ${summary['fraud_missed_amount']:,.0f}")
        logger.info(f"  Net Value: ${summary['net_value']:,.0f}")
    
    # Capacity analysis
    logger.info("\n" + "=" * 60)
    logger.info("CAPACITY ANALYSIS")
    logger.info("=" * 60)
    
    capacity_results = simulator.simulate_with_varying_capacity(val_df)
    
    # Save results
    output_path = REPORTS_DIR / "simulation_results.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    capacity_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_simulation()
