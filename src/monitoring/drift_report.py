"""
Data drift detection and reporting using Evidently.

Compares reference (training) data to current (production) data
to detect feature drift and data quality issues.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.config import (
    CATEGORICAL_COLS,
    DRIFT_REPORT_FILE,
    NUMERIC_COLS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COL,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_data: Optional[pd.DataFrame] = None,
    current_data: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    feature_cols: Optional[List[str]] = None,
) -> Path:
    """
    Generate data drift report comparing reference to current data.
    
    Args:
        reference_data: Training/reference DataFrame
        current_data: Production/current DataFrame
        output_path: Path to save HTML report
        feature_cols: Columns to analyze (uses defaults if None)
        
    Returns:
        Path to generated report
    """
    logger.info("=" * 60)
    logger.info("GENERATING DRIFT REPORT")
    logger.info("=" * 60)
    
    ensure_dirs()
    output_path = output_path or DRIFT_REPORT_FILE
    
    # Load data if not provided
    if reference_data is None or current_data is None:
        train_path = PROCESSED_DATA_DIR / "train_features.parquet"
        val_path = PROCESSED_DATA_DIR / "val_features.parquet"
        
        if not train_path.exists() or not val_path.exists():
            logger.error("Processed data not found. Run 'make make_dataset' first.")
            raise FileNotFoundError("Processed data not found")
        
        logger.info("Loading data from disk...")
        reference_data = pd.read_parquet(train_path)
        current_data = pd.read_parquet(val_path)
    
    logger.info(f"Reference data: {len(reference_data):,} samples")
    logger.info(f"Current data: {len(current_data):,} samples")
    
    # Determine feature columns
    if feature_cols is None:
        # Use numeric columns that exist in both datasets
        all_numeric = NUMERIC_COLS + [f"V{i}" for i in range(1, 20)]  # Subset of V cols
        feature_cols = [
            c for c in all_numeric 
            if c in reference_data.columns and c in current_data.columns
        ]
    
    # Get categorical columns that exist
    cat_cols = [
        c for c in CATEGORICAL_COLS[:5]  # Limit for performance
        if c in reference_data.columns and c in current_data.columns
    ]
    
    logger.info(f"Analyzing {len(feature_cols)} numeric features")
    logger.info(f"Analyzing {len(cat_cols)} categorical features")
    
    # Column mapping
    column_mapping = ColumnMapping(
        target=TARGET_COL if TARGET_COL in reference_data.columns else None,
        numerical_features=feature_cols,
        categorical_features=cat_cols,
    )
    
    # Build report
    logger.info("Building drift report...")
    
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    
    # Sample data if too large (for performance)
    max_samples = 50000
    if len(reference_data) > max_samples:
        reference_data = reference_data.sample(n=max_samples, random_state=42)
    if len(current_data) > max_samples:
        current_data = current_data.sample(n=max_samples, random_state=42)
    
    # Select only relevant columns
    cols_to_use = feature_cols + cat_cols
    if TARGET_COL in reference_data.columns:
        cols_to_use.append(TARGET_COL)
    
    reference_subset = reference_data[cols_to_use].copy()
    current_subset = current_data[cols_to_use].copy()
    
    report.run(
        reference_data=reference_subset,
        current_data=current_subset,
        column_mapping=column_mapping,
    )
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    
    logger.info(f"Report saved to {output_path}")
    logger.info("=" * 60)
    logger.info("DRIFT REPORT COMPLETE")
    logger.info("=" * 60)
    
    return output_path


def get_drift_summary(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict:
    """
    Get summary of drift metrics (for programmatic access).
    
    Args:
        reference_data: Reference DataFrame
        current_data: Current DataFrame
        
    Returns:
        Dictionary with drift summary
    """
    # Simple drift check using statistical tests
    from scipy import stats
    
    summary = {
        "total_features": 0,
        "drifted_features": 0,
        "drift_ratio": 0.0,
        "feature_drift": {},
    }
    
    numeric_cols = reference_data.select_dtypes(include=["number"]).columns
    
    for col in numeric_cols:
        if col in current_data.columns:
            summary["total_features"] += 1
            
            ref_values = reference_data[col].dropna()
            cur_values = current_data[col].dropna()
            
            if len(ref_values) > 0 and len(cur_values) > 0:
                # Kolmogorov-Smirnov test
                stat, p_value = stats.ks_2samp(ref_values, cur_values)
                
                is_drifted = p_value < 0.05
                if is_drifted:
                    summary["drifted_features"] += 1
                
                summary["feature_drift"][col] = {
                    "statistic": round(stat, 4),
                    "p_value": round(p_value, 4),
                    "is_drifted": is_drifted,
                }
    
    if summary["total_features"] > 0:
        summary["drift_ratio"] = summary["drifted_features"] / summary["total_features"]
    
    return summary


if __name__ == "__main__":
    generate_drift_report()
