"""
Monitoring module.

Provides:
- Data drift detection using Evidently
- Model performance monitoring
- Alert thresholds and reporting
"""

from src.monitoring.drift_report import generate_drift_report

__all__ = ["generate_drift_report"]
