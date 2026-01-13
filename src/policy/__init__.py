"""
Policy and simulation module.

Handles:
- Capacity-aware alert triage
- Expected value calculation
- Fraud-ops simulation
- Queue management
"""

from src.policy.simulate import FraudOpsSimulator

__all__ = ["FraudOpsSimulator"]
