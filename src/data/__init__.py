"""
Data loading and preprocessing module.

Handles:
- Loading raw Kaggle IEEE-CIS data
- Merging transaction and identity tables
- Train/validation splitting
- Saving processed datasets
"""

from src.data.make_dataset import load_raw_data, make_dataset

__all__ = ["load_raw_data", "make_dataset"]
