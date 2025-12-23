"""
Data processing pipeline for IEEE-CIS Fraud Detection dataset.

Loads raw CSV files, merges transaction and identity data,
performs basic cleaning, and saves processed features.
"""

import json
import logging
from typing import Tuple

import pandas as pd

from src.config import (
    ID_COL,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TIME_COL,
    TRAIN_IDENTITY_FILE,
    TRAIN_TRANSACTION_FILE,
    TRAIN_FEATURES_FILE,
    FEATURE_NAMES_FILE,
    VALIDATION_SIZE,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and identity CSV files.
    
    Returns:
        Tuple of (transaction_df, identity_df)
    """
    logger.info(f"Loading transaction data from {TRAIN_TRANSACTION_FILE}")
    transaction_df = pd.read_csv(TRAIN_TRANSACTION_FILE)
    logger.info(f"  Loaded {len(transaction_df):,} transactions")
    
    logger.info(f"Loading identity data from {TRAIN_IDENTITY_FILE}")
    identity_df = pd.read_csv(TRAIN_IDENTITY_FILE)
    logger.info(f"  Loaded {len(identity_df):,} identity records")
    
    return transaction_df, identity_df


def merge_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transaction and identity data on TransactionID.
    
    Args:
        transaction_df: Transaction features
        identity_df: Identity/device features
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging transaction and identity data...")
    merged = transaction_df.merge(identity_df, on=ID_COL, how="left")
    logger.info(f"  Merged shape: {merged.shape}")
    return merged


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype("int8")
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype("int16")
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype("int32")
            else:
                if c_min > -3.4e38 and c_max < 3.4e38:
                    df[col] = df[col].astype("float32")
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory reduced: {start_mem:.1f}MB → {end_mem:.1f}MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


def time_based_split(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    val_size: float = VALIDATION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time to prevent leakage.
    
    Args:
        df: Input DataFrame
        time_col: Column containing time information
        val_size: Fraction of data for validation
        
    Returns:
        Tuple of (train_df, val_df)
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_size))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    return train_df, val_df


def make_dataset() -> None:
    """
    Main pipeline to create processed dataset from raw data.
    """
    logger.info("=" * 60)
    logger.info("MAKING DATASET")
    logger.info("=" * 60)
    
    ensure_dirs()
    
    # Load raw data
    transaction_df, identity_df = load_raw_data()
    
    # Merge
    df = merge_data(transaction_df, identity_df)
    
    # Basic cleaning
    logger.info("Performing basic cleaning...")
    fraud_rate = df[TARGET_COL].mean()
    logger.info(f"  Fraud rate: {fraud_rate:.2%}")
    
    # Reduce memory
    logger.info("Optimizing memory usage...")
    df = reduce_memory_usage(df)
    
    # Time-based split
    logger.info("Splitting data (time-based)...")
    train_df, val_df = time_based_split(df)
    
    # Save feature names
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    with open(FEATURE_NAMES_FILE, "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"  Saved {len(feature_cols)} feature names to {FEATURE_NAMES_FILE}")
    
    # Save processed data
    logger.info(f"Saving processed data to {PROCESSED_DATA_DIR}...")
    train_df.to_parquet(TRAIN_FEATURES_FILE, index=False)
    val_path = PROCESSED_DATA_DIR / "val_features.parquet"
    val_df.to_parquet(val_path, index=False)
    
    logger.info("=" * 60)
    logger.info("DATASET CREATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    make_dataset()
