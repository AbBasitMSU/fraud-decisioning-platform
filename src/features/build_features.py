"""
Feature engineering pipeline for fraud detection.

Creates derived features from raw transaction data including:
- Categorical encodings
- Transaction aggregations
- Time-based features
- Interaction features
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import CATEGORICAL_COLS, TARGET_COL, ID_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for fraud detection."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.frequency_maps: Dict[str, Dict] = {}
        self.is_fitted: bool = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit encoders on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self
        """
        logger.info("Fitting feature engineer...")
        
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                # Label encoding
                le = LabelEncoder()
                valid_values = df[col].fillna("MISSING").astype(str)
                le.fit(valid_values)
                self.label_encoders[col] = le
                
                # Frequency encoding
                freq = df[col].value_counts(normalize=True).to_dict()
                self.frequency_maps[col] = freq
        
        self.is_fitted = True
        logger.info(f"  Fitted {len(self.label_encoders)} categorical encoders")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        df = df.copy()
        
        # Apply categorical encodings
        df = self._encode_categoricals(df)
        
        # Create derived features
        df = self._create_amount_features(df)
        df = self._create_time_features(df)
        df = self._create_card_features(df)
        
        # Fill remaining NaNs
        df = self._fill_missing(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label and frequency encoding to categorical columns."""
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                values = df[col].fillna("MISSING").astype(str)
                
                # Handle unseen categories
                known_classes = set(le.classes_)
                values = values.apply(lambda x: x if x in known_classes else "MISSING")
                
                df[f"{col}_encoded"] = le.transform(values)
                
                # Frequency encoding
                freq_map = self.frequency_maps.get(col, {})
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction amount derived features."""
        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
            df["TransactionAmt_decimal"] = (df["TransactionAmt"] % 1).round(2)
            df["TransactionAmt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from TransactionDT."""
        if "TransactionDT" in df.columns:
            # TransactionDT is seconds from reference time
            df["hour"] = (df["TransactionDT"] // 3600) % 24
            df["day"] = (df["TransactionDT"] // 86400) % 7
            df["is_weekend"] = (df["day"] >= 5).astype(int)
            df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        
        return df
    
    def _create_card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create card-related interaction features."""
        # Card-email domain interaction
        if "card1" in df.columns and "P_emaildomain" in df.columns:
            df["card1_P_email"] = (
                df["card1"].astype(str) + "_" + df["P_emaildomain"].fillna("none").astype(str)
            )
        
        # Card-address interaction
        if "card1" in df.columns and "addr1" in df.columns:
            df["card1_addr1"] = (
                df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str)
            )
        
        return df
    
    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill remaining missing values."""
        # Numeric columns: fill with median or -999
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(-999)
        
        # Object columns: fill with "MISSING"
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna("MISSING")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (excluding ID and target)."""
        exclude = [ID_COL, TARGET_COL]
        return [c for c in df.columns if c not in exclude]


def build_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], FeatureEngineer]:
    """
    Build features for training and validation sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        
    Returns:
        Tuple of (train_features, val_features, feature_engineer)
    """
    fe = FeatureEngineer()
    
    logger.info("Building training features...")
    train_features = fe.fit_transform(train_df)
    
    val_features = None
    if val_df is not None:
        logger.info("Building validation features...")
        val_features = fe.transform(val_df)
    
    return train_features, val_features, fe
