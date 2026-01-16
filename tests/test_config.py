"""
Tests for configuration module.
"""

import pytest
from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    LIGHTGBM_PARAMS,
    RISK_TIERS,
    ensure_dirs,
    get_feature_cols,
)


class TestPaths:
    """Test path configurations."""
    
    def test_project_root_exists(self):
        """Project root should exist."""
        assert PROJECT_ROOT.exists()
    
    def test_data_dir_under_project(self):
        """Data directory should be under project root."""
        assert str(DATA_DIR).startswith(str(PROJECT_ROOT))
    
    def test_path_types(self):
        """All paths should be Path objects."""
        paths = [PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]
        for p in paths:
            assert isinstance(p, Path)


class TestConstants:
    """Test constant configurations."""
    
    def test_target_column(self):
        """Target column should be defined."""
        assert TARGET_COL == "isFraud"
    
    def test_lightgbm_params_has_objective(self):
        """LightGBM params should have objective."""
        assert "objective" in LIGHTGBM_PARAMS
        assert LIGHTGBM_PARAMS["objective"] == "binary"
    
    def test_risk_tiers_ordered(self):
        """Risk tiers should be in descending order."""
        thresholds = list(RISK_TIERS.values())
        assert thresholds == sorted(thresholds, reverse=True)


class TestFunctions:
    """Test configuration functions."""
    
    def test_ensure_dirs_creates_directories(self, tmp_path, monkeypatch):
        """ensure_dirs should create all directories."""
        # This test just verifies the function runs without error
        ensure_dirs()
    
    def test_get_feature_cols_returns_list(self):
        """get_feature_cols should return a list of strings."""
        cols = get_feature_cols()
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all(isinstance(c, str) for c in cols)
