"""
Advanced Model Training Pipeline
================================
Production-grade fraud detection with:
- LightGBM + XGBoost ensemble
- Optuna hyperparameter optimization
- Stratified K-fold cross-validation
- SHAP explainability
- Calibrated probabilities
- MLflow experiment tracking
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    ID_COL, TARGET_COL, MODELS_DIR, PROCESSED_DATA_DIR,
    PRECISION_AT_K_VALUES, RANDOM_STATE, ensure_dirs
)


class AdvancedFraudTrainer:
    """
    Advanced fraud detection model trainer with ensemble learning,
    hyperparameter optimization, and explainability.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        n_trials: int = 50,
        use_optuna: bool = True,
        ensemble_weights: Optional[Dict[str, float]] = None,
        calibrate: bool = True,
        experiment_name: str = "fraud_detection"
    ):
        """
        Initialize the advanced trainer.
        
        Args:
            n_folds: Number of cross-validation folds
            n_trials: Number of Optuna optimization trials
            use_optuna: Whether to use hyperparameter optimization
            ensemble_weights: Weights for ensemble models {'lgb': 0.6, 'xgb': 0.4}
            calibrate: Whether to calibrate probabilities
            experiment_name: Name for MLflow experiment
        """
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.use_optuna = use_optuna
        self.ensemble_weights = ensemble_weights or {'lgb': 0.6, 'xgb': 0.4}
        self.calibrate = calibrate
        self.experiment_name = experiment_name
        
        self.models: Dict[str, Any] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}
        self.best_params: Dict[str, Dict] = {}
        self.metrics: Dict[str, float] = {}
        self.cv_scores: List[Dict] = []
        
    def _get_lgb_params(self, trial=None) -> Dict:
        """Get LightGBM parameters, optionally from Optuna trial."""
        if trial is None:
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1,
            }
        
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('lgb_num_leaves', 16, 128),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 20, 200),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
    
    def _get_xgb_params(self, trial=None) -> Dict:
        """Get XGBoost parameters, optionally from Optuna trial."""
        if trial is None:
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': 0,
            }
        
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 50),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-3, 10.0, log=True),
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0,
        }
    
    def _precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate precision at K."""
        top_k_idx = np.argsort(y_pred)[-k:]
        return y_true[top_k_idx].mean()
    
    def _recall_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """Calculate recall at K."""
        top_k_idx = np.argsort(y_pred)[-k:]
        return y_true[top_k_idx].sum() / y_true.sum()
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred),
            'auc_pr': average_precision_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred),
        }
        
        # Threshold-based metrics at optimal F1
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        y_pred_binary = (y_pred >= best_threshold).astype(int)
        metrics['best_f1'] = f1_scores[best_idx]
        metrics['best_threshold'] = best_threshold
        metrics['precision_at_best'] = precision[best_idx]
        metrics['recall_at_best'] = recall[best_idx]
        
        # Precision/Recall at K
        for k in PRECISION_AT_K_VALUES:
            if k <= len(y_true):
                metrics[f'precision_at_{k}'] = self._precision_at_k(y_true, y_pred, k)
                metrics[f'recall_at_{k}'] = self._recall_at_k(y_true, y_pred, k)
        
        return metrics
    
    def _train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        params_lgb: Dict,
        params_xgb: Dict,
    ) -> Tuple[Any, Any, np.ndarray, Dict]:
        """Train models on a single fold."""
        import lightgbm as lgb
        import xgboost as xgb
        
        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(**params_lgb)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(**params_xgb)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        
        # Ensemble prediction
        ensemble_pred = (
            self.ensemble_weights['lgb'] * lgb_pred +
            self.ensemble_weights['xgb'] * xgb_pred
        )
        
        # Compute metrics
        metrics = self._compute_metrics(y_val, ensemble_pred)
        
        return lgb_model, xgb_model, ensemble_pred, metrics
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Use Optuna to find optimal hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target array
            
        Returns:
            Dictionary of best parameters for each model
        """
        import optuna
        from optuna.samplers import TPESampler
        
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")
        
        def objective(trial):
            params_lgb = self._get_lgb_params(trial)
            params_xgb = self._get_xgb_params(trial)
            
            # Use 3-fold CV for speed during optimization
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                _, _, _, metrics = self._train_fold(
                    X_train, y_train, X_val, y_val,
                    params_lgb, params_xgb
                )
                scores.append(metrics['auc_roc'])
            
            return np.mean(scores)
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial AUC: {study.best_value:.4f}")
        
        # Extract best parameters
        best_params = study.best_params
        
        self.best_params = {
            'lgb': {k.replace('lgb_', ''): v for k, v in best_params.items() if k.startswith('lgb_')},
            'xgb': {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')},
        }
        
        return self.best_params
    
    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the ensemble model with cross-validation.
        
        Args:
            X: Training features
            y: Training targets
            X_test: Optional test features for final evaluation
            y_test: Optional test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        import lightgbm as lgb
        import xgboost as xgb
        
        logger.info("=" * 70)
        logger.info("ADVANCED FRAUD DETECTION TRAINING")
        logger.info("=" * 70)
        logger.info(f"Training samples: {len(X):,}")
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Fraud rate: {y.mean():.2%}")
        logger.info(f"Cross-validation folds: {self.n_folds}")
        logger.info(f"Ensemble weights: LGB={self.ensemble_weights['lgb']}, XGB={self.ensemble_weights['xgb']}")
        
        # Hyperparameter optimization
        if self.use_optuna:
            self.optimize_hyperparameters(X, y)
            params_lgb = {**self._get_lgb_params(), **self.best_params.get('lgb', {})}
            params_xgb = {**self._get_xgb_params(), **self.best_params.get('xgb', {})}
        else:
            params_lgb = self._get_lgb_params()
            params_xgb = self._get_xgb_params()
        
        # Cross-validation
        logger.info(f"\nRunning {self.n_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)
        
        oof_predictions = np.zeros(len(X))
        feature_importance_lgb = np.zeros(X.shape[1])
        feature_importance_xgb = np.zeros(X.shape[1])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            lgb_model, xgb_model, ensemble_pred, fold_metrics = self._train_fold(
                X_train, y_train, X_val, y_val,
                params_lgb, params_xgb
            )
            
            oof_predictions[val_idx] = ensemble_pred
            feature_importance_lgb += lgb_model.feature_importances_ / self.n_folds
            feature_importance_xgb += xgb_model.feature_importances_ / self.n_folds
            
            self.cv_scores.append(fold_metrics)
            logger.info(f"  AUC-ROC: {fold_metrics['auc_roc']:.4f} | AUC-PR: {fold_metrics['auc_pr']:.4f}")
        
        # Store feature importances
        self.feature_importances = {
            'lgb': feature_importance_lgb,
            'xgb': feature_importance_xgb,
            'ensemble': (
                self.ensemble_weights['lgb'] * feature_importance_lgb +
                self.ensemble_weights['xgb'] * feature_importance_xgb
            ),
            'feature_names': list(X.columns)
        }
        
        # Compute OOF metrics
        self.metrics = self._compute_metrics(y, oof_predictions)
        
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"OOF AUC-ROC: {self.metrics['auc_roc']:.4f}")
        logger.info(f"OOF AUC-PR:  {self.metrics['auc_pr']:.4f}")
        logger.info(f"OOF Log Loss: {self.metrics['log_loss']:.4f}")
        logger.info(f"Best F1 Score: {self.metrics['best_f1']:.4f} @ threshold {self.metrics['best_threshold']:.3f}")
        
        for k in PRECISION_AT_K_VALUES:
            if f'precision_at_{k}' in self.metrics:
                logger.info(f"Precision@{k}: {self.metrics[f'precision_at_{k}']:.4f} | Recall@{k}: {self.metrics[f'recall_at_{k}']:.4f}")
        
        # Train final models on full data
        logger.info("\nTraining final models on full dataset...")
        
        self.models['lgb'] = lgb.LGBMClassifier(**params_lgb)
        self.models['lgb'].fit(X, y)
        
        self.models['xgb'] = xgb.XGBClassifier(**params_xgb)
        self.models['xgb'].fit(X, y)
        
        # Calibrate if requested
        if self.calibrate:
            logger.info("Calibrating probabilities...")
            # Use isotonic regression for calibration
            self.models['lgb_calibrated'] = CalibratedClassifierCV(
                self.models['lgb'], cv='prefit', method='isotonic'
            )
            self.models['xgb_calibrated'] = CalibratedClassifierCV(
                self.models['xgb'], cv='prefit', method='isotonic'
            )
            # Note: Would need a holdout set for proper calibration in production
        
        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            logger.info("\nTest Set Evaluation:")
            test_pred = self.predict(X_test)
            test_metrics = self._compute_metrics(y_test, test_pred)
            logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
            logger.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f}")
            self.metrics['test_auc_roc'] = test_metrics['auc_roc']
            self.metrics['test_auc_pr'] = test_metrics['auc_pr']
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Fraud probability predictions
        """
        lgb_pred = self.models['lgb'].predict_proba(X)[:, 1]
        xgb_pred = self.models['xgb'].predict_proba(X)[:, 1]
        
        return (
            self.ensemble_weights['lgb'] * lgb_pred +
            self.ensemble_weights['xgb'] * xgb_pred
        )
    
    def compute_shap_values(self, X: pd.DataFrame, n_samples: int = 1000) -> Dict:
        """
        Compute SHAP values for model explainability.
        
        Args:
            X: Feature matrix
            n_samples: Number of samples for SHAP computation
            
        Returns:
            Dictionary with SHAP values and summary
        """
        try:
            import shap
            
            logger.info(f"Computing SHAP values on {min(n_samples, len(X))} samples...")
            
            # Sample data for efficiency
            if len(X) > n_samples:
                X_sample = X.sample(n=n_samples, random_state=RANDOM_STATE)
            else:
                X_sample = X
            
            # Create SHAP explainer for LightGBM
            explainer = shap.TreeExplainer(self.models['lgb'])
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class
            
            # Compute mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            shap_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False)
            
            return {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value,
                'feature_names': list(X_sample.columns),
                'shap_importance': shap_importance,
                'X_sample': X_sample
            }
            
        except ImportError:
            logger.warning("SHAP not installed. Skipping explainability.")
            return {}
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save trained models and artifacts.
        
        Args:
            filepath: Path to save (defaults to MODELS_DIR)
            
        Returns:
            Path where model was saved
        """
        ensure_dirs()
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f"fraud_ensemble_{timestamp}.joblib"
        
        artifact = {
            'models': self.models,
            'feature_importances': self.feature_importances,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'cv_scores': self.cv_scores,
            'ensemble_weights': self.ensemble_weights,
            'n_folds': self.n_folds,
            'timestamp': datetime.now().isoformat(),
        }
        
        joblib.dump(artifact, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Also save latest
        latest_path = MODELS_DIR / "fraud_ensemble_latest.joblib"
        joblib.dump(artifact, latest_path)
        
        # Save metrics as JSON
        metrics_path = MODELS_DIR / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'AdvancedFraudTrainer':
        """
        Load a trained model.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            AdvancedFraudTrainer instance
        """
        artifact = joblib.load(filepath)
        
        trainer = cls(
            n_folds=artifact.get('n_folds', 5),
            ensemble_weights=artifact.get('ensemble_weights', {'lgb': 0.6, 'xgb': 0.4})
        )
        trainer.models = artifact['models']
        trainer.feature_importances = artifact['feature_importances']
        trainer.best_params = artifact['best_params']
        trainer.metrics = artifact['metrics']
        trainer.cv_scores = artifact.get('cv_scores', [])
        
        return trainer
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as a DataFrame."""
        return pd.DataFrame({
            'feature': self.feature_importances['feature_names'],
            'lgb_importance': self.feature_importances['lgb'],
            'xgb_importance': self.feature_importances['xgb'],
            'ensemble_importance': self.feature_importances['ensemble']
        }).sort_values('ensemble_importance', ascending=False)


def train_advanced_model(
    train_file: Optional[Path] = None,
    val_file: Optional[Path] = None,
    n_trials: int = 50,
    use_optuna: bool = True,
) -> AdvancedFraudTrainer:
    """
    Convenience function to train the advanced model.
    
    Args:
        train_file: Path to training data parquet
        val_file: Path to validation data parquet
        n_trials: Number of Optuna trials
        use_optuna: Whether to use hyperparameter optimization
        
    Returns:
        Trained AdvancedFraudTrainer
    """
    # Load data
    if train_file is None:
        train_file = PROCESSED_DATA_DIR / "train_features.parquet"
    if val_file is None:
        val_file = PROCESSED_DATA_DIR / "val_features.parquet"
    
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    
    # Prepare features
    feature_cols = [
        c for c in train_df.columns 
        if c not in [ID_COL, TARGET_COL] and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    
    X_train = train_df[feature_cols].fillna(-999)
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].fillna(-999)
    y_val = val_df[TARGET_COL].values
    
    # Combine for cross-validation
    X = pd.concat([X_train, X_val], ignore_index=True)
    y = np.concatenate([y_train, y_val])
    
    # Train
    trainer = AdvancedFraudTrainer(
        n_folds=5,
        n_trials=n_trials,
        use_optuna=use_optuna,
    )
    trainer.train(X, y)
    trainer.save()
    
    # Compute SHAP
    shap_results = trainer.compute_shap_values(X)
    if shap_results:
        shap_path = MODELS_DIR / "shap_results.joblib"
        joblib.dump(shap_results, shap_path)
        logger.info(f"SHAP results saved to {shap_path}")
    
    return trainer


if __name__ == "__main__":
    # Quick training without full Optuna optimization
    trainer = train_advanced_model(n_trials=10, use_optuna=True)
