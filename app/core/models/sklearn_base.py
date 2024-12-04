from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from app.core.models.base import BaseModel
from app.utils import get_logger

logger = get_logger(__name__)

class SklearnBaseModel(BaseModel):
    """Base class for scikit-learn models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing model parameters
        """
        self.model: Optional[BaseEstimator] = None
        super().__init__(config)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: bool = True
    ) -> Dict[str, Any]:
        """
        Train the scikit-learn model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame
        y : pd.Series
            Target series
        validation_split : bool, optional
            Whether to perform validation split, by default True

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training metrics
        """
        # Get training parameters
        test_size = self.config['training_params'].get('test_size', 0.2)
        random_state = self.config['training_params'].get('random_state', 42)

        # Store feature names
        self.config['feature_params']['feature_names'] = list(X.columns)

        if validation_split:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if self.config['training_params'].get('stratify', True) else None
            )

            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)

            # Get predictions
            train_preds = self.predict(X_train)
            val_preds = self.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'train': self._calculate_metrics(y_train, train_preds),
                'validation': self._calculate_metrics(y_val, val_preds)
            }
        else:
            # Train on full dataset
            logger.info("Training model on full dataset...")
            self.model.fit(X, y)
            
            # Get predictions
            train_preds = self.predict(X)
            
            # Calculate metrics
            metrics = {
                'train': self._calculate_metrics(y, train_preds)
            }

        logger.info("Training completed")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the scikit-learn model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame

        Returns
        -------
        np.ndarray
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
        return self.model.predict_proba(X)

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Calculate scikit-learn model metrics.

        Parameters
        ----------
        y_true : pd.Series
            True labels
        y_pred : np.ndarray
            Predicted labels
        average : str, optional
            Averaging method for metrics, by default 'binary'

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average)
        }
        
        # Add ROC AUC if probability predictions are available
        if hasattr(self.model, 'predict_proba'):
            try:
                y_prob = self.predict_proba(y_true.index)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics

    def _save_model(self, path: Path):
        """
        Save scikit-learn model to disk.

        Parameters
        ----------
        path : Path
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump({
            'model': self.model,
            'config': self.config
        }, path)

    def _load_model(self, path: Path):
        """
        Load scikit-learn model from disk.

        Parameters
        ----------
        path : Path
            Path to load the model from
        """
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.config = loaded['config']

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance from scikit-learn model.

        Returns
        -------
        Optional[pd.Series]
            Feature importance series if available, None otherwise
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.config['feature_params'].get('feature_names', None)
            )
        elif hasattr(self.model, 'coef_'):
            return pd.Series(
                self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_,
                index=self.config['feature_params'].get('feature_names', None)
            )
        else:
            logger.warning("Model does not provide feature importance")
            return None 