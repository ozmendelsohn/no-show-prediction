from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from app.utils import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing model parameters
        """
        self.config = config
        self._validate_config()
        self._setup_model()

    def _validate_config(self):
        """Validate the configuration parameters."""
        required_keys = ['model_params', 'training_params', 'feature_params']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    @abstractmethod
    def _setup_model(self):
        """Setup the specific model implementation."""
        pass

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.

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
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame

        Returns
        -------
        np.ndarray
            Predictions
        """
        pass

    @abstractmethod
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate model metrics.

        Parameters
        ----------
        y_true : pd.Series
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics
        """
        pass

    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and config
        self._save_model(path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: Union[str, Path]):
        """
        Load the model from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load model and config
        self._load_model(path)
        logger.info(f"Model loaded from: {path}")

    @abstractmethod
    def _save_model(self, path: Path):
        """
        Implementation-specific model saving.

        Parameters
        ----------
        path : Path
            Path to save the model
        """
        pass

    @abstractmethod
    def _load_model(self, path: Path):
        """
        Implementation-specific model loading.

        Parameters
        ----------
        path : Path
            Path to load the model from
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.

        Returns
        -------
        Optional[pd.Series]
            Feature importance series if available, None otherwise
        """
        pass