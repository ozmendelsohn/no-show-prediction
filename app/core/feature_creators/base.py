from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
from app.utils import get_logger

logger = get_logger(__name__)

class BaseFeatureCreator(ABC):
    """Base class for feature creation following sklearn naming conventions."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature creator.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing feature creation parameters
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration parameters."""
        required_keys = ['features_to_create']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with additional features
        """
        logger.info("Starting feature creation...")
        
        # Get list of features to create
        features_to_create = self.config.get('features_to_create', [])
        
        # Create each feature
        for feature in features_to_create:
            feature_type = feature.get('type')
            if not feature_type:
                logger.warning(f"Feature type not specified in config: {feature}")
                continue
                
            try:
                if feature_type == 'datetime':
                    df = self._create_datetime_features(df, feature)
                elif feature_type == 'categorical':
                    df = self._create_categorical_features(df, feature)
                elif feature_type == 'numeric':
                    df = self._create_numeric_features(df, feature)
                elif feature_type == 'text':
                    df = self._create_text_features(df, feature)
                elif feature_type == 'custom':
                    df = self._create_custom_features(df, feature)
                else:
                    logger.warning(f"Unsupported feature type: {feature_type}")
            except Exception as e:
                logger.error(f"Error creating feature: {feature}. Error: {str(e)}")
                continue
        
        logger.info("Feature creation completed")
        return df

    def _create_datetime_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Create datetime-based features."""
        source_col = feature_config.get('source_column')
        if not source_col or source_col not in df.columns:
            logger.warning(f"Source column {source_col} not found for datetime features")
            return df
            
        components = feature_config.get('components', [])
        for component in components:
            try:
                if component == 'hour':
                    df[f"{source_col}__hour"] = df[source_col].dt.hour
                elif component == 'day':
                    df[f"{source_col}__day"] = df[source_col].dt.day
                elif component == 'month':
                    df[f"{source_col}__month"] = df[source_col].dt.month
                elif component == 'year':
                    df[f"{source_col}__year"] = df[source_col].dt.year
                elif component == 'weekday':
                    df[f"{source_col}__weekday"] = df[source_col].dt.weekday
                elif component == 'is_weekend':
                    df[f"{source_col}__is_weekend"] = df[source_col].dt.weekday.isin([5, 6]).astype(int)
            except Exception as e:
                logger.error(f"Error creating datetime feature {component} for {source_col}: {str(e)}")
                
        return df

    def _create_categorical_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Create categorical features."""
        source_col = feature_config.get('source_column')
        if not source_col or source_col not in df.columns:
            logger.warning(f"Source column {source_col} not found for categorical features")
            return df
            
        operations = feature_config.get('operations', [])
        for operation in operations:
            try:
                if operation == 'one_hot':
                    # Standardize category names before one-hot encoding
                    standardized_values = (df[source_col]
                        .str.lower()
                        .str.replace(r'[^a-z0-9\s]', '')  # Remove special chars except spaces
                        .str.replace(r'\s+', '_'))  # Replace spaces with underscore
                    
                    # Create one-hot encoded columns with standardized names
                    one_hot = pd.get_dummies(standardized_values, prefix=f"{source_col}__onehot")
                    df = pd.concat([df, one_hot], axis=1)
                elif operation == 'count_encoding':
                    df[f"{source_col}__count"] = df.groupby(source_col)[source_col].transform('count')
            except Exception as e:
                logger.error(f"Error creating categorical feature {operation} for {source_col}: {str(e)}")
                
        return df

    def _create_numeric_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Create numeric features."""
        source_col = feature_config.get('source_column')
        if not source_col or source_col not in df.columns:
            logger.warning(f"Source column {source_col} not found for numeric features")
            return df
            
        operations = feature_config.get('operations', [])
        for operation in operations:
            try:
                if operation == 'binning':
                    bins = feature_config.get('bins', 5)
                    df[f"{source_col}__binned"] = pd.qcut(df[source_col], q=bins, labels=False)
                elif operation == 'scaling':
                    df[f"{source_col}__scaled"] = (df[source_col] - df[source_col].mean()) / df[source_col].std()
            except Exception as e:
                logger.error(f"Error creating numeric feature {operation} for {source_col}: {str(e)}")
                
        return df

    def _create_text_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Create text-based features."""
        source_col = feature_config.get('source_column')
        if not source_col or source_col not in df.columns:
            logger.warning(f"Source column {source_col} not found for text features")
            return df
            
        operations = feature_config.get('operations', [])
        for operation in operations:
            try:
                if operation == 'length':
                    df[f"{source_col}__length"] = df[source_col].str.len()
                elif operation == 'word_count':
                    df[f"{source_col}__word_count"] = df[source_col].str.split().str.len()
            except Exception as e:
                logger.error(f"Error creating text feature {operation} for {source_col}: {str(e)}")
                
        return df

    @abstractmethod
    def _create_custom_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create custom features specific to each dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        feature_config : Dict[str, Any]
            Configuration for custom feature creation

        Returns
        -------
        pd.DataFrame
            DataFrame with custom features
        """
        pass 