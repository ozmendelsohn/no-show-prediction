from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from app.utils import get_logger

logger = get_logger(__name__)

class BaseDataProcessor(ABC):
    """Base class for data processors that handles data cleaning and standardization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing data processing parameters
        """
        self.config = config

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame through a series of standardization and cleaning steps.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to process

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with standardized column names, handled missing values,
            removed duplicates and custom processing applied
        """
        logger.info("Starting data processing pipeline...")
        
        # Standardize column names
        logger.info("Step: Standardizing column names")
        df = self._standardize_column_names(df)

        # Handle missing values
        logger.info("Step: Handling missing values") 
        df = self._handle_missing_values(df)

        # Remove duplicates
        logger.info("Step: Removing duplicate records")
        df = self._remove_duplicates(df)

        # Apply custom processing
        logger.info("Step: Applying custom processing")
        df = self._custom_processing(df)
            
        logger.info("Data processing pipeline completed")
        return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Applies basic standardization (lowercase, underscores) and optional custom mapping.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names
        """
        # Apply automatic standardization
        std_columns = self._get_standardized_column_names(df.columns)
        df = df.rename(columns=std_columns)
        
        # Apply custom mapping if configured
        if column_mapping := self.config.get("column_mapping"):
            df = self._apply_custom_column_mapping(df, column_mapping)
            
        return df
    
    def _get_standardized_column_names(self, columns: pd.Index) -> Dict[str, str]:
        """Generate standardized names for columns."""
        std_columns = {}
        for col in columns:
            # Convert to lowercase and replace special chars with underscore
            std_name = (col.lower()
                      .replace(" ", "_")
                      .replace("-", "_")
                      .replace(".", "_")
                      .replace("/", "_"))
            # Remove multiple underscores
            std_name = "_".join(filter(None, std_name.split("_")))
            std_columns[col] = std_name
            
            if col != std_name:
                logger.info(f"Standardized column: {col} -> {std_name}")
                
        return std_columns
    
    def _apply_custom_column_mapping(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """Apply custom column name mapping from config."""
        force_mapping = self.config.get("force_column_mapping", False)
        valid_mapping = {}
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                self._handle_column_mapping(old_name, new_name, force_mapping, valid_mapping)
            else:
                self._try_case_insensitive_mapping(df, old_name, new_name, force_mapping, valid_mapping)
        
        if valid_mapping:
            logger.info("Applied custom column mappings:")
            for old, new in valid_mapping.items():
                logger.info(f"  {old} -> {new}")
            df = df.rename(columns=valid_mapping)
            
        return df
    
    def _handle_column_mapping(
        self,
        old_name: str,
        new_name: str, 
        force_mapping: bool,
        valid_mapping: Dict[str, str]
    ) -> None:
        """Handle mapping for a single column."""
        if not force_mapping:
            logger.warning(
                f"Column '{old_name}' mapping to '{new_name}' requires force_mapping: true"
            )
        else:
            valid_mapping[old_name] = new_name
            
    def _try_case_insensitive_mapping(
        self,
        df: pd.DataFrame,
        old_name: str,
        new_name: str,
        force_mapping: bool,
        valid_mapping: Dict[str, str]
    ) -> None:
        """Try to find and map column names case-insensitively."""
        matches = [c for c in df.columns if c.lower() == old_name.lower()]
        if matches:
            self._handle_column_mapping(matches[0], new_name, force_mapping, valid_mapping)
        else:
            logger.warning(f"Column '{old_name}' not found in DataFrame")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Supports two configuration approaches:
        1. Global Strategy: Apply same strategy to all columns
        2. Per Column Strategy: Configure handling per column

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with handled missing values
        """
        if "missing_values" not in self.config:
            logger.warning("No missing value handling configuration provided")
            return df

        if "missing_value_strategy" in self.config:
            return self._handle_missing_values_global(df)
        elif "columns" in self.config["missing_values"]:
            return self._handle_missing_values_per_column(df)
            
        logger.warning("Invalid missing value configuration")
        return df

    def _handle_missing_values_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply global strategy for handling missing values."""
        strategy = self.config.get("missing_value_strategy")
        
        if strategy == "drop":
            return df.dropna()
        elif strategy == "fill":
            fill_values = self.config.get("fill_values", {})
            return self._fill_missing_values(df, fill_values)
        else:
            raise ValueError(f"Unsupported missing value strategy: {strategy}")
            
    def _fill_missing_values(
        self, 
        df: pd.DataFrame, 
        fill_values: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fill missing values based on column type."""
        default_value = fill_values.get("default")
        type_defaults = {
            "numeric": fill_values.get("numeric", 0),
            "categorical": fill_values.get("categorical", "unknown"),
            "boolean": fill_values.get("boolean", False)
        }
        
        for col in df.columns:
            fill_value = self._get_fill_value(df[col], default_value, type_defaults)
            df[col] = self._fill_column(df[col], fill_value)
            
        return df
        
    def _get_fill_value(
        self, 
        series: pd.Series, 
        default: Any, 
        type_defaults: Dict[str, Any]
    ) -> Any:
        """Determine appropriate fill value based on column type."""
        if default is not None:
            return default
            
        if pd.api.types.is_numeric_dtype(series):
            return type_defaults["numeric"]
        elif pd.api.types.is_bool_dtype(series):
            return type_defaults["boolean"]
        else:
            return type_defaults["categorical"]
            
    def _fill_column(self, series: pd.Series, fill_value: Any) -> pd.Series:
        """Fill missing values in a single column."""
        if pd.api.types.is_categorical_dtype(series) and pd.isna(series).any():
            if fill_value not in series.cat.categories:
                series = series.cat.add_categories(fill_value)
        return series.fillna(fill_value)

    def _handle_missing_values_per_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-column strategy for handling missing values."""
        for column, config in self.config["missing_values"]["columns"].items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
                
            strategy = config.get("strategy")
            if strategy == "drop":
                df = df.dropna(subset=[column])
            elif strategy == "fill":
                if fill_value := config.get("value"):
                    df[column] = df[column].fillna(fill_value)
            else:
                logger.warning(f"Unsupported strategy {strategy} for column {column}")
                
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicates removed
        """
        subset = self._get_valid_duplicate_subset(df)
        keep = self.config.get("duplicate_keep", "first")
        
        return df.drop_duplicates(subset=subset, keep=keep)
        
    def _get_valid_duplicate_subset(self, df: pd.DataFrame) -> Optional[List[str]]:
        """Get valid columns for duplicate checking."""
        if not (subset := self.config.get("duplicate_subset")):
            return None
            
        valid_subset = []
        for col in subset:
            if col in df.columns:
                valid_subset.append(col)
            else:
                normalized_col = col.lower().replace(" ", "_")
                matches = [c for c in df.columns 
                          if c.lower().replace(" ", "_") == normalized_col]
                if matches:
                    valid_subset.append(matches[0])
                else:
                    logger.warning(f"Column '{col}' not found for duplicate check")
        
        if not valid_subset:
            logger.warning("No valid columns for duplicate removal. Using all columns.")
            return None
            
        return valid_subset

    @abstractmethod
    def _custom_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement custom processing steps specific to each dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            Processed DataFrame
        """
        pass