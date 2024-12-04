from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pandas as pd
from app.utils import get_logger

logger = get_logger(__name__)


class BaseDataLoader(ABC):
    """Base class for data loaders."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing data loading parameters
        """
        self.config = config

    @abstractmethod
    def _load_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load data.
        """
        pass

    def load_data(self) -> pd.DataFrame:
        """
        Load data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing data
        """
        df = self._load_raw_data(**self.config.get("raw_data_kargs"))
        df = self._load_processed_data(df)
        df = self._schema_enforcement(df, self.config.get("schema"))
        self._save_interim_data(df, **self.config.get("interim_data_kargs"))
        return df

    def _schema_enforcement(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Enforce the schema on the DataFrame

        Support the following schema configurations:
        schema:
        ===== Option 1 =====
        - columns:
            - column1_name: "column1_type"
            - column2_name: "column2_type"
            - column3_name: "column3_type"
        ===== Option 2 =====
        - boolean columns:
            - boolean_column1_name
            - boolean_column2_name
        - integer columns:
            - integer_column1_name
            - integer_column2_name
        - numeric columns:
            - numeric_column1_name
            - numeric_column2_name
        - categorical columns:
            - categorical_column1_name
            - categorical_column2_name
        - datetime columns:
            - datetime_column1_name
            - datetime_column2_name
        =====================
        datetime_format: "%Y-%m-%dT%H:%M:%S"
        or per column:
        - column1_name: "%Y-%m-%dT%H:%M:%S"
        - column2_name: "%Y-%m-%dT%H:%M:%S"

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        schema : Dict[str, Any], optional
            Schema dictionary containing schema configuration

        Returns
        -------
        pd.DataFrame
            Validated DataFrame
        """

        if schema is None:
            logger.warning("No schema provided, skipping validation")
            return df

        columns_enforced = []

        # Option 1: enforce per column
        if "columns" in schema:
            for column, column_type in schema["columns"].items():
                if column not in df.columns:
                    raise ValueError(f"Column {column} not found in DataFrame")
                columns_enforced.append(column)
                # check if timestamp column and use the datetime_format if provided
                if column_type == "datetime":
                    timestamp_format = self._get_datetime_format(schema, column)
                    df[column] = pd.to_datetime(df[column], format=timestamp_format if timestamp_format else None)
                else:
                    df[column] = df[column].astype(column_type)
        # Option 2: enforce boolean, numeric, categorical, datetime columns
        else:
            for int_col in schema["integer_columns"]:
                self._enforce_column_type(df, int_col, dtype=int, columns_enforced=columns_enforced)
            for num_col in schema["numeric_columns"]:
                self._enforce_column_type(df, num_col, dtype=float, columns_enforced=columns_enforced)
            for bool_col in schema["boolean_columns"]:
                self._enforce_column_type(df, bool_col, dtype=bool, columns_enforced=columns_enforced)
            for cat_col in schema["categorical_columns"]:
                self._enforce_column_type(df, cat_col, dtype="category", columns_enforced=columns_enforced)
            for datetime_col in schema["datetime_columns"]:
                self._enforce_column_type(
                    df, datetime_col, dtype="datetime", schema=schema, columns_enforced=columns_enforced
                )

        return df

    def _get_datetime_format(self, schema: Dict[str, Any], column: str) -> str:
        """Get datetime format for a specific column from schema config.

        Parameters
        ----------
        column : str
            Name of the datetime column

        Returns
        -------
        str
            Datetime format string if specified, None otherwise
        """
        if "datetime_format" not in schema:
            return None

        timestamp_format = schema["datetime_format"]
        if isinstance(timestamp_format, dict):
            return timestamp_format[column]
        return

    def _enforce_column_type(self, df, column, dtype, schema=None, columns_enforced=None):
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        if columns_enforced is not None:
            columns_enforced.append(column)

        if dtype == "datetime":
            timestamp_format = self._get_datetime_format(schema, column)
            df[column] = pd.to_datetime(df[column], format=timestamp_format if timestamp_format else None)
        else:
            df[column] = df[column].astype(dtype)


    def _save_interim_data(self, df: pd.DataFrame, path: str, format: str, **kwargs):
        """Save data to a file."""
        if format == "csv":
            df.to_csv(path, **kwargs)
        elif format == "parquet":
            df.to_parquet(path, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported")

    def _load_raw_data(self, path: str, format: str, **kwargs) -> pd.DataFrame:
        """Load raw data from a file."""
        if format == "csv":
            return pd.read_csv(path, **kwargs)
        elif format == "parquet":
            return pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Format {format} not supported")