import os
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
from datetime import datetime

from app.core.data_loaders.base import BaseDataLoader

class CaliforniaLoader(BaseDataLoader):
    """Data loader for California medical appointments dataset."""

    def _load_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and process the California appointments data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame

        Returns 
        -------
        pd.DataFrame
            Processed DataFrame containing appointments data
        """
        # Convert date columns to datetime
        date_columns = ['appointment_date', 'schedule_date']
        datetime_format = self.config.get('datetime_format', '%Y-%m-%d %H:%M:%S')
        
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format=datetime_format)
                except Exception as e:
                    self.logger.warning(f"Error converting {col} to datetime: {str(e)}")
                    # Try without format specification
                    df[col] = pd.to_datetime(df[col])
        
        # Convert boolean columns
        bool_columns = ['no_show', 'hypertension', 'diabetes']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Convert categorical columns
        cat_columns = ['gender', 'specialty', 'neighborhood']
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert numeric columns
        num_columns = ['age']
        for col in num_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
