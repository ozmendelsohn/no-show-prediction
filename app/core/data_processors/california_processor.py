from typing import Dict, Any
import pandas as pd
from app.core.data_processors.base import BaseDataProcessor

class CaliforniaProcessor(BaseDataProcessor):
    """Data processor for California medical appointments dataset."""

    def _custom_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom processing steps for California appointments data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            Processed DataFrame
        """
        # Remove any whitespace in string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].str.strip()

        # Convert all string columns to lowercase
        if self.config.get("convert_strings_to_lowercase", True):
            for col in string_columns:
                df[col] = df[col].str.lower()

        return df 