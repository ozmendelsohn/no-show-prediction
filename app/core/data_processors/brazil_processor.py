from typing import Dict, Any
import pandas as pd
from app.core.data_processors.base import BaseDataProcessor
from app.utils import get_logger

logger = get_logger(__name__)

class BrazilDataProcessor(BaseDataProcessor):
    """Data processor for Brazil medical appointments dataset."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing processing parameters
        """
        super().__init__(config)
        self.logger = logger

    def _custom_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom processing steps for Brazil data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        pd.DataFrame
            Processed DataFrame
        """
        # Convert no_show to boolean
        if 'no_show' in df.columns:
            df['no_show'] = df['no_show'].str.lower().map({'yes': True, 'no': False})
        
        # Handle disability values
        if 'disability' in df.columns:
            df['disability'] = df['disability'].fillna('none')
        
        # Handle age outliers and missing values
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].clip(0, 120)  # Cap age at reasonable limits
            df['age'] = df['age'].fillna(df['age'].median())
        
        # Clean specialty names
        if 'specialty' in df.columns:
            df['specialty'] = (
                df['specialty']
                .str.strip()
                .str.lower()
                .fillna('unknown')
            )
        
        # Clean city names
        if 'city' in df.columns:
            df['city'] = (
                df['city']
                .str.strip()
                .str.upper()
                .fillna('UNKNOWN')
            )
        
        # Handle appointment time
        if 'appointment_time' in df.columns:
            # Convert to 24-hour format if needed
            df['appointment_time'] = pd.to_datetime(df['appointment_time'], format='%H:%M', errors='coerce').dt.time
        
        # Clean gender values
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.upper().map({'M': 'M', 'F': 'F'}).fillna('U')
        
        # Handle weather-related features
        weather_columns = ['average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day']
        for col in weather_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Clean intensity categories
        intensity_columns = ['rain_intensity', 'heat_intensity']
        for col in intensity_columns:
            if col in df.columns:
                df[col] = df[col].str.lower().fillna('unknown')
        
        # Handle boolean columns
        bool_columns = ['under_12_years_old', 'over_60_years_old', 'patient_needs_companion', 
                       'rainy_day_before', 'storm_day_before']
        for col in bool_columns:
            if col in df.columns:
                # Convert various forms of boolean values
                df[col] = df[col].map({
                    True: True, 'True': True, '1': True, 1: True, 'yes': True,
                    False: False, 'False': False, '0': False, 0: False, 'no': False
                }).fillna(False)
        
        # Apply column name standardization if configured
        if self.config.get('force_column_mapping', False):
            column_mapping = self.config.get('column_mapping', {})
            df = df.rename(columns=column_mapping)
        
        return df