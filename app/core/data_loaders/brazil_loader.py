from typing import Dict, Any
import pandas as pd
from app.core.data_loaders.base import BaseDataLoader
from app.utils import get_logger

logger = get_logger(__name__)

class BrazilDataLoader(BaseDataLoader):
    """Data loader for Brazil medical appointments dataset."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing data loading parameters
        """
        super().__init__(config)
        self.logger = logger

    def _load_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the loaded Brazil appointments data.

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
        date_columns = ['appointment_date', 'date_of_birth', 'entry_service_date']
        datetime_format = self.config.get('datetime_format')
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Replace empty strings with NaN before conversion
                    df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
                    df[col] = pd.to_datetime(df[col], format=datetime_format, errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting {col} to datetime: {str(e)}")
                    # Try without format specification
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert boolean columns
        bool_columns = ['under_12_years_old', 'over_60_years_old', 'patient_needs_companion', 'rainy_day_before', 'storm_day_before']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Convert categorical columns
        cat_columns = ['specialty', 'gender', 'city', 'appointment_shift', 'rain_intensity', 'heat_intensity']
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert numeric columns
        num_columns = ['appointment_time', 'age', 'average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day']
        for col in num_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert no_show to boolean
        if 'no_show' in df.columns:
            df['no_show'] = df['no_show'].str.lower().map({'yes': True, 'no': False})
        
        # Validate the processed data
        self._validate_data(df)
        
        return df

    def _load_raw_data(self, path: str, format: str = 'csv', **kwargs) -> pd.DataFrame:
        """
        Load raw data from the specified path.

        Parameters
        ----------
        path : str
            Path to the data file
        format : str, optional
            Format of the data file, by default 'csv'
        **kwargs : dict
            Additional keyword arguments to pass to the data loading function

        Returns
        -------
        pd.DataFrame
            Raw data loaded from source
        """
        # Load data based on format
        if format.lower() == 'csv':
            df = pd.read_csv(path, **kwargs)
        else:
            raise ValueError(f"Unsupported data format: {format}")
        
        # Apply schema if provided
        if 'schema' in self.config:
            schema = self.config['schema'].get('columns', {})
            for col, dtype in schema.items():
                if col in df.columns:
                    try:
                        if dtype == 'datetime':
                            # Replace empty strings with NaN before conversion
                            df[col] = df[col].replace(r'^\s*$', pd.NA, regex=True)
                            df[col] = pd.to_datetime(
                                df[col], 
                                format=self.config.get('datetime_format'),
                                errors='coerce'
                            )
                        elif dtype == 'category':
                            df[col] = df[col].astype('category')
                        elif dtype == 'boolean':
                            df[col] = df[col].astype(bool)
                        else:
                            df[col] = df[col].astype(dtype)
                    except Exception as e:
                        self.logger.warning(f"Error converting {col} to {dtype}: {str(e)}")
        
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check required columns
        required_columns = [
            'specialty', 'appointment_time', 'gender', 'appointment_date', 
            'no_show', 'age', 'under_12_years_old', 'over_60_years_old', 
            'patient_needs_companion', 'average_temp_day', 'average_rain_day', 
            'max_temp_day', 'max_rain_day', 'rainy_day_before', 'storm_day_before', 
            'rain_intensity', 'heat_intensity'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['appointment_date']):
            raise ValueError("appointment_date must be datetime")
        
        # Check value ranges
        if 'age' in df.columns and not pd.isna(df['age']).all():
            if df['age'].min() < 0:
                raise ValueError("Age cannot be negative")
        
        # Check for duplicates in appointment data
        duplicate_cols = ['appointment_date', 'appointment_time', 'specialty']
        if df.duplicated(subset=duplicate_cols).any():
            self.logger.warning("Found duplicate appointments")