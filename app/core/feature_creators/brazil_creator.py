from typing import Dict, Any
import pandas as pd
import numpy as np
from app.core.feature_creators.base import BaseFeatureCreator
from app.utils import get_logger

logger = get_logger(__name__)

class BrazilFeatureCreator(BaseFeatureCreator):
    """Feature creator for Brazil medical appointments dataset."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature creator.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing feature creation parameters
        """
        super().__init__(config)
        self.logger = logger

    def _create_custom_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create custom features for Brazil appointments data.

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
        operations = feature_config.get('operations', [])
        
        for operation in operations:
            try:
                if isinstance(operation, dict):
                    operation_name = operation.get('name')
                    params = operation.get('params', {})
                else:
                    operation_name = operation
                    params = {}
                
                if operation_name == 'patient_history':
                    # Calculate patient's appointment history
                    history_features = params.get('features', [])
                    
                    if 'no_show_rate' in history_features:
                        # Convert target to numeric for mean calculation
                        df['target_numeric'] = df['target'].astype(int)
                        df['patient__no_show_rate'] = (
                            df.groupby('specialty')['target_numeric'].transform('mean')
                        )
                        df = df.drop('target_numeric', axis=1)
                    
                    if 'total_appointments' in history_features:
                        df['patient__total_appointments'] = (
                            df.groupby('specialty')['specialty'].transform('count')
                        )
                
                elif operation_name == 'health_profile':
                    # Create health profile features
                    conditions = params.get('conditions', [])
                    if 'disability' in conditions and 'disability' in df.columns:
                        df['patient__has_disability'] = df['disability'].notna() & (df['disability'] != 'none')
                    
                    if params.get('needs_companion', False):
                        df['patient__needs_companion'] = df['needs_companion']
                
                elif operation_name == 'weather_profile':
                    # Create weather profile features
                    features = params.get('features', [])
                    
                    if 'rain_level' in features:
                        df['weather__rain_severity'] = (
                            df['rain_level'].map({
                                'no_rain': 0, 'light': 1, 'moderate': 2, 'heavy': 3
                            }).fillna(0)
                        )
                    
                    if 'heat_level' in features:
                        df['weather__heat_severity'] = (
                            df['heat_level'].map({
                                'mild': 0, 'moderate': 1, 'high': 2, 'extreme': 3
                            }).fillna(0)
                        )
                    
                    if 'prev_rain' in features:
                        df['weather__prev_rain'] = df['prev_rain']
                    
                    if 'prev_storm' in features:
                        df['weather__prev_storm'] = df['prev_storm']
                
            except Exception as e:
                self.logger.error(f"Error creating custom feature {operation_name}: {str(e)}")
        
        return df 