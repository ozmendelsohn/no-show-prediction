from typing import Dict, Any
import pandas as pd
from app.core.feature_creators.base import BaseFeatureCreator

class CaliforniaFeatureCreator(BaseFeatureCreator):
    """Feature creator for California medical appointments dataset."""

    def _create_custom_features(self, df: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Create custom features for California appointments data.

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
                if operation == 'appointment_lead_time':
                    # Calculate time between scheduling and appointment
                    df['schedule_appointment__lead_time_days'] = (
                        (df['appointment_date'] - df['schedule_date']).dt.total_seconds() / (24 * 60 * 60)
                    )
                
                elif operation == 'patient_history':
                    # Calculate patient's appointment history
                    df['patient__total_appointments'] = df.groupby('patient_id')['patient_id'].transform('count')
                    df['patient__no_show_rate'] = (
                        df.groupby('patient_id')['no_show'].transform('mean')
                    )
                
                elif operation == 'health_profile':
                    # Create combined health profile features
                    df['patient__health_conditions'] = (
                        df['hypertension'].astype(int) + df['diabetes'].astype(int)
                    )
                
                elif operation == 'time_based':
                    # Create time-based features
                    df['appointment__is_morning'] = (df['appointment_date'].dt.hour < 12).astype(int)
                    df['appointment__is_monday'] = (df['appointment_date'].dt.weekday == 0).astype(int)
                
            except Exception as e:
                self.logger.error(f"Error creating custom feature {operation}: {str(e)}")
        
        return df 