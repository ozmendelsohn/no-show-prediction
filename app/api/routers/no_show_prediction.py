from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import mlflow
import joblib
from pathlib import Path

from app import load_config
from app.core.data_processors.factory import get_processor
from app.core.feature_creators.factory import get_feature_creator

router = APIRouter()

# Load configuration and model
CONFIG_PATH = Path("artifacts/california_synthetic/config.yaml")
config = load_config(CONFIG_PATH)
MODEL_PATH = Path("artifacts/california_synthetic/models")
ENCODERS_PATH = Path("artifacts/california_synthetic/encoders.joblib")
FEATURE_LISTS_PATH = Path("artifacts/california_synthetic/feature_lists.joblib")

# Load model and artifacts
try:
    model = mlflow.sklearn.load_model(str(MODEL_PATH))
    encoders = joblib.load(ENCODERS_PATH)
    feature_lists = joblib.load(FEATURE_LISTS_PATH)
    print(f"Loaded feature lists: {feature_lists}")
except Exception as e:
    print(f"Error loading artifacts: {str(e)}")
    print(f"Please ensure the artifacts are saved at: {MODEL_PATH}, {ENCODERS_PATH}, and {FEATURE_LISTS_PATH}")
    raise

class AppointmentInput(BaseModel):
    """
    Input data model for appointment prediction.
    Only includes the necessary fields based on feature importance.
    """
    patient_id: str = Field(..., description="Unique identifier for the patient")
    gender: str = Field(..., description="Patient gender (M/F)")
    age: int = Field(..., description="Patient age", ge=0, le=120)
    appointment_date: datetime = Field(..., description="Appointment date and time")
    schedule_date: datetime = Field(..., description="Date when appointment was scheduled")
    alcohol_consumption: str = Field(..., description="Alcohol consumption level (None/Low/Medium/High)")
    hypertension: bool = Field(..., description="Whether patient has hypertension")
    diabetes: bool = Field(..., description="Whether patient has diabetes")
    specialty: str = Field(..., description="Medical specialty for the appointment")
    prediction_timestamp: Optional[datetime] = Field(None, description="Optional timestamp for simulating predictions at different points in time")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P123456",
                "gender": "F",
                "age": 45,
                "appointment_date": "2024-12-01T10:30:00",
                "schedule_date": "2024-11-25T14:20:00",
                "alcohol_consumption": "Low",
                "hypertension": False,
                "diabetes": False,
                "specialty": "Cardiology",
                "prediction_timestamp": None
            }
        }

def create_features(appointment: AppointmentInput) -> pd.DataFrame:
    """
    Create features from raw input data using existing pipeline.
    
    Parameters
    ----------
    appointment : AppointmentInput
        Raw appointment data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features
    """
    # Convert input to DataFrame with proper column names matching the training data
    data = pd.DataFrame([{
        'Patient ID': appointment.patient_id,
        'Gender': appointment.gender,
        'Age': appointment.age,
        'Appointment Date': appointment.appointment_date,
        'Schedule Date': appointment.schedule_date,
        'Alcohol Consumption': appointment.alcohol_consumption,
        'Hypertension': appointment.hypertension,
        'Diabetes': appointment.diabetes,
        'Specialty': appointment.specialty
    }])
    
        # Initialize data processor with config
    processor_name = config['data_processing'].get('data_processor_name')
    processor_class = get_processor(data_processor_name=processor_name)
    processor = processor_class(config=config['data_processing'])
    
        # Process data using the same pipeline as training
    processed_df = processor.process_data(data)
    
    # Initialize feature creator with config
    creator_name = config['feature_creation'].get('feature_creator_name')
    creator_class = get_feature_creator(feature_creator_name=creator_name)
    creator = creator_class(config=config['feature_creation'])
    
    # Create features using the same pipeline as training
    featured_df = creator.create_features(processed_df)
    
    # Load historical data for patient statistics
    try:
        historical_data = pd.read_csv("data/california_synthetic/raw/no_show.csv")
        
        # Convert date columns to datetime
        historical_data['Appointment Date'] = pd.to_datetime(historical_data['Appointment Date'])
        
        # Filter historical data up to prediction_timestamp if provided
        if appointment.prediction_timestamp is not None:
            historical_data = historical_data[
                historical_data['Appointment Date'] <= appointment.prediction_timestamp
            ]
            print(f"Filtered historical data up to: {appointment.prediction_timestamp}")
            print(f"Historical data size after filtering: {len(historical_data)}")
        
        # Calculate patient statistics
        patient_stats = historical_data[historical_data['Patient ID'] == appointment.patient_id]
        
        if len(patient_stats) > 0:
            featured_df['patient__no_show_rate'] = patient_stats['no_show'].mean()
            featured_df['patient__total_appointments'] = len(patient_stats)
            print(f"Patient statistics at {appointment.prediction_timestamp if appointment.prediction_timestamp else 'current time'}:")
            print(f"- Total appointments: {len(patient_stats)}")
            print(f"- No-show rate: {patient_stats['no_show'].mean():.2%}")
        else:
            # New patient
            featured_df['patient__no_show_rate'] = 0.0
            featured_df['patient__total_appointments'] = 0
            print("No historical data found for patient")
            
    except Exception as e:
        print(f"Error loading historical data: {str(e)}")
        # Default values if historical data is not available
        featured_df['patient__no_show_rate'] = 0.0
        featured_df['patient__total_appointments'] = 0
    
    
    # Get required features from saved feature lists
    required_features = feature_lists['feature_names']
    
    # Initialize missing features with zeros
    for feature in required_features:
        if feature not in featured_df.columns:
            featured_df[feature] = 0
    
    # Encode categorical variables using saved encoders
    for col, encoder in encoders.items():
        if col in featured_df.columns:
            try:
                featured_df[col] = encoder.transform(featured_df[col].astype(str))
            except:
                # Handle unseen categories
                featured_df[col] = -1
    
    # Convert boolean columns to int
    bool_columns = ['hypertension', 'diabetes']
    for col in bool_columns:
        if col in featured_df.columns:
            featured_df[col] = featured_df[col].astype(int)
    
    # Select and order features according to the model's expectations
    model_features = model.feature_names_in_
    final_df = pd.DataFrame(0, index=featured_df.index, columns=model_features, dtype=np.float32)
    
    # Fill in available features
    for col in model_features:
        if col in featured_df.columns:
            final_df[col] = featured_df[col].astype(np.float32)
    
    return final_df

@router.post("/predict")
async def predict_no_show(appointment: AppointmentInput):
    """
    Predict the probability of a no-show for an appointment.
    """
    try:
        # Create features
        features_df = create_features(appointment)
        
        # Make prediction
        no_show_prob = model.predict_proba(features_df)[0, 1]
        prediction = bool(no_show_prob >= 0.5)
        
        return {
            "prediction": prediction,
            "no_show_probability": float(no_show_prob),
            "risk_level": "High" if no_show_prob >= 0.7 else "Medium" if no_show_prob >= 0.3 else "Low",
            "features_used": features_df.columns.tolist(),
            "patient_history": {
                "total_appointments": int(features_df['patient__total_appointments'].iloc[0]),
                "no_show_rate": float(features_df['patient__no_show_rate'].iloc[0]),
                "history_cutoff_date": appointment.prediction_timestamp.isoformat() if appointment.prediction_timestamp else None
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@router.get("/model-info")
async def get_model_info():
    """
    Get information about the model and its features.
    """
    return {
        "required_features": list(AppointmentInput.model_json_schema()["properties"].keys()),
        "model_features": model.feature_names_in_.tolist(),
        "feature_importance": {
            name: float(importance) 
            for name, importance in zip(
                model.feature_names_in_,
                model.feature_importances_
            )
        },
        "feature_lists": feature_lists
    } 