# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import logging
from sklearn.svm import SVC
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, precision_score, roc_curve, auc
from xgboost import XGBClassifier
import seaborn as sns
from pandarallel import pandarallel
from multiprocessing import Pool, cpu_count
import tqdm
from pathlib import Path
from dotenv import load_dotenv
import os
import itertools
from sklearn.cluster import KMeans

# Load environment variables from .env file
load_dotenv()

print("Imported all required libraries successfully")

# Initialize parallel processing for pandas
pandarallel.initialize(progress_bar=True, nb_workers=cpu_count())

tqdm.tqdm.pandas()

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the medical appointment dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the medical appointment data
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe
    """
    print(f"\nLoading data from {filepath}...")
    
    # Read CSV file
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert date columns to datetime
    print("Converting date columns to datetime format...")
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    print(f"Final dataframe shape: {df.shape}")
    
    return df

def create_patient_clusters(df):
    """
    Create patient clusters based on raw features for better imputation.
    """
    print("\nCreating patient clusters...")
    
    # Select features for clustering
    cluster_features = [
        'Age',
        'Gender',
        'Scholarship',
        'Hipertension',
        'Diabetes',
        'Alcoholism',
        'Handcap',
        'SMS_received'
    ]
    
    # Prepare data for clustering
    cluster_data = df[cluster_features].copy()
    
    # Encode categorical variables
    gender_encoder = LabelEncoder()
    cluster_data['Gender'] = gender_encoder.fit_transform(cluster_data['Gender'])
    
    # Scale the data
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_data_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('models/elbow_curve.png')
    plt.close()
    
    # Create clusters
    n_clusters = 10  # Choose based on elbow curve
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['patient_cluster'] = kmeans.fit_predict(cluster_data_scaled)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster in range(n_clusters):
        cluster_mask = df['patient_cluster'] == cluster
        cluster_df = df[cluster_mask].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Calculate lead times
        lead_times = (cluster_df['AppointmentDay'] - cluster_df['ScheduledDay']).dt.total_seconds() / (24 * 60 * 60)
        
        # Calculate time-based patterns
        cluster_df['hour'] = cluster_df['AppointmentDay'].dt.hour
        cluster_df['is_weekend'] = cluster_df['AppointmentDay'].dt.dayofweek >= 5
        cluster_df['is_morning'] = cluster_df['hour'] < 12
        cluster_df['month'] = cluster_df['AppointmentDay'].dt.month
        cluster_df['no_show_binary'] = (cluster_df['No-show'] == 'Yes').astype(int)
        
        # Calculate no-show rates for different conditions
        weekend_mask = cluster_df['is_weekend']
        morning_mask = cluster_df['is_morning']
        sms_mask = cluster_df['SMS_received'] == 1
        
        # Calculate seasonal no-show rate
        seasonal_rates = cluster_df.groupby('month')['no_show_binary'].mean()
        
        cluster_stats[cluster] = {
            'size': cluster_mask.sum(),
            'avg_noshow_rate': cluster_df['no_show_binary'].mean(),
            'avg_age': cluster_df['Age'].mean(),
            'common_gender': cluster_df['Gender'].mode().iloc[0],
            'scholarship_rate': cluster_df['Scholarship'].mean(),
            'hipertension_rate': cluster_df['Hipertension'].mean(),
            'diabetes_rate': cluster_df['Diabetes'].mean(),
            'alcoholism_rate': cluster_df['Alcoholism'].mean(),
            'avg_lead_time': lead_times.mean(),
            'lead_time_std': lead_times.std(),
            'weekend_noshow_rate': cluster_df[weekend_mask]['no_show_binary'].mean() if weekend_mask.any() else 0,
            'morning_noshow_rate': cluster_df[morning_mask]['no_show_binary'].mean() if morning_mask.any() else 0,
            'afternoon_noshow_rate': cluster_df[~morning_mask]['no_show_binary'].mean() if (~morning_mask).any() else 0,
            'seasonal_noshow_rate': seasonal_rates.mean(),
            'sms_response_rate': (1 - cluster_df[sms_mask]['no_show_binary'].mean()) if sms_mask.any() else 0.5
        }
    
    print("\nCluster Statistics:")
    for cluster, stats in cluster_stats.items():
        print(f"\nCluster {cluster}:")
        print(f"Size: {stats['size']}")
        print(f"Average no-show rate: {stats['avg_noshow_rate']:.3f}")
        print(f"Average age: {stats['avg_age']:.1f}")
        print(f"Common gender: {stats['common_gender']}")
        print(f"Average lead time: {stats['avg_lead_time']:.1f} days")
        print(f"Weekend no-show rate: {stats['weekend_noshow_rate']:.3f}")
        print(f"Morning no-show rate: {stats['morning_noshow_rate']:.3f}")
        
    return df['patient_cluster'], cluster_stats

def create_historical_features(df):
    """
    Create time series based features from historical no-show patterns.
    Uses clustering for better imputation of new patients.
    """
    print("\nCreating historical features...")
    
    # Create binary no-show indicator
    print("Creating binary no-show indicator...")
    df['no_show'] = (df['No-show'] == 'Yes').astype(int)
    
    # Create clusters for imputation
    print("Creating patient clusters for imputation...")
    patient_clusters, cluster_stats = create_patient_clusters(df)
    
    print("Calculating patient history features...")
    
    def calculate_patient_history(group):
        # Get only historical data before prediction date
        history = group[group['AppointmentDay'] < group['prediction_date']]
        cluster = group['patient_cluster'].iloc[0]
        
        # If no history, use cluster statistics
        if len(history) == 0:
            return pd.Series({
                'patient_noshow_rate': cluster_stats[cluster]['avg_noshow_rate'],
                'patient_noshow_rate_last_3': cluster_stats[cluster]['avg_noshow_rate'],
                'patient_noshow_rate_last_5': cluster_stats[cluster]['avg_noshow_rate'],
                'days_since_last_appointment': 365,
                'total_appointments': 0,
                'appointment_frequency': 0,
                'no_show_streak': 0,
                'show_streak': 0,
                'prev_appointment_was_noshow': 0,
                'avg_lead_time': cluster_stats[cluster]['avg_lead_time'],
                'lead_time_std': 0,
                'weekend_noshow_rate': cluster_stats[cluster]['weekend_noshow_rate'],
                'morning_noshow_rate': cluster_stats[cluster]['morning_noshow_rate'],
                'afternoon_noshow_rate': cluster_stats[cluster]['afternoon_noshow_rate'],
                'seasonal_noshow_rate': cluster_stats[cluster]['seasonal_noshow_rate'],
                'sms_response_rate': cluster_stats[cluster]['sms_response_rate'],
                'cancellation_rate': 0,
                'rescheduling_rate': 0,
                'last_month_appointments': 0,
                'last_week_appointments': 0,
                'appointment_time_consistency': 0,
                'appointment_day_consistency': 0,
                'peak_hour_noshow_rate': cluster_stats[cluster]['morning_noshow_rate'],
                'off_peak_noshow_rate': cluster_stats[cluster]['afternoon_noshow_rate'],
                'recent_sms_response': 0,
                'season_appointment_freq': 0
            })
        
        # Sort history by appointment date
        history = history.sort_values('AppointmentDay')
        
        # Basic statistics
        total_appointments = len(history)
        noshow_rate = history['no_show'].mean()
        
        # Recent history rates
        last_3_rate = history.tail(3)['no_show'].mean() if len(history) >= 3 else noshow_rate
        last_5_rate = history.tail(5)['no_show'].mean() if len(history) >= 5 else noshow_rate
        
        # Time-based features
        current_date = group['prediction_date'].iloc[0]
        days_since_last = (current_date - history['AppointmentDay'].max()).days
        date_range = (history['AppointmentDay'].max() - history['AppointmentDay'].min()).days / 30
        appointment_freq = total_appointments / max(1, date_range)
        
        # Recent appointment frequency
        last_month_mask = history['AppointmentDay'] >= (current_date - pd.Timedelta(days=30))
        last_week_mask = history['AppointmentDay'] >= (current_date - pd.Timedelta(days=7))
        last_month_appointments = history[last_month_mask].shape[0]
        last_week_appointments = history[last_week_mask].shape[0]
        
        # Calculate streaks
        reversed_history = history['no_show'].iloc[::-1]
        no_show_streak = len(list(itertools.takewhile(lambda x: x == 1, reversed_history)))
        show_streak = len(list(itertools.takewhile(lambda x: x == 0, reversed_history)))
        
        # Get last appointment status
        prev_was_noshow = history.iloc[-1]['no_show']
        
        # Lead time statistics
        lead_times = (history['AppointmentDay'] - history['ScheduledDay']).dt.total_seconds() / (24 * 60 * 60)
        avg_lead_time = lead_times.mean()
        lead_time_std = lead_times.std() if len(lead_times) > 1 else 0
        
        # Time of day patterns
        history['hour'] = history['AppointmentDay'].dt.hour
        history['day_of_week'] = history['AppointmentDay'].dt.dayofweek
        history['is_weekend'] = history['day_of_week'] >= 5
        history['is_morning'] = history['hour'] < 12
        history['is_peak_hour'] = (history['hour'].between(8, 10) | history['hour'].between(16, 18))
        
        # Calculate time consistency
        appointment_time_consistency = 1 - (history['hour'].std() / 12) if len(history) > 1 else 0
        appointment_day_consistency = 1 - (history['day_of_week'].std() / 3.5) if len(history) > 1 else 0
        
        # Calculate various no-show rates
        weekend_noshow_rate = history[history['is_weekend']]['no_show'].mean() if history['is_weekend'].any() else noshow_rate
        morning_noshow_rate = history[history['is_morning']]['no_show'].mean() if history['is_morning'].any() else noshow_rate
        afternoon_noshow_rate = history[~history['is_morning']]['no_show'].mean() if (~history['is_morning']).any() else noshow_rate
        peak_hour_noshow_rate = history[history['is_peak_hour']]['no_show'].mean() if history['is_peak_hour'].any() else noshow_rate
        off_peak_noshow_rate = history[~history['is_peak_hour']]['no_show'].mean() if (~history['is_peak_hour']).any() else noshow_rate
        
        # Seasonal patterns
        history['month'] = history['AppointmentDay'].dt.month
        history['season'] = (history['month'] % 12 + 3) // 3
        current_season = (current_date.month % 12 + 3) // 3
        seasonal_noshow_rate = history[history['season'] == current_season]['no_show'].mean() if (history['season'] == current_season).any() else noshow_rate
        season_appointment_freq = history[history['season'] == current_season].shape[0] / max(1, total_appointments)
        
        # SMS response patterns
        sms_history = history[history['SMS_received'] == 1]
        sms_response_rate = 1 - sms_history['no_show'].mean() if len(sms_history) > 0 else 0.5
        recent_sms = sms_history[sms_history['AppointmentDay'] >= (current_date - pd.Timedelta(days=30))]
        recent_sms_response = 1 - recent_sms['no_show'].mean() if len(recent_sms) > 0 else sms_response_rate
        
        # Appointment behavior patterns
        same_day_reschedules = history[lead_times == 0].shape[0] / max(1, total_appointments)
        cancellation_rate = history[lead_times < 0].shape[0] / max(1, total_appointments)
        
        # If only one appointment, combine with cluster statistics
        if len(history) == 1:
            noshow_rate = (noshow_rate + cluster_stats[cluster]['avg_noshow_rate']) / 2
            last_3_rate = noshow_rate
            last_5_rate = noshow_rate
            weekend_noshow_rate = (weekend_noshow_rate + cluster_stats[cluster]['weekend_noshow_rate']) / 2
            morning_noshow_rate = (morning_noshow_rate + cluster_stats[cluster]['morning_noshow_rate']) / 2
            afternoon_noshow_rate = (afternoon_noshow_rate + cluster_stats[cluster]['afternoon_noshow_rate']) / 2
            seasonal_noshow_rate = (seasonal_noshow_rate + cluster_stats[cluster]['seasonal_noshow_rate']) / 2
            sms_response_rate = (sms_response_rate + cluster_stats[cluster]['sms_response_rate']) / 2
        
        return pd.Series({
            'patient_noshow_rate': noshow_rate,
            'patient_noshow_rate_last_3': last_3_rate,
            'patient_noshow_rate_last_5': last_5_rate,
            'days_since_last_appointment': days_since_last,
            'total_appointments': total_appointments,
            'appointment_frequency': appointment_freq,
            'no_show_streak': no_show_streak,
            'show_streak': show_streak,
            'prev_appointment_was_noshow': prev_was_noshow,
            'avg_lead_time': avg_lead_time,
            'lead_time_std': lead_time_std,
            'weekend_noshow_rate': weekend_noshow_rate,
            'morning_noshow_rate': morning_noshow_rate,
            'afternoon_noshow_rate': afternoon_noshow_rate,
            'seasonal_noshow_rate': seasonal_noshow_rate,
            'sms_response_rate': sms_response_rate,
            'cancellation_rate': cancellation_rate,
            'rescheduling_rate': same_day_reschedules,
            'last_month_appointments': last_month_appointments,
            'last_week_appointments': last_week_appointments,
            'appointment_time_consistency': appointment_time_consistency,
            'appointment_day_consistency': appointment_day_consistency,
            'peak_hour_noshow_rate': peak_hour_noshow_rate,
            'off_peak_noshow_rate': off_peak_noshow_rate,
            'recent_sms_response': recent_sms_response,
            'season_appointment_freq': season_appointment_freq
        })

    # Apply calculations in parallel using groupby
    print("Calculating patient features...")
    patient_features = df.groupby('PatientId').parallel_apply(calculate_patient_history)
    df = df.join(patient_features, on='PatientId')
    
    # Create interaction features
    df['noshow_rate_trend'] = df['patient_noshow_rate_last_3'] - df['patient_noshow_rate']
    df['high_risk_patient'] = ((df['patient_noshow_rate'] > 0.5) & 
                              (df['total_appointments'] > 2)).astype(int)
    df['recent_high_risk'] = ((df['patient_noshow_rate_last_3'] > 0.5) & 
                             (df['total_appointments'] > 2)).astype(int)
    df['irregular_schedule'] = (df['lead_time_std'] > df['lead_time_std'].mean()).astype(int)
    df['frequent_rescheduler'] = (df['rescheduling_rate'] > df['rescheduling_rate'].mean()).astype(int)
    df['sms_responsive'] = (df['sms_response_rate'] > 0.7).astype(int)
    
    # Advanced interaction features
    df['engagement_score'] = (df['appointment_frequency'] * 
                            (1 + df['appointment_time_consistency']) * 
                            (1 + df['appointment_day_consistency']))
    
    df['risk_score'] = (df['patient_noshow_rate'] * 
                       (1 + df['noshow_rate_trend']) * 
                       (1 + df['irregular_schedule']))
    
    df['time_risk'] = (df['peak_hour_noshow_rate'] * df['is_rush_hour_appointment'] +
                      df['weekend_noshow_rate'] * df['is_weekend_appointment'])
    
    df['seasonal_risk'] = df['seasonal_noshow_rate'] * df['season_appointment_freq']
    
    print("Historical feature creation complete")
    return df

def engineer_features(df, prediction_days):
    """
    Create features from the raw data with more sophisticated interactions.
    """
    print("\nStarting feature engineering process...")
    
    # Sort by scheduled date first to ensure temporal order
    df = df.sort_index()
    
    print("Creating temporal features...")
    # Create temporal mask for prediction_days before appointment
    df['prediction_date'] = df['AppointmentDay'] - pd.Timedelta(days=prediction_days)
    
    # Filter out appointments that were scheduled after the prediction date
    df = df[df['ScheduledDay'] <= df['prediction_date']]
    print(f"After filtering for {prediction_days} days prediction: {len(df)} appointments")
    
    # Calculate days between scheduled and appointment
    df['lead_time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds() / (24 * 60 * 60)
    df['days_until_appointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['lead_time_squared'] = df['lead_time'] ** 2
    
    # Time-based features
    df['scheduled_hour'] = df['ScheduledDay'].dt.hour
    df['scheduled_day_of_week'] = df['ScheduledDay'].dt.dayofweek
    df['appointment_hour'] = df['AppointmentDay'].dt.hour
    df['appointment_day_of_week'] = df['AppointmentDay'].dt.dayofweek
    df['is_weekend_appointment'] = (df['appointment_day_of_week'] >= 5).astype(int)
    df['is_morning_appointment'] = (df['appointment_hour'] < 12).astype(int)
    df['is_rush_hour_appointment'] = ((df['appointment_hour'] >= 8) & (df['appointment_hour'] <= 10) | 
                                    (df['appointment_hour'] >= 16) & (df['appointment_hour'] <= 18)).astype(int)
    
    # Age-based features
    df['is_child'] = (df['Age'] < 12).astype(int)
    df['is_teen'] = ((df['Age'] >= 12) & (df['Age'] < 18)).astype(int)
    df['is_elderly'] = (df['Age'] >= 65).astype(int)
    df['age_group'] = pd.qcut(df['Age'], q=5, labels=['very_young', 'young', 'middle', 'senior', 'elderly'])
    
    # Appointment timing features
    df['rush_appointment'] = (df['lead_time'] < df['lead_time'].quantile(0.05)).astype(int)
    df['planned_appointment'] = (df['lead_time'] > df['lead_time'].quantile(0.95)).astype(int)
    
    # Health condition features
    df['multiple_conditions'] = ((df['Hipertension'] + df['Diabetes'] + 
                                df['Alcoholism'] + df['Handcap']) >= 2).astype(int)
    df['health_score'] = (df['Hipertension'] * 2 + df['Diabetes'] * 2 + 
                         df['Alcoholism'] * 1.5 + df['Handcap'] * 1.5)
    
    # SMS and scheduling interaction
    df['sms_lead_time'] = df['SMS_received'] * df['lead_time']
    df['late_sms'] = ((df['lead_time'] < 2) & (df['SMS_received'] == 1)).astype(int)
    
    print("Adding historical features...")
    df = create_historical_features(df)
    
    # Create interaction features
    df['risk_score'] = (df['patient_noshow_rate'] * df['health_score'] * 
                       (1 + df['multiple_conditions']) * 
                       (1 + df['rush_appointment']))
    
    df['accessibility_score'] = ((1 - df['is_weekend_appointment']) * 
                               (1 - df['is_rush_hour_appointment']) * 
                               (1 + df['Scholarship']))
    
    df['engagement_score'] = ((1 - df['patient_noshow_rate']) * 
                            (1 + df['sms_response_rate']) * 
                            (1 + df['appointment_frequency']))
    
    # Temporal patterns
    df['seasonal_risk'] = df['seasonal_noshow_rate'] * df['patient_noshow_rate']
    df['time_risk'] = ((df['is_weekend_appointment'] * df['weekend_noshow_rate']) + 
                      (df['is_morning_appointment'] * df['morning_noshow_rate']))
    
    print("Feature engineering complete")

    # Select the most important features based on domain knowledge and feature importance
    selected_columns = [
        'days_until_appointment',
        'lead_time',
        'lead_time_squared',
        'patient_noshow_rate',
        'patient_noshow_rate_last_3',
        'patient_noshow_rate_last_5',
        'days_since_last_appointment',
        'total_appointments',
        'appointment_frequency',
        'no_show_streak',
        'show_streak',
        'prev_appointment_was_noshow',
        'noshow_rate_trend',
        'high_risk_patient',
        'recent_high_risk',
        'is_child',
        'is_teen',
        'is_elderly',
        'rush_appointment',
        'planned_appointment',
        'Scholarship',
        'Age',
        'Alcoholism',
        'multiple_conditions',
        'health_score',
        'sms_lead_time',
        'late_sms',
        'risk_score',
        'accessibility_score',
        'engagement_score',
        'seasonal_risk',
        'time_risk',
        'is_weekend_appointment',
        'is_morning_appointment',
        'is_rush_hour_appointment',
        'sms_response_rate',
        'weekend_noshow_rate',
        'morning_noshow_rate',
        'seasonal_noshow_rate'
    ]

    return df[selected_columns], df['No-show'].replace({'Yes': 1, 'No': 0})

def cost_metric(y_true, 
                  y_pred_proba, 
                  threshold,
                  fn_cost=1,
                  fp_cost=2,
                  tp_cost=0,
                  tn_cost=0):
    """
    Calculate custom metric:
    fn_cost for FN (default: -1)
    fp_cost for FP (default: -2)    
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    threshold : float
        Classification threshold
    fn_cost : float
        Cost of false negatives
    fp_cost : float
        Cost of false positives
        
    Returns
    -------
    float
        Custom metric score (improvement over baseline)
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix elements
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    # Calculate model's cost
    model_cost = (FN * fn_cost) + (FP * fp_cost) + (TP * tp_cost) + (TN * tn_cost)
    # Calculate baseline cost (predicting all negatives)
    n_positives = np.sum(y_true == 1)
    baseline_cost = n_positives * fn_cost  # Cost of missing all actual positives
    
    # Avoid division by zero
    if baseline_cost == 0:
        return 0
    
    # Calculate improvement over baseline
    improvement = (baseline_cost - model_cost) / abs(baseline_cost)
    
    return improvement

def cost_of_double_appointment(y_true, y_pred_proba, threshold):
    """
    Calculate cost of double appointment strategy
    FN cost = 1 (missed no-show)
    FP cost = 2 (unnecessary double booking)
    """
    return cost_metric(y_true, y_pred_proba, threshold, fn_cost=1, fp_cost=2)

def cost_of_giving_a_call(y_true, y_pred_proba, threshold):
    """
    Calculate cost of giving a call strategy
    FN cost = 1 (missed no-show)
    FP cost = 0.1 (unnecessary call)
    TP cost = 0.4 (the call only raise the chance of showing up 40%
    """
    return cost_metric(y_true, y_pred_proba, threshold, fn_cost=1, fp_cost=0.1, tp_cost=0.6)

def find_optimal_threshold(model, X_test, y_test, prediction_days):
    """
    Find optimal threshold by maximizing custom metrics (doctor time savings)
    for both double appointment and call strategies
    """
    print("\nFinding optimal classification thresholds...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Search full probability range
    thresholds = np.linspace(0, 1, 100)
    double_appointment_scores = []
    call_scores = []
    precisions = []
    recalls = []
    
    print("Testing different threshold values...")
    # Calculate scores for all thresholds
    for threshold in thresholds:
        # Calculate scores for both strategies
        double_score = cost_of_double_appointment(y_test, y_pred_proba, threshold)
        call_score = cost_of_giving_a_call(y_test, y_pred_proba, threshold)
        double_appointment_scores.append(double_score * 100)
        call_scores.append(call_score * 100)
        
        # Calculate precision and recall
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
    
    # Find best thresholds
    double_best_idx = np.argmax(double_appointment_scores)
    call_best_idx = np.argmax(call_scores)
    
    double_best_threshold = thresholds[double_best_idx]
    call_best_threshold = thresholds[call_best_idx]
    
    double_best_score = double_appointment_scores[double_best_idx] / 100
    call_best_score = call_scores[call_best_idx] / 100
    
    # Calculate metrics for both thresholds
    def calculate_metrics(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        return tp, fp, tn, fn, recall, specificity, precision
    
    # Calculate metrics for both strategies
    double_metrics = calculate_metrics(double_best_threshold)
    call_metrics = calculate_metrics(call_best_threshold)
    
    # Print results for both strategies
    print("\nDouble Appointment Strategy:")
    print(f"Optimal threshold: {double_best_threshold:.3f}")
    print(f"Cost savings: {double_best_score*100:.1f}%")
    print(f"True Positives: {double_metrics[0]}")
    print(f"False Positives: {double_metrics[1]}")
    print(f"True Negatives: {double_metrics[2]}")
    print(f"False Negatives: {double_metrics[3]}")
    print(f"Recall: {double_metrics[4]:.3f}")
    print(f"Specificity: {double_metrics[5]:.3f}")
    print(f"Precision: {double_metrics[6]:.3f}")
    
    print("\nCall Strategy:")
    print(f"Optimal threshold: {call_best_threshold:.3f}")
    print(f"Cost savings: {call_best_score*100:.1f}%")
    print(f"True Positives: {call_metrics[0]}")
    print(f"False Positives: {call_metrics[1]}")
    print(f"True Negatives: {call_metrics[2]}")
    print(f"False Negatives: {call_metrics[3]}")
    print(f"Recall: {call_metrics[4]:.3f}")
    print(f"Specificity: {call_metrics[5]:.3f}")
    print(f"Precision: {call_metrics[6]:.3f}")
    
    print("\nGenerating visualization plots...")
    
    # Plot 1: Cost savings for both strategies
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, double_appointment_scores, '-b', linewidth=2, label='Double Appointment Strategy')
    plt.plot(thresholds, call_scores, '-g', linewidth=2, label='Call Strategy')
    plt.axvline(x=double_best_threshold, color='b', linestyle='--',
                label=f'Double Apt threshold = {double_best_threshold:.2f}\nSavings = {double_best_score*100:.1f}%')
    plt.axvline(x=call_best_threshold, color='g', linestyle='--',
                label=f'Call threshold = {call_best_threshold:.2f}\nSavings = {call_best_score*100:.1f}%')
    plt.ylim(-50, 50)
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Cost Savings (%)', fontsize=12)
    plt.title('Impact on Costs\n(compared to no intervention baseline)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/cost_savings.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot 2: ROC curve with both operating points
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    # Add operating points with more detailed labels
    double_fpr = double_metrics[1] / (double_metrics[1] + double_metrics[2])
    double_tpr = double_metrics[0] / (double_metrics[0] + double_metrics[3])
    plt.plot(double_fpr, double_tpr, 'bo', markersize=10,
             label=f'Double Apt (TPR={double_tpr:.2f}, FPR={double_fpr:.2f})')
    
    call_fpr = call_metrics[1] / (call_metrics[1] + call_metrics[2])
    call_tpr = call_metrics[0] / (call_metrics[0] + call_metrics[3])
    plt.plot(call_fpr, call_tpr, 'go', markersize=10,
             label=f'Call Strategy (TPR={call_tpr:.2f}, FPR={call_fpr:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve with Operating Points\nfor Both Intervention Strategies', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/roc_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 3: Precision-Recall vs Threshold
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, '-b', linewidth=2, label='Precision')
    plt.plot(thresholds, recalls, '-r', linewidth=2, label='Recall')
    plt.axvline(x=double_best_threshold, color='orange', linestyle='--',
                label=f'Double Apt (P={double_metrics[6]:.2f}, R={double_metrics[4]:.2f})')
    plt.axvline(x=call_best_threshold, color='g', linestyle='--',
                label=f'Call (P={call_metrics[6]:.2f}, R={call_metrics[4]:.2f})')
    plt.xlabel('Classification Threshold', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.title('Precision and Recall vs Threshold\nTrade-off Analysis', fontsize=20, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/precision_recall_threshold.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 4: Precision-Recall curve
    plt.figure(figsize=(12, 8))
    plt.plot(recalls, precisions, color='purple', lw=2,
             label='Precision-Recall curve')
    
    # Calculate precision and recall for both thresholds
    double_precision = double_metrics[6]  # precision is the 7th metric
    double_recall = double_metrics[4]     # recall is the 5th metric
    call_precision = call_metrics[6]
    call_recall = call_metrics[4]
    
    # Add operating points with detailed metrics
    plt.plot(double_recall, double_precision, 'bo', markersize=10,
             label=f'Double Apt\nPrecision={double_precision:.2f}\nRecall={double_recall:.2f}\nThreshold={double_best_threshold:.2f}')
    plt.plot(call_recall, call_precision, 'go', markersize=10,
             label=f'Call Strategy\nPrecision={call_precision:.2f}\nRecall={call_recall:.2f}\nThreshold={call_best_threshold:.2f}')
    
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curve with Operating Points\nModel Performance at Different Decision Thresholds', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/precision_recall_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Generating probability distribution plot...")
    plt.figure(figsize=(12, 8))
    sns.histplot(data=pd.DataFrame({'probability': y_pred_proba, 'target': y_test}), 
                x='probability', hue='target', bins=50)
    plt.axvline(x=double_best_threshold, color='b', linestyle='--',
                label=f'Double Apt threshold = {double_best_threshold:.2f}')
    plt.axvline(x=call_best_threshold, color='g', linestyle='--',
                label=f'Call threshold = {call_best_threshold:.2f}')
    plt.title('Distribution of Predicted Probabilities\nby Actual No-show Status', fontsize=14, pad=20)
    plt.xlabel('Predicted Probability of No-show', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Actual No-show', title_fontsize=10, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/predicted_probabilities_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Generating confusion matrices...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Double appointment strategy confusion matrix
    sns.heatmap(confusion_matrix(y_test, (y_pred_proba >= double_best_threshold).astype(int)), 
                annot=True, fmt='d', cmap='Blues', ax=ax1,
                annot_kws={'size': 12})
    ax1.set_title('Confusion Matrix\nDouble Appointment Strategy\n' + 
                  f'Threshold = {double_best_threshold:.2f}', fontsize=14, pad=20)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    
    # Call strategy confusion matrix
    sns.heatmap(confusion_matrix(y_test, (y_pred_proba >= call_best_threshold).astype(int)), 
                annot=True, fmt='d', cmap='Blues', ax=ax2,
                annot_kws={'size': 12})
    ax2.set_title('Confusion Matrix\nCall Strategy\n' + 
                  f'Threshold = {call_best_threshold:.2f}', fontsize=14, pad=20)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/confusion_matrices.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return double_best_threshold, call_best_threshold, (X_test, y_test)

def train_model(X, y, full_run=True, prediction_days=7):
    """
    Train an XGBoost model with improved configuration.
    """
    print("\nStarting model training process...")
    
    print("Splitting data into train and test sets...")
    # Use index to split the data into train and test
    train_idx = X.index[:int(0.8 * len(X))]
    test_idx = X.index[int(0.8 * len(X)):]
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    time_index_train = X_train.index
    time_index_test = X_test.index
    
    print("Scaling features...")
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Using original class distribution (no SMOTE)...")
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    print("Configuring logistic regression model...")
    # logistic regression model
    # model_constructor = LogisticRegression
    # model_config = dict(
    #     penalty='l1',
    #     solver='liblinear',
    #     random_state=42,
    # )

    # # kernel svm model
    # model_constructor = SVC
    # model_config = dict(
    #     kernel='rbf',
    #     random_state=42,
    # )
    
# 

    # # random forest model
    # model_constructor = RandomForestClassifier
    # model_config = dict(
    #     n_estimators=1000,
    #     max_depth=18,
    #     min_samples_split=2,
    #     random_state=42,
    #     verbose=3,
    #     n_jobs=-1
    # )
    
    # # xgboost model
    model_constructor = XGBClassifier
    model_config = dict(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.005,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        scale_pos_weight=5,
        reg_alpha=0.1,
        reg_lambda=1,
        tree_method='hist',
        grow_policy='lossguide',
        max_leaves=128,
        max_bin=512,
        eval_metric=['auc', 'error', 'logloss'],
        early_stopping_rounds=150,
        random_state=42,
        n_jobs=-1,
        verbosity=2
    )

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print(f"Class weights: {class_weights}")

        # Create a validation set from training data by splitting on index
    n_samples = len(X_train_balanced)
    n_val = int(0.2 * n_samples)  # 20% for validation
    val_indices = np.arange(n_samples - n_val, n_samples)
    train_indices = np.arange(0, n_samples - n_val)
    
    X_train_final = X_train_balanced[train_indices]
    X_val = X_train_balanced[val_indices] 
    y_train_final = y_train_balanced[train_indices]
    y_val = y_train_balanced[val_indices]

    
    if full_run:
        print("Creating base model for cross-validation...")
        # Enhanced hyperparameter search space
        param_dist = {
            'n_estimators': [3000, 4000, 5000],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.003, 0.005, 0.01],
            'min_child_weight': [2, 3, 4],
            'gamma': [0.1, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'colsample_bylevel': [0.7, 0.8, 0.9],
            'colsample_bynode': [0.7, 0.8, 0.9],
            'scale_pos_weight': [4, 5, 6],
            'reg_alpha': [0.05, 0.1, 0.2],
            'reg_lambda': [0.5, 1, 2],
            'max_leaves': [64, 128, 256],
            'max_bin': [256, 512, 1024]
        }
        
        # Create RandomizedSearchCV object
        random_search = RandomizedSearchCV(
            estimator=XGBClassifier(
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method='hist',
                grow_policy='lossguide'
            ),
            param_distributions=param_dist,
            n_iter=30,  # Increased number of iterations
            scoring='roc_auc',
            cv=TimeSeriesSplit(n_splits=5),
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit RandomizedSearchCV
        random_search.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_train_final, y_train_final), (X_val, y_val)],
            verbose=False
        )
        
        # Print best parameters and score
        print("\nBest parameters found:")
        print(random_search.best_params_)
        print(f"\nBest cross-validation score: {random_search.best_score_:.3f}")
        
        # Update model_config with best parameters
        model_config.update(random_search.best_params_)
        print("\nUpdated model configuration:")
        print(model_config)


    
    final_model = model_constructor(**model_config)
    
    # Train final model with early stopping
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=[(X_train_final, y_train_final), (X_val, y_val)],
        verbose=True
    )
    
    # Print training vs validation performance
    print("\nTraining Performance:")
    y_train_pred = final_model.predict(X_train_final)
    print(classification_report(y_train_final, y_train_pred))
    
    print("\nValidation Performance:")
    y_val_pred = final_model.predict(X_val)
    print(classification_report(y_val, y_val_pred))
    
    # Find optimal thresholds with modified threshold search
    double_threshold, call_threshold, test_data = find_optimal_threshold(final_model, X_test_scaled, y_test, prediction_days)
    
    return final_model, scaler, (double_threshold, call_threshold), test_data

def save_model(model, scaler, thresholds, output_dir='models', prediction_days=7):
    """
    Save the model, scaler, and best thresholds.
    
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    thresholds : tuple
        Tuple containing (double_appointment_threshold, call_threshold)
    output_dir : str
        Directory to save the model files
    """
    # Save model
    model_filename = f"{output_dir}/medical_noshow_model_{prediction_days}_days.joblib"
    joblib.dump(model, model_filename)

    # Save scaler
    scaler_filename = f"{output_dir}/medical_noshow_scaler_{prediction_days}_days.joblib"
    joblib.dump(scaler, scaler_filename)
    
    # Save thresholds and other parameters
    params = {
        'double_appointment_threshold': thresholds[0],
        'call_threshold': thresholds[1],
        'model_params': model.get_params(),
        'feature_names': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None,
    }
    
    params_filename = f"{output_dir}/medical_noshow_params.joblib"
    joblib.dump(params, params_filename)
    
    print(f"\nModel saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")
    print(f"Parameters saved as: {params_filename}")

def predict_with_threshold(model, X, threshold=0.3):
    """Adjust prediction threshold for better recall"""
    y_pred_proba = model.predict_proba(X)
    return (y_pred_proba[:, 1] >= threshold).astype(int)

# %%
import os
os.chdir('/mnt/c/Users/ozmen/OneDrive - Reichman University/Courses/Mini Semester 6/Machine Learning for Business/final_project')
# %%
base_dir = 'data/brazil_short_term/'
overwrite = True
full_run = False

# Load and preprocess data
print("Loading data...")
df = load_data(os.path.join(base_dir, 'raw/KaggleV2-May-2016_csv'))

# %%
# plot pairplot with target as hue take 1000 rows
plt.figure(figsize=(10, 6))
sns.pairplot(df.sample(1000), hue='No-show')
plt.title('Pairplot of the subset of the data (1000)')
plt.savefig('models/pairplot.png')
plt.close()

# %%
# plot average daily no-show rate by SMS_received
plt.figure(figsize=(10, 6))
values = df.groupby(['SMS_received', 'AppointmentDay'])['No-show'].value_counts().unstack().reset_index()
values['no_show_rate'] = values['Yes'] / (values['Yes'] + values['No'])
sns.boxplot(x='SMS_received', y='no_show_rate', data=values, hue='SMS_received')
plt.xlabel('SMS Received')
plt.ylabel('Average Daily No-Show Rate')
plt.title('Average Daily No-Show Rate by SMS Received')
plt.savefig('models/sms_received_boxplot.png')
plt.close()

# Engineer features
# %%
# Engineer features
print("Engineering features...")
# add nano second based on the count of each day to make it a unique index
df['appointment_daily_id'] = df.groupby('AppointmentDay').cumcount()
df.index = df['AppointmentDay'] + pd.to_timedelta(df['appointment_daily_id'], unit='ns')

prediction_days = [7, 5, 3, 1]
for prediction_days in prediction_days:
    print("="*100)
    print("\n")
    print("="*100)
    print(f"Processing {prediction_days} days...")
    print("="*100)
    print("\n")
    print("="*100)
    run_path = os.path.join(f'models/{prediction_days}_days')
    os.makedirs(run_path, exist_ok=True)
    print(f"Engineering features for {prediction_days} days...")
    features_path = os.path.join(run_path, 'features')
    os.makedirs(features_path, exist_ok=True)
    if not os.path.exists(os.path.join(features_path, 'X.parquet')) or overwrite:
        X, y = engineer_features(df, prediction_days)
        X.to_parquet(os.path.join(features_path, 'X.parquet'))
        y.to_frame().to_parquet(os.path.join(features_path, 'y.parquet'))
    else:
        X, y = pd.read_parquet(os.path.join(features_path, 'X.parquet')), pd.read_parquet(os.path.join(features_path, 'y.parquet'))
        y = y.iloc[:, 0]


    # Impute missing values (later will be based on clustering)
    print("Imputing missing values...")
    ## replacing inf with non
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    if full_run:
        # clustering
        # kmeans
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        # cluster the data with kmeans and plot the clusters on pca
        #scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)

        kmeans = KMeans(n_clusters=2, random_state=42)
        df_scaled['cluster'] = kmeans.fit_predict(X_scaled)
        df_scaled['No-show'] = y.values
        # plot the clusters on pca
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['pca1', 'pca2'])
        df_pca['No-show'] = y.values
        df_pca['cluster'] = df_scaled['cluster']
        plt.figure(figsize=(10, 6))
        sns.scatterplot(df_pca, x='pca1', y='pca2', hue='No-show', style='cluster')
        plt.title('KMeans Clusters')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.savefig(f'models/{prediction_days}_days/kmeans_pca_clusters.png')
        plt.close()

        # plot the clusters on t-sne
        from sklearn.manifold import TSNE
        # Sample 5000 points for t-SNE visualization
        sample_size = 5000
        sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_scaled_sample = X_scaled[sample_idx]
        df_sample = df_scaled.iloc[sample_idx]

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled_sample)
        df_tsne = pd.DataFrame(X_tsne, columns=['tsne1', 'tsne2'])
        df_tsne['No-show'] = y[sample_idx].values
        df_tsne['cluster'] = df_sample['cluster']
        plt.figure(figsize=(10, 6))
        sns.scatterplot(df_tsne, x='tsne1', y='tsne2', hue='No-show', style='cluster')
        plt.title('t-SNE Clusters (5000 samples)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(f'models/{prediction_days}_days/tsne_pca_clusters.png')
        plt.close()

        # dbscan
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=2, min_samples=5)
        df_pca['cluster'] = dbscan.fit_predict(X_scaled)
        df_tsne['cluster'] = df_pca.loc[sample_idx, 'cluster']

        # plot the clusters on pca
        plt.figure(figsize=(10, 6))
        sns.scatterplot(df_pca, x='pca1', y='pca2', hue='No-show', style='cluster')
        plt.title('DBSCAN Clusters')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.savefig(f'models/{prediction_days}_days/dbscan_pca_clusters.png')
        plt.close()

        # plot the clusters on t-sne
        plt.figure(figsize=(10, 6))
        sns.scatterplot(df_tsne, x='tsne1', y='tsne2', hue='No-show', style='cluster')
        plt.title('DBSCAN Clusters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(f'models/{prediction_days}_days/dbscan_tsne_clusters.png')
        plt.close()


        # %%
        # do classification using the k means clusters (split the data into train and test )
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # redo the k means clustering on the scaled train data
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train_scaled)

        # map the  number of clusters to the target column based on the cluster with the highest mean of the target column
        cluster_num = kmeans.predict(X_train_scaled)

        test_labels = kmeans.predict(X_test_scaled)

        # do classification report on the k means clusters using the target column
        from sklearn.metrics import classification_report
        print("Classification Report on KMeans Clusters:")
        print(classification_report(y_test, test_labels))


    # %%k
    # Train model
    print("Training model...")
    model, scaler, best_threshold, test_data = train_model(X, y, full_run=full_run, prediction_days=prediction_days)
    # %%
    # Save model
    print("Saving model...")
    save_model(model, scaler, best_threshold, prediction_days=prediction_days)

    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0] if hasattr(model, 'coef_') else None
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Save the plot
    plt.figure(figsize=(10, 6))
    feature_importance.plot(x='feature', y='importance', kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    # plot top 10 features
    plt.figure(figsize=(10, 6))
    feature_importance.head(10).plot(x='feature', y='importance', kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'models/{prediction_days}_days/feature_importance_top10.png')
    plt.close()
    # save the feature importance to a csv
    feature_importance.to_csv(f'models/{prediction_days}_days/feature_importance.csv', index=False)
        


# %%
