import sys
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
import yaml
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from app import load_config
from app.core.data_loaders.factory import get_loader
from app.core.data_processors.factory import get_processor
from app.core.feature_creators.factory import get_feature_creator
from app.core.models.factory import get_model
from app.utils import get_logger
from app.core.evaluation.metrics import ModelEvaluator

logger = get_logger(__name__)

def setup_mlflow(config: Dict[str, Any]):
    """
    Setup MLflow tracking.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """
    # Set MLflow tracking URI if specified
    if "mlflow" in config and "tracking_uri" in config["mlflow"]:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    # Set experiment name
    experiment_name = config.get("mlflow", {}).get("experiment_name", "no_show_prediction")
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Error setting up MLflow experiment: {str(e)}")
        # Fall back to default experiment
        pass

def log_feature_importance(feature_importance: pd.Series, artifact_path: str):
    """
    Log feature importance plot and data to MLflow.

    Parameters
    ----------
    feature_importance : pd.Series
        Feature importance series
    artifact_path : str
        Path to save artifacts
    """
    import matplotlib.pyplot as plt
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance.head(20).plot(kind='barh')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(artifact_path) / "feature_importance.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    # Log plot and data
    mlflow.log_artifact(str(plot_path))
    
    # Save and log feature importance data
    csv_path = Path(artifact_path) / "feature_importance.csv"
    feature_importance.to_csv(csv_path)
    mlflow.log_artifact(str(csv_path))

def prepare_data(
    config_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Prepare data for training by running the data pipeline.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with features
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize and run data loader
    logger.info("Loading data...")
    loader_name = config['data_loading'].get('data_loader_name')
    loader_class = get_loader(data_loader_name=loader_name)
    loader = loader_class(config=config['data_loading'])
    df = loader.load_data()
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Initialize and run data processor
    logger.info("Processing data...")
    processor_name = config['data_processing'].get('data_processor_name')
    processor_class = get_processor(data_processor_name=processor_name)
    processor = processor_class(config=config['data_processing'])
    processed_df = processor.process_data(df)
    logger.info(f"Processed data shape: {processed_df.shape}")
    
    # Initialize and run feature creator
    logger.info("Creating features...")
    creator_name = config['feature_creation'].get('feature_creator_name')
    creator_class = get_feature_creator(feature_creator_name=creator_name)
    creator = creator_class(config=config['feature_creation'])
    featured_df = creator.create_features(processed_df)
    logger.info(f"Final data shape with features: {featured_df.shape}")
    
    return featured_df

def preprocess_features(
    data: pd.DataFrame,
    categorical_cols: List[str],
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Preprocess features by encoding categorical variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    categorical_cols : List[str]
        List of categorical column names
    encoders : Dict[str, LabelEncoder], optional
        Pre-fitted label encoders, by default None
    fit : bool, optional
        Whether to fit the encoders, by default True

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, LabelEncoder]]
        Preprocessed DataFrame and fitted encoders
    """
    df = data.copy()
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        if fit:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
        else:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError as e:
                    logger.warning(f"Error encoding {col}: {str(e)}")
                    # Handle unseen categories by assigning a default value
                    df[col] = -1
    
    return df, encoders

def filter_features(
    data: pd.DataFrame,
    target_col: str,
    exclude_patterns: List[str] = None,
    preserve_columns: List[str] = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Filter features for model training.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    target_col : str
        Target column name
    exclude_patterns : List[str], optional
        Patterns to exclude from features, by default None
    preserve_columns : List[str], optional
        Columns to preserve regardless of patterns, by default None

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        Filtered DataFrame, numeric features, and categorical features
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '_id$',  # ID columns
            'reason',  # Text columns
            'date',  # Date columns (except preserved ones)
            'name',  # Name columns
            'location',  # Location columns
            'neighborhood'  # Location columns
        ]
    
    if preserve_columns is None:
        preserve_columns = []
    
    # Get numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['category', 'bool', 'object']).columns
    datetime_cols = data.select_dtypes(include=['datetime64']).columns
    
    # Combine valid feature columns
    valid_cols = list(numeric_cols) + list(categorical_cols) + list(datetime_cols)
    
    # Remove target column from features if it's in the list
    if target_col in valid_cols:
        valid_cols.remove(target_col)
    
    # Remove columns matching exclude patterns, but preserve specified columns
    import re
    filtered_cols = []
    for col in valid_cols:
        if col in preserve_columns:
            filtered_cols.append(col)
        else:
            exclude = False
            for pattern in exclude_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    exclude = True
                    break
            if not exclude:
                filtered_cols.append(col)
    
    # Split features into numeric and categorical
    numeric_features = [col for col in filtered_cols if col in numeric_cols]
    categorical_features = [col for col in filtered_cols if col in categorical_cols]
    
    logger.info(f"Selected {len(filtered_cols)} features for training")
    logger.info("Feature types:")
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    # Include target column in the returned DataFrame
    return data[filtered_cols + [target_col]], numeric_features, categorical_features

def optimize_model_hyperparameters(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Optimize model hyperparameters using time series cross-validation.

    Parameters
    ----------
    model : Any
        Model instance to optimize
    X : pd.DataFrame
        Feature DataFrame
    y : pd.Series
        Target series
    cv_splits : List[Tuple[np.ndarray, np.ndarray]]
        Time series cross-validation splits
    categorical_features : List[str]
        List of categorical feature names

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # Preprocess features before optimization
    X_processed, encoders = preprocess_features(X, categorical_features)
    
    # Define parameter space
    param_space = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(range(10, 50, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample'],
        'max_samples': uniform(0.5, 0.5)  # Bootstrap sample size
    }
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=50,
        scoring='f1',
        n_jobs=-1,
        cv=cv_splits,
        verbose=1,
        random_state=42
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_processed, y)
    
    # Log best parameters and score
    logger.info("\nBest Hyperparameters:")
    for param, value in random_search.best_params_.items():
        logger.info(f"{param}: {value}")
    logger.info(f"Best CV score: {random_search.best_score_:.4f}")
    
    # Log to MLflow
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("best_cv_score", random_search.best_score_)
    
    return random_search.best_params_

def train_model(
    config_path: Union[str, Path],
    data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Train model using prepared data with time series cross-validation.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file
    data : pd.DataFrame
        Prepared DataFrame with features

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model, encoders, and metrics
    """
    # Load configuration
    config = load_config(config_path)
    model_config = config.get('model', {})
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id
        model_save_config = model_config.get('save_model', {})
        artifact_path = Path(model_save_config.get('path', 'artifacts'))
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Filter features
        target_col = model_config.get('target_column', 'no_show')
        exclude_patterns = model_config.get('exclude_patterns', None)
        
        # Ensure date columns are preserved
        if exclude_patterns is None:
            exclude_patterns = []
        exclude_patterns = [p for p in exclude_patterns if p != 'date']
        
        filtered_data, numeric_features, categorical_features = filter_features(
            data.copy(), 
            target_col, 
            exclude_patterns,
            preserve_columns=['appointment_date', 'schedule_date']
        )
        
        # Log data stats
        mlflow.log_param("n_samples", len(filtered_data))
        mlflow.log_param("n_features", len(numeric_features) + len(categorical_features))
        
        # Log class distribution
        class_dist = filtered_data[target_col].value_counts(normalize=True)
        for label, freq in class_dist.items():
            mlflow.log_metric(f"class_dist_{label}", freq)
        
        # Prepare features
        feature_columns = [col for col in filtered_data.columns 
                         if col not in [target_col]]
        X = filtered_data[feature_columns].copy()
        y = filtered_data[target_col].copy()
        
        # Initialize evaluator with time series settings
        evaluator = ModelEvaluator(config={
            'n_splits': config.get('evaluation', {}).get('n_splits', 5),
            'gap': config.get('evaluation', {}).get('gap', 7),
            'test_size': config.get('evaluation', {}).get('test_size', 30),
            'artifact_path': str(artifact_path)
        })
        
        # Get time series splits
        splits = evaluator.create_time_series_splits(
            X,
            y,
            'appointment_date'
        )
        
        # Initialize base model
        model_name = model_config.get('model_name', 'random_forest')
        model_class = get_model(model_name)
        base_model = model_class(config=model_config)
        
        # Prepare features for model training (excluding date columns)
        model_features = [col for col in feature_columns 
                         if col not in ['appointment_date', 'schedule_date']]
        X_model = X[model_features].copy()
        
        # Optimize hyperparameters
        logger.info("\nOptimizing hyperparameters...")
        best_params = optimize_model_hyperparameters(
            base_model.model,
            X_model,
            y,
            splits,
            categorical_features
        )
        
        # Update model with best parameters
        model_config['model_params'].update(best_params)
        model = model_class(config=model_config)
        
        # Perform time series cross-validation
        logger.info("\nStarting time series cross-validation with optimized model...")
        cv_metrics = evaluator.evaluate_time_series(
            model=model.model,
            X=X,
            y=y,
            date_column='appointment_date',
            categorical_features=categorical_features
        )
        
        # Train final model on all data (excluding date columns)
        logger.info("\nTraining final model on all data...")
        X_processed, encoders = preprocess_features(X_model, categorical_features)
        model.train(X_processed, y)
        
        # Save artifacts and return results
        feature_lists = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_names': X_processed.columns.tolist()
        }
        
        save_artifacts(
            model=model,
            encoders=encoders,
            feature_lists=feature_lists,
            artifact_path=artifact_path,
            evaluator=evaluator
        )
        
        return {
            'model': model,
            'encoders': encoders,
            'feature_lists': feature_lists,
            'cv_metrics': cv_metrics,
            'mlflow_run_id': mlflow_run_id,
            'artifact_path': str(artifact_path)
        }

def save_artifacts(
    model: Any,
    encoders: Dict[str, Any],
    feature_lists: Dict[str, Any],
    artifact_path: Path,
    evaluator: Any
) -> None:
    """Save model artifacts and log to MLflow."""
    # Create artifacts directory if it doesn't exist
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    # Save model using MLflow
    mlflow_model_path = artifact_path / "models"
    if mlflow_model_path.exists():
        import shutil
        shutil.rmtree(mlflow_model_path)
    
    # Save model with MLflow format
    mlflow.sklearn.save_model(
        model.model,
        str(mlflow_model_path)
    )
    
    # Save model separately as joblib for backup
    model_path = artifact_path / "model.joblib"
    joblib.dump(model, model_path)
    
    # Save encoders
    encoder_path = artifact_path / "encoders.joblib"
    joblib.dump(encoders, encoder_path)
    
    # Save feature lists
    feature_list_path = artifact_path / "feature_lists.joblib"
    joblib.dump(feature_lists, feature_list_path)
    
    # Save evaluation plots
    plot_path = artifact_path / "plots"
    if plot_path.exists():
        import shutil
        shutil.rmtree(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    evaluator.save_all_plots(plot_path)
    
    # Save feature importance if available
    if hasattr(model.model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_lists['feature_names'],
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance CSV
        importance_path = artifact_path / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        # Create and save feature importance plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=feature_importance.head(20),
            y='feature',
            x='importance',
            palette='viridis'
        )
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save feature importance plot
        plot_importance_path = plot_path / "feature_importance.png"
        plt.savefig(plot_importance_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    # Save model configuration
    if hasattr(model, 'config'):
        config_path = artifact_path / "model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(model.config, f)
    
    # Save model parameters
    if hasattr(model.model, 'get_params'):
        params_path = artifact_path / "model_params.yaml"
        with open(params_path, 'w') as f:
            yaml.dump(model.model.get_params(), f)
    
    # Log all artifacts to MLflow
    mlflow.log_artifacts(str(artifact_path))

def main(
    config_path: Union[str, Path]
):
    """Main function to run the training pipeline."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Record start time
        start_time = datetime.now()
        logger.info(f"Training pipeline started at: {start_time}")
        
        # Prepare data
        logger.info("Preparing data...")
        data = prepare_data(config_path)
        
        # Train model
        logger.info("Training model...")
        results = train_model(config_path, data)
        
        # Record end time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\nTraining pipeline completed at: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"MLflow Run ID: {results['mlflow_run_id']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main(
        "artifacts/california_synthetic/config.yaml"
    )