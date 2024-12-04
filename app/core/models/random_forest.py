from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from app.core.models.sklearn_base import SklearnBaseModel
from app.utils import get_logger

logger = get_logger(__name__)

class RandomForestModel(SklearnBaseModel):
    """Random Forest model implementation."""
    
    def _setup_model(self):
        """Setup Random Forest model with configuration parameters."""
        model_params = self.config['model_params']
        
        # Get model parameters with defaults
        n_estimators = model_params.get('n_estimators', 100)
        max_depth = model_params.get('max_depth', None)
        min_samples_split = model_params.get('min_samples_split', 2)
        min_samples_leaf = model_params.get('min_samples_leaf', 1)
        max_features = model_params.get('max_features', 'sqrt')
        bootstrap = model_params.get('bootstrap', True)
        random_state = model_params.get('random_state', 42)
        n_jobs = model_params.get('n_jobs', -1)
        class_weight = model_params.get('class_weight', 'balanced')
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            verbose=1
        )
        
        logger.info("Random Forest model initialized with parameters:")
        logger.info(f"n_estimators: {n_estimators}")
        logger.info(f"max_depth: {max_depth}")
        logger.info(f"min_samples_split: {min_samples_split}")
        logger.info(f"min_samples_leaf: {min_samples_leaf}")
        logger.info(f"max_features: {max_features}")
        logger.info(f"bootstrap: {bootstrap}")
        logger.info(f"class_weight: {class_weight}")
        logger.info(f"n_jobs: {n_jobs}") 