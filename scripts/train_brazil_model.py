"""
Train the Brazil no-show prediction model.
"""
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.train_model import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Brazil model training started at: {start_time}")
    
    try:
        # Train model using Brazil configuration
        results = main(
            config_path="artifacts/brazil_long_term/config.yaml"
        )
        
        # Record end time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\nBrazil model training completed at: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"MLflow Run ID: {results['mlflow_run_id']}")
        
    except Exception as e:
        logger.error(f"Error in Brazil model training: {str(e)}")
        raise 