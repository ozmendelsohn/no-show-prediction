import sys
from pathlib import Path
from typing import Union
import logging
import pandas as pd

from app import load_config
from app.core.data_loaders.factory import get_loader
from app.core.data_processors.factory import get_processor
from app.core.feature_creators.factory import get_feature_creator
from app.utils import get_logger

logger = get_logger(__name__)

def test_data_pipeline(
    config_path: Union[str, Path]
):
    """
    Test data loading, processing, and feature creation pipeline.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Initialize and run data loader
        logger.info("Initializing data loader...")
        loader_name = config['data_loading'].get('data_loader_name')
        loader_class = get_loader(data_loader_name=loader_name)
        loader = loader_class(config=config['data_loading'])
        
        # Load data
        logger.info(f"Loading data using {loader_name} loader...")
        df = loader.load_data()
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Print basic information about loaded data
        logger.info("\nLoaded Data Info:")
        logger.info(df.info())
        logger.info("\nMissing Values Summary:")
        logger.info(df.isnull().sum())
        
        # Initialize and run data processor
        logger.info("\nInitializing data processor...")
        processor_name = config['data_processing'].get('data_processor_name')
        processor_class = get_processor(data_processor_name=processor_name)
        processor = processor_class(config=config['data_processing'])
        
        # Process data
        logger.info(f"Processing data using {processor_name} processor...")
        processed_df = processor.process_data(df)
        logger.info(f"Processed data shape: {processed_df.shape}")
        
        # Print basic information about processed data
        logger.info("\nProcessed Data Info:")
        logger.info(processed_df.info())
        logger.info("\nMissing Values After Processing:")
        logger.info(processed_df.isnull().sum())
        
        # Initialize and run feature creator
        logger.info("\nInitializing feature creator...")
        creator_name = config['feature_creation'].get('feature_creator_name')
        creator_class = get_feature_creator(feature_creator_name=creator_name)
        creator = creator_class(config=config['feature_creation'])
        
        # Create features
        logger.info(f"Creating features using {creator_name} creator...")
        featured_df = creator.create_features(processed_df)
        logger.info(f"Final data shape with features: {featured_df.shape}")
        
        # Print information about created features
        logger.info("\nFeature Creation Summary:")
        logger.info("Original columns:")
        logger.info(set(processed_df.columns))
        logger.info("\nNew features created:")
        new_features = set(featured_df.columns) - set(processed_df.columns)
        logger.info(new_features)
        
        # Print sample of final data
        logger.info("\nSample of final data with features:")
        pd.set_option('display.max_columns', None)
        logger.info(featured_df.head())
        
        # Print feature statistics
        logger.info("\nFeature Statistics:")
        logger.info(featured_df[list(new_features)].describe())
    
        
        return featured_df
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise

def main(
    config_path: Union[str, Path]
):
    """Main function to run the test script."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_data_pipeline(config_path)

if __name__ == "__main__":
    main(
        "artifacts/california_synthetic/config.yaml"
    )
