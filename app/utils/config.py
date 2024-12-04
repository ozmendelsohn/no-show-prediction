import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path, optional
        Path to configuration file. If None, uses default path.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("artifacts/config/data_loading.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 