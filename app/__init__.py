import yaml
from pathlib import Path
from typing import Dict, Any, Union

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing configuration parameters loaded from YAML file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
