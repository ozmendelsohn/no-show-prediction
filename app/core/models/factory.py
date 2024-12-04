from app.core.models.random_forest import RandomForestModel
from app.core.models.base import BaseModel

def get_model(
    model_name: str,
) -> BaseModel:
    """
    Get the appropriate model based on the name.

    Parameters
    ----------
    model_name : str
        Name of the model to use

    Returns
    -------
    BaseModel
        Model instance

    Raises
    ------
    ValueError
        If the specified model is not supported
    """
    if model_name.lower() == 'random_forest':
        return RandomForestModel
    else:
        raise ValueError(f"Model {model_name} not supported") 