from app.core.feature_creators.california_creator import CaliforniaFeatureCreator
from app.core.feature_creators.brazil_creator import BrazilFeatureCreator
from app.core.feature_creators.base import BaseFeatureCreator

def get_feature_creator(
    feature_creator_name: str,
) -> BaseFeatureCreator:
    """
    Get the appropriate feature creator based on the name.

    Parameters
    ----------
    feature_creator_name : str
        Name of the feature creator to use

    Returns
    -------
    BaseFeatureCreator
        Feature creator instance

    Raises
    ------
    ValueError
        If the specified feature creator is not supported
    """
    if feature_creator_name.lower() == 'california':
        return CaliforniaFeatureCreator
    elif feature_creator_name.lower() == 'brazil':
        return BrazilFeatureCreator
    else:
        raise ValueError(f"Feature creator {feature_creator_name} not supported") 