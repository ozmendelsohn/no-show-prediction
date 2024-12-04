from app.core.data_processors.california_processor import CaliforniaProcessor
from app.core.data_processors.brazil_processor import BrazilDataProcessor
from app.core.data_processors.base import BaseDataProcessor

def get_processor(
    data_processor_name: str,
) -> BaseDataProcessor:
    """
    Get the appropriate data processor based on the name.

    Parameters
    ----------
    data_processor_name : str
        Name of the data processor to use

    Returns
    -------
    BaseDataProcessor
        Data processor instance

    Raises
    ------
    ValueError
        If the specified data processor is not supported
    """
    if data_processor_name.lower() == 'california':
        return CaliforniaProcessor
    elif data_processor_name.lower() == 'brazil':
        return BrazilDataProcessor
    else:
        raise ValueError(f"Data processor {data_processor_name} not supported") 