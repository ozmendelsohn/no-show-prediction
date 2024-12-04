from app.core.data_loaders.california_loader import CaliforniaLoader
from app.core.data_loaders.brazil_loader import BrazilDataLoader
from app.core.data_loaders.base import BaseDataLoader

def get_loader(
    data_loader_name: str,
    ) -> BaseDataLoader:
    if data_loader_name.lower() == 'california':
        return CaliforniaLoader
    elif data_loader_name.lower() == 'brazil':
        return BrazilDataLoader
    else:
        raise ValueError(f"Data source {data_loader_name} not supported")