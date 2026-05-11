__version__ = "1.0.0"

from .registry import (
    register_model,
    register_dataset,
    register_dataset_builder,
    register_evaluation_strategy,
    get_model,
    get_dataset,
    get_dataset_builder,
    get_evaluation_strategy,
    list_models,
    list_datasets,
    list_dataset_builders,
    list_evaluation_strategies,
)


from .utils.logger import setup_logger
from .utils.metrics import AverageMeter, set_random_seed

__all__ = [
    '__version__',
    'register_model', 'register_dataset', 'register_dataset_builder', 'register_evaluation_strategy',
    'get_model', 'get_dataset', 'get_dataset_builder', 'get_evaluation_strategy',
    'list_models', 'list_datasets', 'list_dataset_builders', 'list_evaluation_strategies',
    'setup_logger', 'AverageMeter', 'set_random_seed'
]
