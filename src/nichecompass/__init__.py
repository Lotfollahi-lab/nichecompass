from importlib.metadata import version

from . import benchmarking, data, models, modules, nn, train, utils

__all__ = ["benchmarking", "data", "models", "modules", "nn", "train", "utils"]

__version__ = version("nichecompass")