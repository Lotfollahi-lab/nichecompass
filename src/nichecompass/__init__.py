from importlib.metadata import version

from . import data, models, modules, nn, train, utils

__all__ = ["data", "models", "modules", "nn", "train", "utils"]

__version__ = version("nichecompass")