"""Mutual Fund Analyzer — source package."""
from . import data_loader, feature_engineering, model, optimizer, simulation, recommendation  # noqa

__all__ = [
    "data_loader", "feature_engineering", "model",
    "optimizer", "simulation", "recommendation",
]
__version__ = "1.0.0"
