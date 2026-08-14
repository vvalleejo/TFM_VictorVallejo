"""Multivariate dynamical systems and benchmark data pipelines."""

from jepa_wm.data.dynamical_systems import (
    Lorenz63System,
    Lorenz96System,
    CoupledOscillatorsSystem,
)
from jepa_wm.data.industrial_sensors import MultiSensorIndustrialBenchmark
from jepa_wm.data.dataset import TrajectoryDataset, create_dataloaders

__all__ = [
    "Lorenz63System",
    "Lorenz96System",
    "CoupledOscillatorsSystem",
    "MultiSensorIndustrialBenchmark",
    "TrajectoryDataset",
    "create_dataloaders",
]
