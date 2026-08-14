"""Unit tests for dynamical systems and dataset pipelines."""

import numpy as np
import pytest
import torch
from jepa_wm.data import (
    Lorenz63System,
    Lorenz96System,
    CoupledOscillatorsSystem,
    MultiSensorIndustrialBenchmark,
    TrajectoryDataset,
    create_dataloaders,
)


def test_lorenz63_simulation():
    """Verify Lorenz-63 RK4 integration, dimensions, and finite values."""
    system = Lorenz63System(dt=0.01)
    states, actions = system.generate_trajectories(num_trajectories=5, num_steps=100, seed=42)

    assert states.shape == (5, 100, 3)
    assert actions.shape == (5, 100, 3)
    assert not np.isnan(states).any()
    assert not np.isinf(states).any()


def test_lorenz96_simulation():
    """Verify N-dimensional Lorenz-96 ring simulation."""
    dim = 8
    system = Lorenz96System(dim=dim, forcing=8.0, dt=0.01)
    states, actions = system.generate_trajectories(num_trajectories=4, num_steps=50, seed=42)

    assert states.shape == (4, 50, dim)
    assert actions.shape == (4, 50, dim)
    assert not np.isnan(states).any()


def test_coupled_oscillators():
    """Verify Coupled Oscillators Hamiltonian mechanics simulation."""
    system = CoupledOscillatorsSystem(num_oscillators=3, dt=0.02)
    states, actions = system.generate_trajectories(num_trajectories=4, num_steps=60, seed=42)

    assert states.shape == (4, 60, 6)
    assert actions.shape == (4, 60, 3)
    assert not np.isnan(states).any()


def test_industrial_sensors_benchmark():
    """Verify multi-sensor telemetry generation with physical and nuisance noise channels."""
    benchmark = MultiSensorIndustrialBenchmark(
        system_type="lorenz",
        num_physical_channels=6,
        num_nuisance_channels=10,
    )
    obs, acts, states = benchmark.generate_dataset(num_trajectories=8, num_steps=80, seed=42)

    assert obs.shape == (8, 80, 16)
    assert acts.shape == (8, 80, 3)
    assert states.shape == (8, 80, 3)
    assert not np.isnan(obs).any()


def test_trajectory_dataset_and_dataloaders():
    """Verify TrajectoryDataset slicing and DataLoader batch collation."""
    B, T, C, d_a, d_s = 10, 100, 8, 3, 4
    obs = np.random.randn(B, T, C).astype(np.float32)
    acts = np.random.randn(B, T, d_a).astype(np.float32)
    states = np.random.randn(B, T, d_s).astype(np.float32)

    train_loader, val_loader = create_dataloaders(
        observations=obs,
        actions=acts,
        states=states,
        context_len=12,
        horizon=6,
        train_ratio=0.8,
        batch_size=16,
    )

    batch = next(iter(train_loader))
    assert "obs_ctx" in batch
    assert "obs_fut" in batch
    assert "act_fut" in batch
    assert "state_ctx" in batch
    assert "state_fut" in batch

    assert batch["obs_ctx"].shape == (16, 12, C)
    assert batch["obs_fut"].shape == (16, 6, C)
    assert batch["act_fut"].shape == (16, 6, d_a)
    assert batch["state_ctx"].shape == (16, d_s)
    assert batch["state_fut"].shape == (16, 6, d_s)
