"""Unit tests for latent space planning and MPC algorithms."""

import pytest
import torch
from jepa_wm.core import LatentJEPAWorldModel
from jepa_wm.planning import LatentCEMPlanner, LatentMPPIPlanner


def test_latent_cem_planner():
    """Verify LatentCEMPlanner trajectory optimization towards goal latent."""
    torch.manual_seed(42)
    in_channels, action_dim, latent_dim = 6, 2, 16
    model = LatentJEPAWorldModel(
        in_channels=in_channels,
        action_dim=action_dim,
        latent_dim=latent_dim,
        encoder_d_model=32,
    )

    planner = LatentCEMPlanner(
        model=model,
        horizon=6,
        num_candidates=64,
        num_elites=8,
        num_iterations=4,
    )

    z_init = torch.randn(1, latent_dim)
    z_goal = torch.randn(1, latent_dim)

    best_actions, info = planner.plan(z_init, z_goal, action_dim=action_dim)

    assert best_actions.shape == (6, action_dim)
    assert "best_cost" in info
    assert info["best_cost"] <= info["initial_cost"]


def test_latent_mppi_planner():
    """Verify LatentMPPIPlanner Boltzmann weighted trajectory optimization."""
    torch.manual_seed(42)
    in_channels, action_dim, latent_dim = 4, 3, 16
    model = LatentJEPAWorldModel(
        in_channels=in_channels,
        action_dim=action_dim,
        latent_dim=latent_dim,
        encoder_d_model=32,
    )

    planner = LatentMPPIPlanner(
        model=model,
        horizon=5,
        num_samples=64,
        temperature=0.5,
    )

    z_init = torch.randn(1, latent_dim)
    z_goal = torch.randn(1, latent_dim)

    opt_actions, info = planner.plan(z_init, z_goal, action_dim=action_dim)

    assert opt_actions.shape == (5, action_dim)
    assert "min_cost" in info
    assert "mean_cost" in info
