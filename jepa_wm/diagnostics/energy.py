"""
Energy Landscapes, Prediction Errors, and Noise Rejection Ratio Diagnostics.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F


def compute_noise_rejection_ratio(
    latents: np.ndarray,
    phys_observations: np.ndarray,
    nuis_observations: np.ndarray,
) -> Dict[str, float]:
    """
    Measures the Noise Rejection Ratio (NRR):
    Compares the proportion of latent variance explained by true physical channels
    versus nuisance noise channels:
        NRR = R^2(latents ~ physical) / (R^2(latents ~ nuisance) + eps)

    A high NRR (> 10.0) demonstrates that the JEPA encoder successfully ignores
    uninformative stochastic sensor noise.

    Args:
        latents (np.ndarray): Learned latents of shape (N, d_z).
        phys_observations (np.ndarray): Informative sensor channels (N, C_phys).
        nuis_observations (np.ndarray): Nuisance noise channels (N, C_nuis).

    Returns:
        Dict[str, float]: NRR metrics and individual R2 values.
    """
    # 1. Regress latents from physical sensor channels
    probe_phys = Ridge(alpha=1e-3).fit(phys_observations, latents)
    r2_phys = float(r2_score(latents, probe_phys.predict(phys_observations), multioutput="uniform_average"))

    # 2. Regress latents from nuisance noise channels
    if nuis_observations.shape[1] > 0:
        probe_nuis = Ridge(alpha=1e-3).fit(nuis_observations, latents)
        r2_nuis = float(r2_score(latents, probe_nuis.predict(nuis_observations), multioutput="uniform_average"))
        # Clamp r2_nuis to non-negative
        r2_nuis = max(0.0, r2_nuis)
    else:
        r2_nuis = 0.0

    nrr = (max(0.0, r2_phys) + 1e-4) / (r2_nuis + 1e-4)

    return {
        "noise_rejection_ratio": float(nrr),
        "r2_physical_channels": float(r2_phys),
        "r2_nuisance_channels": float(r2_nuis),
    }


def compute_latent_energy_surface(
    model: torch.nn.Module,
    z_center: torch.Tensor,
    action: torch.Tensor,
    grid_range: float = 3.0,
    grid_points: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the energy landscape (prediction surprise ||P(z, a) - z_target||^2)
    over a 2D slice through the latent space around z_center.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (X_grid, Y_grid, Energy_surface).
    """
    model.eval()
    device = z_center.device
    d_z = z_center.size(-1)

    # Choose top 2 principal directions or coordinate axes (e.g. e_1, e_2)
    v1 = torch.zeros(d_z, device=device)
    v1[0] = 1.0
    v2 = torch.zeros(d_z, device=device)
    v2[1] = 1.0

    xs = np.linspace(-grid_range, grid_range, grid_points)
    ys = np.linspace(-grid_range, grid_range, grid_points)
    X, Y = np.meshgrid(xs, ys)

    energies = np.zeros((grid_points, grid_points))

    with torch.no_grad():
        for i in range(grid_points):
            for j in range(grid_points):
                perturb = xs[i] * v1 + ys[j] * v2
                z_pert = (z_center + perturb).unsqueeze(0)  # (1, d_z)
                act = action.unsqueeze(0)  # (1, 1, d_a) if 1-step

                z_pred = model.predict_rollout(z_pert, act)  # (1, 1, d_z)
                # Energy: squared displacement / mismatch
                energy = F.mse_loss(z_pred, z_pert.unsqueeze(1))
                energies[j, i] = energy.item()

    return X, Y, energies
