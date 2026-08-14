"""
Sketched-Isotropic-Gaussian Regularizer (SIGReg).

Implements the anti-collapse regularization via the Cramér-Wold device and
vectorized closed-form 1D Epps-Pulley empirical characteristic function testing.

Theoretical Foundation:
Klindt, LeCun, Balestriero (2025/2026): "When Does LeJEPA Learn a World Model?"
Balestriero & LeCun (2025): "LeWorldModel: Stable End-to-End JEPA from Pixels"
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer (SIGReg).

    Projects latent representation vectors Z in R^(N x d) onto M random 1D directions
    sampled uniformly from the unit hypersphere S^(d-1), and evaluates departure from
    standard Gaussianity N(0, 1) using the closed-form Epps-Pulley test statistic.

    Attributes:
        num_slices (int): Number of random 1D projections M (default: 64).
        bandwidth (float): Bandwidth parameter sigma_w for Gaussian kernel weight (default: 1.0).
        normalize (bool): Whether to standardize empirical latents before testing (default: False).
    """

    def __init__(
        self,
        num_slices: int = 64,
        bandwidth: float = 1.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.bandwidth = bandwidth
        self.normalize = normalize

    def forward(
        self,
        z: torch.Tensor,
        num_slices: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes SIGReg loss on latents z.

        Args:
            z (torch.Tensor): Latent tensor of shape (N, d) or (B, T, d).
            num_slices (int, optional): Override number of projection directions.

        Returns:
            torch.Tensor: Scalar SIGReg penalty >= 0.
        """
        # Flatten batch and time dimensions into a single sample dimension N
        if z.ndim > 2:
            z = z.reshape(-1, z.size(-1))

        N, d = z.shape
        if N < 2:
            # Need at least 2 samples to compute pairwise distribution statistics
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        M = num_slices or self.num_slices

        # 1. Sample M directions uniformly from unit hypersphere S^(d-1)
        # Using standard Gaussian vectors normalized to unit norm
        u = torch.randn(d, M, device=z.device, dtype=z.dtype)
        u = F.normalize(u, p=2, dim=0)  # Shape (d, M)

        # 2. Project latents onto 1D slices: H in R^(N x M)
        h = torch.matmul(z, u)  # Shape (N, M)

        if self.normalize:
            # Optional standardization per slice
            mean = h.mean(dim=0, keepdim=True)
            std = h.std(dim=0, keepdim=True).clamp(min=1e-6)
            h = (h - mean) / std

        # 3. Evaluate vectorized closed-form Epps-Pulley statistic on each slice
        loss = self._epps_pulley_statistic(h, self.bandwidth)
        return loss

    @staticmethod
    def _epps_pulley_statistic(h: torch.Tensor, sigma_w: float = 1.0) -> torch.Tensor:
        """
        Computes the exact closed-form 1D Epps-Pulley test statistic across M slices.

        Args:
            h (torch.Tensor): Projected latents of shape (N, M).
            sigma_w (float): Gaussian weight function bandwidth.

        Returns:
            torch.Tensor: Mean Epps-Pulley loss averaged over all M slices.
        """
        N, M = h.shape
        s2 = sigma_w ** 2

        # Term 1: Double sum over empirical samples |phi_N(t)|^2
        # Pairwise differences (h_j - h_l) per slice
        # h: (N, 1, M), h.transpose: (1, N, M) -> diffs: (N, N, M)
        diffs = h.unsqueeze(1) - h.unsqueeze(0)  # (N, N, M)
        diff_sq = diffs.pow(2)

        gamma1 = s2 / 2.0
        c1 = 1.0  # Normalized Gaussian integral
        term1 = torch.exp(-gamma1 * diff_sq).mean(dim=(0, 1))  # (M,)

        # Term 2: Cross term 2 * Re(phi_N(t) * phi_0(t))
        # Exponent: - (s2 / (2 * (s2 + 1))) * h_j^2
        denom2 = s2 + 1.0
        c2 = 1.0 / math.sqrt(denom2)
        gamma2 = s2 / (2.0 * denom2)
        term2 = 2.0 * c2 * torch.exp(-gamma2 * (h ** 2)).mean(dim=0)  # (M,)

        # Term 3: Target Gaussian constant |phi_0(t)|^2
        denom3 = 2.0 * s2 + 1.0
        term3 = 1.0 / math.sqrt(denom3)

        # Total Epps-Pulley statistic per slice
        t_stat = term1 - term2 + term3  # (M,)

        # Clamp at zero for numerical stability (analytical integral is non-negative)
        t_stat = F.relu(t_stat)

        # Average over all M slices
        return t_stat.mean()
