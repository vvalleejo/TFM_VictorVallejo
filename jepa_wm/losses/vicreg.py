"""
Variance-Invariance-Covariance Regularization (VICReg) Loss Baseline.

Reference:
Bardes, Ponce, LeCun (2022): "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning", ICLR 2022.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg loss function computing:
    1. Invariance (MSE between representations)
    2. Variance (Hinge loss forcing std >= gamma)
    3. Covariance (Penalizing off-diagonal covariance entries to decorrelate features)

    Attributes:
        sim_coeff (float): Weight for invariance term.
        std_coeff (float): Weight for variance term.
        cov_coeff (float): Weight for covariance term.
        gamma (float): Target standard deviation threshold (default: 1.0).
        epsilon (float): Numerical stability epsilon.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the complete VICReg loss.

        Args:
            z_pred (torch.Tensor): Predicted latents of shape (N, d).
            z_target (torch.Tensor): Target latents of shape (N, d).

        Returns:
            Tuple[torch.Tensor, dict]: Total scalar loss and dictionary of component losses.
        """
        if z_pred.ndim > 2:
            z_pred = z_pred.reshape(-1, z_pred.size(-1))
        if z_target.ndim > 2:
            z_target = z_target.reshape(-1, z_target.size(-1))

        N, d = z_pred.shape

        # 1. Invariance term (Mean Squared Error)
        sim_loss = F.mse_loss(z_pred, z_target)

        # 2. Variance term
        std_pred = torch.sqrt(z_pred.var(dim=0) + self.epsilon)
        std_target = torch.sqrt(z_target.var(dim=0) + self.epsilon)
        std_loss = (
            torch.mean(F.relu(self.gamma - std_pred))
            + torch.mean(F.relu(self.gamma - std_target))
        ) / 2.0

        # 3. Covariance term
        # Center latents
        z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
        z_target_centered = z_target - z_target.mean(dim=0, keepdim=True)

        # Covariance matrices: (d, d)
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (N - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (N - 1)

        cov_loss = (
            self._off_diagonal_penalty(cov_pred)
            + self._off_diagonal_penalty(cov_target)
        ) / 2.0

        # Weighted combination
        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        loss_dict = {
            "loss": total_loss.item(),
            "sim_loss": sim_loss.item(),
            "std_loss": std_loss.item(),
            "cov_loss": cov_loss.item(),
        }

        return total_loss, loss_dict

    @staticmethod
    def _off_diagonal_penalty(c: torch.Tensor) -> torch.Tensor:
        """Computes off-diagonal sum of squares divided by d."""
        d = c.size(0)
        diag = torch.eye(d, device=c.device, dtype=torch.bool)
        off_diag = c[~diag]
        return off_diag.pow(2).sum() / d
