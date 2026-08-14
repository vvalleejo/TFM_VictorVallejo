"""
Multi-Step Latent Rollout Loss and Complete LeWorldModel (LeWM) Loss.

Theoretical Foundation:
Balestriero & LeCun (2025): "LeWorldModel: Stable End-to-End JEPA from Pixels"
Terver et al. (2026): "A Lightweight Library for Energy-Based JEPAs"
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wm.losses.sigreg import SIGReg


class MultiStepPredictiveLoss(nn.Module):
    """
    Computes discounted multi-step predictive loss in the latent embedding space:

    L_pred = (1 / K) * sum_{k=1}^K gamma^(k-1) * || z_hat_{t+k} - z_{t+k} ||_2^2

    Attributes:
        discount (float): Temporal discount factor gamma in (0, 1] (default: 0.95).
        loss_type (str): 'mse' or 'smooth_l1' (default: 'mse').
    """

    def __init__(self, discount: float = 0.95, loss_type: str = "mse") -> None:
        super().__init__()
        self.discount = discount
        self.loss_type = loss_type

    def forward(
        self,
        z_pred_seq: torch.Tensor,
        z_target_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_pred_seq (torch.Tensor): Predicted rollout trajectory of shape (B, K, d).
            z_target_seq (torch.Tensor): Ground-truth target trajectory of shape (B, K, d).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Weighted scalar rollout loss and step-by-step metrics.
        """
        B, K, d = z_pred_seq.shape
        device = z_pred_seq.device

        # Compute per-step loss
        if self.loss_type == "mse":
            step_losses = F.mse_loss(z_pred_seq, z_target_seq, reduction="none").mean(dim=(0, 2))  # (K,)
        elif self.loss_type == "smooth_l1":
            step_losses = F.smooth_l1_loss(z_pred_seq, z_target_seq, reduction="none").mean(dim=(0, 2))  # (K,)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Temporal discount weights
        gammas = torch.tensor(
            [self.discount ** k for k in range(K)],
            device=device,
            dtype=z_pred_seq.dtype,
        )
        total_loss = (step_losses * gammas).sum() / gammas.sum()

        metrics = {
            "loss_pred": total_loss.item(),
            "loss_pred_step1": step_losses[0].item() if K > 0 else 0.0,
            "loss_pred_stepK": step_losses[-1].item() if K > 0 else 0.0,
        }

        return total_loss, metrics


class LeWMLoss(nn.Module):
    """
    Complete LeWorldModel (LeWM) Loss:
    L_total = L_pred + lambda * SIGReg(Z)

    Attributes:
        sigreg_weight (float): Regularization coefficient lambda (default: 0.1).
        num_slices (int): Number of random 1D projections M for SIGReg (default: 64).
        discount (float): Rollout discount factor gamma.
    """

    def __init__(
        self,
        sigreg_weight: float = 0.1,
        num_slices: int = 64,
        discount: float = 0.95,
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.pred_loss = MultiStepPredictiveLoss(discount=discount, loss_type=loss_type)
        self.sigreg = SIGReg(num_slices=num_slices)

    def forward(
        self,
        z_pred_seq: torch.Tensor,
        z_target_seq: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_pred_seq (torch.Tensor): Predicted latents (B, K, d).
            z_target_seq (torch.Tensor): Encoded target latents (B, K, d).
            z_context (torch.Tensor, optional): Encoded context latents (B, d).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Total scalar loss and component breakdown.
        """
        # 1. Multi-step prediction loss
        loss_pred, metrics = self.pred_loss(z_pred_seq, z_target_seq)

        # 2. SIGReg on all encoded target embeddings (and context if provided)
        if z_context is not None:
            if z_context.ndim == 2:
                z_all = torch.cat([z_context.unsqueeze(1), z_target_seq], dim=1)
            else:
                z_all = torch.cat([z_context, z_target_seq], dim=1)
        else:
            z_all = z_target_seq

        loss_sigreg = self.sigreg(z_all)

        # 3. Combined total loss
        total_loss = loss_pred + self.sigreg_weight * loss_sigreg

        metrics["loss_total"] = total_loss.item()
        metrics["loss_sigreg"] = loss_sigreg.item()
        metrics["sigreg_weight"] = self.sigreg_weight

        return total_loss, metrics
