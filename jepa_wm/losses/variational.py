"""
Variational JEPA Loss (V-JEPA).

Formulation of probabilistic world models minimizing negative log-likelihood (NLL)
under Gaussian distribution modeling and regularized via KL-divergence.

Theoretical Foundation:
V-JEPA (2026): "Variational Joint Embedding Predictive Architectures as Probabilistic World Models"
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalJEPALoss(nn.Module):
    """
    Variational JEPA loss function.

    Models future latent state as a Gaussian distribution p_phi(z_{t+k} | mu_pred, logvar_pred)
    and evaluates negative log-likelihood against target latents, regularized by KL divergence.

    Attributes:
        beta (float): Weight for KL divergence prior penalty.
    """

    def __init__(self, beta: float = 1e-3) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        mu_pred: torch.Tensor,
        logvar_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            mu_pred (torch.Tensor): Predicted mean (B, K, d) or (B, d).
            logvar_pred (torch.Tensor): Predicted log-variance (B, K, d) or (B, d).
            z_target (torch.Tensor): Target latents (B, K, d) or (B, d).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Variational loss and metrics.
        """
        # Gaussian Negative Log-Likelihood: 0.5 * (logvar + (z - mu)^2 / var)
        var_pred = torch.exp(logvar_pred).clamp(min=1e-6, max=1e3)
        nll = 0.5 * (logvar_pred + (z_target - mu_pred).pow(2) / var_pred).mean()

        # KL divergence to standard isotropic Gaussian N(0, I)
        # KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.mean(1.0 + logvar_pred - mu_pred.pow(2) - var_pred)

        total_loss = nll + self.beta * kl_div

        metrics = {
            "loss_total": total_loss.item(),
            "loss_nll": nll.item(),
            "loss_kl": kl_div.item(),
            "beta": self.beta,
        }

        return total_loss, metrics
