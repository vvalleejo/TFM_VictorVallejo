"""
Latent Model Predictive Path Integral (MPPI) Planner.

Theoretical Foundation:
Williams et al. (2017): "Information Theoretic MPC for Model-Based Reinforcement Learning"
Terver et al. (2026): "A Lightweight Library for Energy-Based JEPAs"
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMPPIPlanner:
    """
    Model Predictive Path Integral (MPPI) trajectory optimization in latent space:
    Computes optimal control actions as a Boltzmann-weighted average over sampled rollouts.

    Attributes:
        model (nn.Module): World model predictor.
        horizon (int): Planning horizon H.
        num_samples (int): Number of trajectory samples N.
        temperature (float): Boltzmann temperature lambda_temp.
        noise_sigma (float): Standard deviation of exploratory Gaussian noise.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int = 10,
        num_samples: int = 256,
        temperature: float = 0.5,
        noise_sigma: float = 0.3,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        action_dim: int,
        prev_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_init (torch.Tensor): Current latent state (1, d_z).
            z_goal (torch.Tensor): Goal latent state (1, d_z).
            action_dim (int): Action dimension.
            prev_actions (torch.Tensor, optional): Nominal action trajectory (H, d_a).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: (Optimized actions (H, d_a), metrics dict).
        """
        device = z_init.device
        d_z = z_init.size(-1)
        z_init = z_init.view(1, d_z)
        z_goal = z_goal.view(1, d_z)

        if prev_actions is not None:
            nominal = torch.cat([prev_actions[1:], torch.zeros(1, action_dim, device=device)], dim=0)
        else:
            nominal = torch.zeros(self.horizon, action_dim, device=device)

        # 1. Sample noise perturbations: (num_samples, H, d_a)
        noise = torch.randn(self.num_samples, self.horizon, action_dim, device=device) * self.noise_sigma
        actions = (nominal.unsqueeze(0) + noise).clamp(self.action_low, self.action_high)

        # Set first trajectory as nominal without noise
        actions[0] = nominal

        # 2. Parallel Rollout
        z_init_batch = z_init.expand(self.num_samples, -1)
        z_goal_batch = z_goal.expand(self.num_samples, -1)

        if hasattr(self.model, "predict_rollout"):
            z_rollouts = self.model.predict_rollout(z_init_batch, actions)  # (num_samples, H, d_z)
        else:
            z_rollouts = self.model(z_init_batch, actions)

        # 3. Compute costs S(tau)
        z_final = z_rollouts[:, -1, :]
        terminal_cost = F.mse_loss(z_final, z_goal_batch, reduction="none").sum(dim=-1)
        path_cost = F.mse_loss(z_rollouts, z_goal_batch.unsqueeze(1).expand(-1, self.horizon, -1), reduction="none").sum(dim=(1, 2))
        total_costs = terminal_cost + 0.1 * path_cost  # (num_samples,)

        # 4. Boltzmann softmax weights: w_i = exp( - (S_i - min(S)) / temperature )
        min_cost = total_costs.min()
        weights = torch.exp(-(total_costs - min_cost) / self.temperature)
        weights = weights / (weights.sum() + 1e-12)  # (num_samples,)

        # 5. Weighted combination of candidate actions
        # weights: (num_samples, 1, 1), actions: (num_samples, H, d_a)
        opt_actions = (weights.view(-1, 1, 1) * actions).sum(dim=0)  # (H, d_a)

        return opt_actions, {
            "min_cost": float(min_cost.item()),
            "mean_cost": float(total_costs.mean().item()),
        }
