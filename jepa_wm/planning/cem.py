"""
Latent Cross-Entropy Method (CEM) Model Predictive Control (MPC) Planner.

Optimizes action trajectories directly in the learned latent world model space
without observation decoding or pixel reconstruction.

Theoretical Foundation:
Balestriero & LeCun (2025): "LeWorldModel: Stable End-to-End JEPA from Pixels"
Klindt et al. (2026): Theorem 4 (Optimal Latent Planning Equivalence).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCEMPlanner:
    """
    Cross-Entropy Method (CEM) trajectory optimization in latent space:
        min_{a_{0:H-1}} || z_H - z_goal ||_2^2 + action_penalty * sum_t ||a_t||_2^2

    Attributes:
        model (nn.Module): Pre-trained LatentJEPAWorldModel or Predictor.
        horizon (int): Planning horizon H.
        num_candidates (int): Number of candidate action trajectories sampled per iteration (N_pop).
        num_elites (int): Number of top-performing elite trajectories kept (N_elite).
        num_iterations (int): Optimization iterations per planning step.
        action_low (float): Lower bound for control actions.
        action_high (float): Upper bound for control actions.
        alpha (float): Momentum smoothing factor for distribution parameter updates.
        action_penalty (float): L2 regularization on control effort.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int = 10,
        num_candidates: int = 256,
        num_elites: int = 32,
        num_iterations: int = 5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        alpha: float = 0.1,
        action_penalty: float = 0.01,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.alpha = alpha
        self.action_penalty = action_penalty

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        action_dim: int,
        prev_solution: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Optimizes action sequence a_{0:H-1} starting from current latent state z_init to reach z_goal.

        Args:
            z_init (torch.Tensor): Current latent state of shape (1, d_z) or (d_z,).
            z_goal (torch.Tensor): Target goal latent of shape (1, d_z) or (d_z,).
            action_dim (int): Action dimension d_a.
            prev_solution (torch.Tensor, optional): Warm-start action sequence (H, d_a).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - best_actions: Optimized action trajectory of shape (H, d_a).
                - info: Optimization metrics (best_cost, initial_cost, final_cost).
        """
        device = z_init.device
        d_z = z_init.size(-1)
        z_init = z_init.view(1, d_z)
        z_goal = z_goal.view(1, d_z)

        # Initialize Gaussian action distribution parameters: mean (H, d_a), std (H, d_a)
        if prev_solution is not None:
            # Shift previous solution by 1 step (warm-start)
            mean = torch.cat([prev_solution[1:], torch.zeros(1, action_dim, device=device)], dim=0)
        else:
            mean = torch.zeros(self.horizon, action_dim, device=device)

        std = 0.5 * torch.ones(self.horizon, action_dim, device=device)

        # Expand z_init for batch rollouts: (num_candidates, d_z)
        z_init_batch = z_init.expand(self.num_candidates, -1)
        z_goal_batch = z_goal.expand(self.num_candidates, -1)

        best_actions = None
        best_cost = float("inf")
        initial_cost = None

        for it in range(self.num_iterations):
            # 1. Sample candidate trajectories from N(mean, std^2): (num_candidates, H, d_a)
            eps = torch.randn(self.num_candidates, self.horizon, action_dim, device=device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * eps).clamp(self.action_low, self.action_high)

            # 2. Rollout through Latent Predictor
            if hasattr(self.model, "predict_rollout"):
                z_rollouts = self.model.predict_rollout(z_init_batch, actions)  # (N_pop, H, d_z)
            else:
                z_rollouts = self.model(z_init_batch, actions)

            # 3. Compute trajectory cost: Terminal distance + Path cost + Action penalty
            z_final = z_rollouts[:, -1, :]  # (N_pop, d_z)
            terminal_cost = F.mse_loss(z_final, z_goal_batch, reduction="none").sum(dim=-1)

            # Distance at all intermediate steps
            path_cost = F.mse_loss(z_rollouts, z_goal_batch.unsqueeze(1).expand(-1, self.horizon, -1), reduction="none").sum(dim=(1, 2))

            action_cost = self.action_penalty * (actions ** 2).sum(dim=(1, 2))
            total_cost = terminal_cost + 0.1 * path_cost + action_cost  # (N_pop,)

            # 4. Select top N_elite trajectories
            costs_sorted, elite_indices = torch.topk(total_cost, self.num_elites, largest=False)
            elites = actions[elite_indices]  # (N_elite, H, d_a)

            if it == 0:
                initial_cost = float(costs_sorted[0].item())

            if costs_sorted[0].item() < best_cost:
                best_cost = float(costs_sorted[0].item())
                best_actions = elites[0].clone()

            # 5. Update Gaussian parameters with momentum smoothing
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0).clamp(min=1e-3)

            mean = (1.0 - self.alpha) * elite_mean + self.alpha * mean
            std = (1.0 - self.alpha) * elite_std + self.alpha * std

        return best_actions, {
            "best_cost": best_cost,
            "initial_cost": initial_cost,
        }
