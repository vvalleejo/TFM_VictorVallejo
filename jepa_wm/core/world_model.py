"""
Latent JEPA World Model (LeWM).

Combines Multivariate Context Encoder, Target Encoder (Siamese or EMA),
and Latent Predictor with SIGReg regularizer for stable, end-to-end self-supervised
latent world modeling from multivariate observations without raw reconstruction.

Theoretical Foundations:
- Yann LeCun (2022/2023): "A Path Towards Autonomous Machine Intelligence"
- Balestriero & LeCun (2025): "LeWorldModel: Stable End-to-End JEPA from Pixels"
- Klindt, LeCun, Balestriero (2026): "When Does LeJEPA Learn a World Model?"
"""

from typing import Dict, Optional, Tuple
import copy
import torch
import torch.nn as nn

from jepa_wm.core.encoder import MultivariatePatchEncoder
from jepa_wm.core.predictor import ResidualGRUPredictor, TransformerLatentPredictor
from jepa_wm.losses.rollout_loss import LeWMLoss


class LatentJEPAWorldModel(nn.Module):
    """
    Latent Joint-Embedding Predictive Architecture World Model.

    Attributes:
        encoder (nn.Module): Context Encoder E_theta.
        target_encoder (nn.Module): Target Encoder E_theta_bar.
        predictor (nn.Module): Latent Dynamics Predictor P_phi.
        loss_fn (LeWMLoss): LeWM multi-step predictive loss with SIGReg.
        use_ema (bool): Whether to update target encoder via Exponential Moving Average.
        ema_decay (float): Momentum parameter for EMA updates.
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        latent_dim: int = 64,
        encoder_d_model: int = 128,
        predictor_type: str = "gru",
        sigreg_weight: float = 0.1,
        discount: float = 0.95,
        use_ema: bool = False,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 1. Context Encoder
        self.encoder = MultivariatePatchEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            d_model=encoder_d_model,
        )

        # 2. Target Encoder (either shared Siamese or EMA copy)
        if use_ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            self.target_encoder = self.encoder

        # 3. Latent Dynamics Predictor
        if predictor_type == "gru":
            self.predictor = ResidualGRUPredictor(
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=encoder_d_model,
            )
        elif predictor_type == "transformer":
            self.predictor = TransformerLatentPredictor(
                latent_dim=latent_dim,
                action_dim=action_dim,
                d_model=encoder_d_model,
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        # 4. LeWM Loss
        self.loss_fn = LeWMLoss(sigreg_weight=sigreg_weight, discount=discount)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes an observation sequence into a latent embedding vector z."""
        return self.encoder(obs)

    def predict_rollout(self, z_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Rolls out predicted future latents given initial latent and action sequence."""
        return self.predictor(z_init, actions)

    def forward(
        self,
        obs_ctx: torch.Tensor,
        act_fut: torch.Tensor,
        obs_fut: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """
        Forward pass of JEPA World Model.

        Args:
            obs_ctx (torch.Tensor): Context history observations of shape (B, T_ctx, C).
            act_fut (torch.Tensor): Future actions sequence of shape (B, K, d_a).
            obs_fut (torch.Tensor, optional): Target observations of shape (B, K, C).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, float]]]:
                - z_pred_seq: Predicted future latents (B, K, d_z).
                - z_target_seq: Target encoded latents (B, K, d_z) if obs_fut provided.
                - metrics: Loss metrics dict if obs_fut provided.
        """
        # 1. Encode context window to initial latent state z_t
        z_ctx = self.encoder(obs_ctx)  # (B, latent_dim)

        # 2. Predict future latent trajectory z_hat_{t+1:t+K}
        z_pred_seq = self.predictor(z_ctx, act_fut)  # (B, K, latent_dim)

        if obs_fut is None:
            return z_pred_seq, None, None, None

        # 3. Fast parallel encoding of future targets: (B, K, C) -> (B, K, latent_dim)
        with torch.set_grad_enabled(not self.use_ema):
            z_target_seq = self.target_encoder(obs_fut, return_sequence=True)  # (B, K, latent_dim)

        # 4. Compute complete LeWM loss (multi-step predictive loss + SIGReg)
        loss, metrics = self.loss_fn(z_pred_seq, z_target_seq, z_ctx)

        return z_pred_seq, z_target_seq, loss, metrics

    @torch.no_grad()
    def update_ema_target(self) -> None:
        """Updates EMA target encoder parameters."""
        if not self.use_ema:
            return
        for online_p, target_p in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1.0 - self.ema_decay)
