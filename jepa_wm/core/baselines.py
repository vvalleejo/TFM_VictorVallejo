"""
Reconstructive World Model Baselines (Autoencoder / VAE World Models).

Serves as the empirical comparative baseline against JEPA.
Reconstructs raw observation trajectories in sensor space:
    L_recon = MSE(x_hat_{t+1:t+K}, x_{t+1:t+K}) + beta * KL
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wm.core.encoder import MultivariatePatchEncoder
from jepa_wm.core.predictor import ResidualGRUPredictor


class ObservationDecoder(nn.Module):
    """Decodes latent representation z in R^(B x d_z) to sensor observation x in R^(B x C)."""

    def __init__(self, latent_dim: int, out_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Shape (B, latent_dim) or (B, K, latent_dim).

        Returns:
            torch.Tensor: Decoded observation shape (B, C) or (B, K, C).
        """
        return self.net(z)


class ReconstructiveWorldModel(nn.Module):
    """
    Reconstructive Generative World Model (PlaNet/Dreamer-style baseline).

    Trained end-to-end via MSE sensor reconstruction:
        L_total = MSE(x_hat_{t+1:t+K}, x_{t+1:t+K}) + beta * KL(z || N(0, I))

    Attributes:
        encoder (MultivariatePatchEncoder): Observation encoder.
        predictor (ResidualGRUPredictor): Latent dynamics predictor.
        decoder (ObservationDecoder): Observation decoder.
        kl_weight (float): Weight beta for Gaussian prior penalty.
    """

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = MultivariatePatchEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            d_model=hidden_dim,
        )
        self.predictor = ResidualGRUPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = ObservationDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            hidden_dim=hidden_dim,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes observation window to latent state."""
        return self.encoder(obs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent state to observation."""
        return self.decoder(z)

    def forward(
        self,
        obs_ctx: torch.Tensor,
        act_fut: torch.Tensor,
        obs_fut: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
        """
        Args:
            obs_ctx (torch.Tensor): History observations (B, T_ctx, C).
            act_fut (torch.Tensor): Future actions (B, K, d_a).
            obs_fut (torch.Tensor, optional): Target observations (B, K, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
                - z_pred_seq: Predicted latents (B, K, latent_dim).
                - obs_pred_seq: Decoded predicted observations (B, K, C).
                - metrics: Loss dictionary if obs_fut is provided.
        """
        # 1. Encode context to z_t
        z_ctx = self.encoder(obs_ctx)

        # 2. Predict future latent trajectory
        z_pred_seq = self.predictor(z_ctx, act_fut)  # (B, K, latent_dim)

        # 3. Decode latents to raw observations
        obs_pred_seq = self.decoder(z_pred_seq)  # (B, K, C)

        if obs_fut is None:
            return z_pred_seq, obs_pred_seq, None

        # 4. Reconstruction loss in sensor space
        recon_loss = F.mse_loss(obs_pred_seq, obs_fut)

        # Optional gentle latent regularizer
        latent_norm_loss = z_pred_seq.pow(2).mean()

        total_loss = recon_loss + self.kl_weight * latent_norm_loss

        metrics = {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_latent_reg": latent_norm_loss.item(),
        }

        return z_pred_seq, obs_pred_seq, metrics
