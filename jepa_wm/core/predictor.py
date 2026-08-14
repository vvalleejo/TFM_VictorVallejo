"""
Latent Dynamics Predictors for World Models.

Predicts future latent state trajectories given an initial latent state and a sequence of control actions:
    P_phi(z_t, a_{t : t + K - 1}) -> z_hat_{t + 1 : t + K}

Includes:
1. TransformerLatentPredictor: Causal Action-Conditioned Transformer Predictor.
2. ResidualGRUPredictor: Continuous-time inspired Residual GRU Transition Predictor.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wm.core.encoder import PositionalEncoding


class ResidualGRUPredictor(nn.Module):
    """
    Residual Recurrent Latent Predictor:
        z_{t+1} = z_t + MLP([z_t, a_t, h_t])

    Models smooth dynamical transitions via residual Euler-like integration steps.

    Attributes:
        latent_dim (int): Dimension of latent state d_z.
        action_dim (int): Dimension of control action d_a.
        hidden_dim (int): Dimension of GRU hidden state.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Action projection
        self.act_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
        )

        # Initial hidden state projector from z_t
        self.init_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.num_layers = num_layers

        # GRU cell
        self.gru = nn.GRU(
            input_size=latent_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Residual delta predictor
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_init: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rolls out latent predictions for K steps.

        Args:
            z_init (torch.Tensor): Initial latent state z_t of shape (B, latent_dim).
            action_seq (torch.Tensor): Future actions a_{t:t+K-1} of shape (B, K, action_dim).

        Returns:
            torch.Tensor: Predicted trajectory z_hat_{t+1:t+K} of shape (B, K, latent_dim).
        """
        B, K, d_a = action_seq.shape
        device = z_init.device

        # Embed actions
        act_emb = self.act_proj(action_seq)  # (B, K, hidden_dim)

        # Initialize GRU hidden state from z_init
        h0 = self.init_hidden(z_init)  # (B, num_layers * hidden_dim)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()  # (num_layers, B, hidden_dim)

        z_preds = []
        curr_z = z_init

        for k in range(K):
            # Input at step k: [curr_z, act_emb_k]
            a_k = act_emb[:, k]  # (B, hidden_dim)
            step_input = torch.cat([curr_z, a_k], dim=-1).unsqueeze(1)  # (B, 1, latent_dim + hidden_dim)

            out, h0 = self.gru(step_input, h0)  # out: (B, 1, hidden_dim)
            delta_z = self.delta_head(out.squeeze(1))  # (B, latent_dim)

            # Residual state update
            curr_z = curr_z + delta_z
            z_preds.append(curr_z)

        return torch.stack(z_preds, dim=1)  # (B, K, latent_dim)


class TransformerLatentPredictor(nn.Module):
    """
    Action-Conditioned Transformer Latent Dynamics Predictor.

    Uses a causal attention mask to predict multi-step future latent states.

    Attributes:
        latent_dim (int): Latent dimension d_z.
        action_dim (int): Action dimension d_a.
        d_model (int): Transformer model dimension.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 3,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = d_model

        self.z_proj = nn.Linear(latent_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(
        self,
        z_init: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_init (torch.Tensor): Shape (B, latent_dim).
            action_seq (torch.Tensor): Shape (B, K, action_dim).

        Returns:
            torch.Tensor: Shape (B, K, latent_dim).
        """
        B, K, _ = action_seq.shape

        # Initial latent token: (B, 1, d_model)
        z_tok = self.z_proj(z_init).unsqueeze(1)

        # Action tokens: (B, K, d_model)
        act_toks = self.act_proj(action_seq)

        # Combined sequence: [z_0, a_0, a_1, ..., a_{K-1}] -> length K + 1
        seq = torch.cat([z_tok, act_toks], dim=1)
        seq = self.pos_encoder(seq)

        # Causal attention mask of size (K + 1, K + 1)
        seq_len = K + 1
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(z_init.device)

        out = self.transformer(seq, mask=causal_mask, is_causal=True)

        # Predict future states from positions 1 to K (conditioned on past)
        pred_toks = out[:, 1:]  # (B, K, d_model)
        z_preds = self.head(pred_toks)  # (B, K, latent_dim)

        return z_preds
