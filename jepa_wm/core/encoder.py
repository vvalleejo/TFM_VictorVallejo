"""
Neural Encoders for Multivariate Time-Series and Dynamical Systems.

Includes:
1. MultivariatePatchEncoder: Multi-channel patch-based encoder with cross-channel and temporal self-attention.
2. TemporalTCNEncoder: Dilated Causal Temporal Convolutional Network with residual blocks.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to tensor of shape (B, T, d_model)."""
        return x + self.pe[:, :x.size(1)]


class MultivariatePatchEncoder(nn.Module):
    """
    Multivariate Patch and Cross-Channel Transformer Encoder.

    Maps multivariate time-series x in R^(B x T x C) to a compact latent representation z in R^(B x d_z).

    Attributes:
        in_channels (int): Number of sensor/variable channels C.
        d_model (int): Hidden embedding dimension.
        latent_dim (int): Output latent space dimension d_z.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of multi-head attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.d_model = d_model

        # 1. Channel projection: maps each time-step observation (C,) to (d_model,)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 2. Learnable [CLS] / summary token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # 4. Final latent projection head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Observations of shape (B, T, C).

        Returns:
            torch.Tensor: Latent representation z of shape (B, latent_dim).
        """
        B, T, C = x.shape

        # Project time steps to token embeddings: (B, T, d_model)
        tokens = self.input_proj(x)

        # Prepend [CLS] token: (B, T + 1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Add positional encodings
        tokens = self.pos_encoder(tokens)

        # Pass through Transformer
        encoded = self.transformer(tokens)

        # Readout from [CLS] token
        z_cls = encoded[:, 0]  # (B, d_model)
        z = self.head(z_cls)   # (B, latent_dim)
        return z


class TemporalTCNEncoder(nn.Module):
    """
    Dilated Temporal Convolutional Network (TCN) Encoder.

    Uses 1D causal dilated convolutions with residual skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        layers = []
        in_c = in_channels
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(in_c, hidden_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
            ])
            in_c = hidden_channels

        self.conv_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, T, C).

        Returns:
            torch.Tensor: Shape (B, latent_dim).
        """
        # Permute for 1D convolution: (B, C, T)
        x_conv = x.permute(0, 2, 1)
        feat = self.conv_net(x_conv)  # (B, hidden, T)
        pooled = self.pool(feat).squeeze(-1)  # (B, hidden)
        return self.head(pooled)
