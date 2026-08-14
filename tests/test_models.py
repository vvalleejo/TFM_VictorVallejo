"""Unit tests for neural network architectures."""

import pytest
import torch
from jepa_wm.core import (
    MultivariatePatchEncoder,
    TemporalTCNEncoder,
    ResidualGRUPredictor,
    TransformerLatentPredictor,
    LatentJEPAWorldModel,
    ReconstructiveWorldModel,
)


def test_multivariate_patch_encoder():
    """Verify MultivariatePatchEncoder forward pass and output dimensions."""
    B, T, C = 8, 15, 6
    latent_dim = 32
    encoder = MultivariatePatchEncoder(in_channels=C, latent_dim=latent_dim, d_model=64)

    x = torch.randn(B, T, C)
    z = encoder(x)

    assert z.shape == (B, latent_dim)
    assert not torch.isnan(z).any()


def test_temporal_tcn_encoder():
    """Verify TemporalTCNEncoder forward pass and output dimensions."""
    B, T, C = 8, 20, 5
    latent_dim = 16
    tcn = TemporalTCNEncoder(in_channels=C, latent_dim=latent_dim, hidden_channels=32)

    x = torch.randn(B, T, C)
    z = tcn(x)

    assert z.shape == (B, latent_dim)
    assert not torch.isnan(z).any()


def test_residual_gru_predictor():
    """Verify ResidualGRUPredictor rollout and gradients."""
    B, K, latent_dim, action_dim = 4, 7, 32, 3
    predictor = ResidualGRUPredictor(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=64)

    z_init = torch.randn(B, latent_dim, requires_grad=True)
    actions = torch.randn(B, K, action_dim)

    z_pred = predictor(z_init, actions)
    assert z_pred.shape == (B, K, latent_dim)

    loss = z_pred.sum()
    loss.backward()
    assert z_init.grad is not None


def test_transformer_latent_predictor():
    """Verify TransformerLatentPredictor rollout and gradients."""
    B, K, latent_dim, action_dim = 4, 5, 32, 2
    predictor = TransformerLatentPredictor(latent_dim=latent_dim, action_dim=action_dim, d_model=64)

    z_init = torch.randn(B, latent_dim, requires_grad=True)
    actions = torch.randn(B, K, action_dim)

    z_pred = predictor(z_init, actions)
    assert z_pred.shape == (B, K, latent_dim)

    loss = z_pred.sum()
    loss.backward()
    assert z_init.grad is not None


def test_latent_jepa_world_model_end_to_end():
    """Verify full end-to-end forward pass and loss computation for LatentJEPAWorldModel."""
    B, T_ctx, K, C, d_a = 6, 10, 5, 8, 3
    latent_dim = 32
    model = LatentJEPAWorldModel(
        in_channels=C,
        action_dim=d_a,
        latent_dim=latent_dim,
        encoder_d_model=64,
        sigreg_weight=0.1,
    )

    obs_ctx = torch.randn(B, T_ctx, C)
    act_fut = torch.randn(B, K, d_a)
    obs_fut = torch.randn(B, K, C)

    z_pred_seq, z_target_seq, loss, metrics = model(obs_ctx, act_fut, obs_fut)

    assert z_pred_seq.shape == (B, K, latent_dim)
    assert z_target_seq.shape == (B, K, latent_dim)
    assert "loss_total" in metrics
    assert "loss_pred" in metrics
    assert "loss_sigreg" in metrics

    loss = metrics["loss_total"]
    assert isinstance(loss, float)
    assert loss > 0.0


def test_reconstructive_world_model_baseline():
    """Verify ReconstructiveWorldModel baseline encoding, prediction, and decoding."""
    B, T_ctx, K, C, d_a = 4, 8, 4, 6, 2
    latent_dim = 24
    baseline = ReconstructiveWorldModel(
        in_channels=C,
        action_dim=d_a,
        latent_dim=latent_dim,
        hidden_dim=48,
    )

    obs_ctx = torch.randn(B, T_ctx, C)
    act_fut = torch.randn(B, K, d_a)
    obs_fut = torch.randn(B, K, C)

    z_pred_seq, obs_pred_seq, metrics = baseline(obs_ctx, act_fut, obs_fut)

    assert z_pred_seq.shape == (B, K, latent_dim)
    assert obs_pred_seq.shape == (B, K, C)
    assert "loss_recon" in metrics
