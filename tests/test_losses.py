"""Unit tests for JEPA loss functions and anti-collapse regularizers."""

import pytest
import torch
from jepa_wm.losses import (
    SIGReg,
    VICRegLoss,
    InfoNCELoss,
    MultiStepPredictiveLoss,
    LeWMLoss,
    VariationalJEPALoss,
)


def test_sigreg_gaussian_vs_collapsed():
    """Verify that isotropic Gaussian latents have much lower SIGReg penalty than collapsed latents."""
    torch.manual_seed(42)
    sigreg = SIGReg(num_slices=128, bandwidth=1.0)

    # 1. Standard normal samples N(0, I)
    z_gaussian = torch.randn(500, 32)
    loss_gaussian = sigreg(z_gaussian)

    # 2. Collapsed constant latents (e.g. all zeros with small jitter)
    z_collapsed = torch.zeros(500, 32) + 0.001 * torch.randn(500, 32)
    loss_collapsed = sigreg(z_collapsed)

    # 3. Off-centered biased latents
    z_biased = 5.0 + torch.randn(500, 32)
    loss_biased = sigreg(z_biased)

    assert loss_gaussian.item() < loss_collapsed.item()
    assert loss_gaussian.item() < loss_biased.item()
    assert loss_gaussian.item() >= 0.0


def test_sigreg_backward():
    """Verify that gradients flow cleanly back through SIGReg."""
    torch.manual_seed(42)
    sigreg = SIGReg(num_slices=32)
    z = torch.randn(64, 16, requires_grad=True)
    loss = sigreg(z)
    loss.backward()

    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
    assert not torch.isinf(z.grad).any()


def test_vicreg_loss():
    """Verify VICReg loss computation and invariance/variance/covariance penalty behavior."""
    torch.manual_seed(42)
    vicreg = VICRegLoss(sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.1)

    z1 = torch.randn(100, 16, requires_grad=True)
    z2 = z1 + 0.1 * torch.randn(100, 16)

    loss, metrics = vicreg(z1, z2)
    assert loss.item() > 0.0
    assert "sim_loss" in metrics
    assert "std_loss" in metrics
    assert "cov_loss" in metrics

    loss.backward()
    assert z1.grad is not None


def test_info_nce_loss():
    """Verify InfoNCE loss and accuracy computation."""
    torch.manual_seed(42)
    info_nce = InfoNCELoss(temperature=0.1)

    z_pred = torch.randn(64, 32, requires_grad=True)
    # Target closely aligned with predictions
    z_target = z_pred + 0.05 * torch.randn(64, 32)

    loss, metrics = info_nce(z_pred, z_target)
    assert loss.item() > 0.0
    assert metrics["top1_acc"] > 0.8

    loss.backward()
    assert z_pred.grad is not None


def test_multistep_and_lewm_loss():
    """Verify MultiStepPredictiveLoss and combined LeWMLoss."""
    torch.manual_seed(42)
    lewm = LeWMLoss(sigreg_weight=0.1, discount=0.9)

    B, K, d = 16, 5, 32
    z_pred_seq = torch.randn(B, K, d, requires_grad=True)
    z_target_seq = torch.randn(B, K, d)
    z_context = torch.randn(B, d)

    loss, metrics = lewm(z_pred_seq, z_target_seq, z_context)
    assert loss.item() > 0.0
    assert "loss_pred" in metrics
    assert "loss_sigreg" in metrics
    assert "loss_total" in metrics

    loss.backward()
    assert z_pred_seq.grad is not None


def test_variational_loss():
    """Verify VariationalJEPALoss computation."""
    torch.manual_seed(42)
    v_loss = VariationalJEPALoss(beta=0.01)

    mu = torch.randn(32, 16, requires_grad=True)
    logvar = torch.zeros(32, 16, requires_grad=True)
    z_target = torch.randn(32, 16)

    loss, metrics = v_loss(mu, logvar, z_target)
    assert loss.item() > 0.0
    assert "loss_nll" in metrics
    assert "loss_kl" in metrics

    loss.backward()
    assert mu.grad is not None
    assert logvar.grad is not None
