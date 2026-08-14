"""Unit tests for latent space diagnostics and identifiability metrics."""

import numpy as np
import pytest
import torch
from jepa_wm.diagnostics import (
    LinearIdentifiabilityProbe,
    OrthogonalProcrustesAlignment,
    CanonicalCorrelationMetrics,
    compute_effective_rank,
    compute_spectral_diagnostics,
    compute_noise_rejection_ratio,
)


def test_linear_identifiability_probe():
    """Verify LinearIdentifiabilityProbe on synthetic linear transformation."""
    np.random.seed(42)
    N, d_z, d_s = 200, 16, 4
    s = np.random.randn(N, d_s)
    # Linear embedding z = s @ W + noise
    W = np.random.randn(d_s, d_z)
    z = s @ W + 0.01 * np.random.randn(N, d_z)

    probe = LinearIdentifiabilityProbe(alpha=1e-4)
    probe.fit(z[:150], s[:150])
    metrics = probe.evaluate(z[150:], s[150:])

    assert metrics["r2_score"] > 0.95
    assert metrics["mse"] < 0.05
    assert len(metrics["r2_per_dim"]) == d_s


def test_orthogonal_procrustes_alignment():
    """Verify Procrustes alignment on pure orthogonal rotation."""
    np.random.seed(42)
    N, d = 100, 4
    s = np.random.randn(N, d)
    # Generate random orthogonal matrix Q via QR decomposition
    A = np.random.randn(d, d)
    Q, _ = np.linalg.qr(A)

    z = s @ Q  # Perfectly rotated

    discrepancy, R, metrics = OrthogonalProcrustesAlignment.align(z, s)

    assert discrepancy < 1e-4
    assert metrics["identifiability_score"] > 0.999


def test_canonical_correlation_metrics():
    """Verify CCA correlations."""
    np.random.seed(42)
    N, d_z, d_s = 100, 8, 3
    s = np.random.randn(N, d_s)
    z = np.zeros((N, d_z))
    z[:, :d_s] = s  # Perfect canonical alignment on first 3 dims

    cca_results = CanonicalCorrelationMetrics.compute(z, s)
    assert cca_results["mean_cca"] > 0.99
    assert cca_results["top1_cca"] > 0.99


def test_effective_rank():
    """Verify effective rank on isotropic Gaussian vs rank-1 collapsed data."""
    np.random.seed(42)
    # 1. Isotropic Gaussian in 32D
    z_isotropic = np.random.randn(1000, 32)
    erank_iso = compute_effective_rank(z_isotropic)

    # 2. Rank-1 collapsed representation
    z_rank1 = np.outer(np.random.randn(1000), np.ones(32)) + 1e-6 * np.random.randn(1000, 32)
    erank_rank1 = compute_effective_rank(z_rank1)

    assert erank_iso > 25.0  # Close to 32
    assert erank_rank1 < 2.0  # Near 1.0


def test_noise_rejection_ratio():
    """Verify Noise Rejection Ratio when latents contain only physical information."""
    np.random.seed(42)
    N, C_phys, C_nuis, d_z = 200, 4, 8, 16
    phys = np.random.randn(N, C_phys)
    nuis = np.random.randn(N, C_nuis)

    # Latents dependent ONLY on phys
    W = np.random.randn(C_phys, d_z)
    latents = phys @ W + 0.05 * np.random.randn(N, d_z)

    nrr_metrics = compute_noise_rejection_ratio(latents, phys, nuis)

    assert nrr_metrics["r2_physical_channels"] > 0.8
    assert nrr_metrics["r2_nuisance_channels"] < 0.15
    assert nrr_metrics["noise_rejection_ratio"] > 5.0
