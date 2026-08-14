"""
Spectral Diagnostics and Dimensionality Analysis of Latent Spaces.

Implements Effective Rank (Roy & Vetterli 2007) and Singular Spectrum Decay.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.linalg
import torch


def compute_effective_rank(z: np.ndarray) -> float:
    """
    Computes the Effective Rank (erank) of representation matrix Z in R^(N x d):
        erank(Z) = exp( - sum_{i=1}^d p_i * ln(p_i) )
        where p_i = sigma_i / sum_j sigma_j

    Quantifies the effective dimensional support of the learned latent manifold.
    Maximum erank = d (uniform isotropic distribution), minimum erank = 1 (complete collapse).

    Args:
        z (np.ndarray): Latent representation matrix (N, d).

    Returns:
        float: Effective rank in [1.0, d].
    """
    # Center representation matrix
    z_c = z - z.mean(axis=0, keepdims=True)

    # Compute singular values
    _, s, _ = scipy.linalg.svd(z_c, full_matrices=False)

    s_sum = np.sum(s)
    if s_sum <= 1e-12:
        return 1.0

    p = s / s_sum
    # Filter out zero probabilities for Shannon entropy computation
    p_nonzero = p[p > 1e-12]
    shannon_entropy = -np.sum(p_nonzero * np.log(p_nonzero))

    return float(np.exp(shannon_entropy))


def compute_spectral_diagnostics(z: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive spectral properties:
    - effective_rank: Effective dimensional coverage.
    - spectral_entropy: Shannon entropy of singular spectrum.
    - top_singular_ratio: Proportion of variance in the leading singular value.
    - condition_number: Ratio of largest to smallest singular value.
    """
    z_c = z - z.mean(axis=0, keepdims=True)
    _, s, _ = scipy.linalg.svd(z_c, full_matrices=False)

    erank = compute_effective_rank(z)
    s_sum = np.sum(s)
    p = s / (s_sum + 1e-12)
    p_nonzero = p[p > 1e-12]
    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))

    top_ratio = float(s[0] / (s_sum + 1e-12)) if len(s) > 0 else 1.0
    cond_num = float(s[0] / (s[-1] + 1e-8)) if len(s) > 0 else 1.0

    return {
        "effective_rank": erank,
        "spectral_entropy": entropy,
        "top_singular_ratio": top_ratio,
        "condition_number": cond_num,
        "latent_dim": float(z.shape[1]),
    }
