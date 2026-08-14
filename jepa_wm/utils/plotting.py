"""
Visualization Utilities for Master's Thesis Experiments and Publications.

Generates publication-quality figures:
1. Phase space attractor orbits (ground truth vs latent probe reconstruction).
2. Singular value spectrum decay across regularizers.
3. Multi-step rollout horizon error curves.
4. Noise rejection ratio vs nuisance channel count.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_phase_space_attractor(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    title: str = "Lorenz Attractor: Ground Truth vs Latent JEPA Linear Probe",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots 3D phase space orbit comparing true physical states to linear probe reconstruction.
    """
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], color="blue", alpha=0.8, lw=0.8)
    ax1.set_title("Ground Truth State $s(t)$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_zlabel("$z$")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(pred_states[:, 0], pred_states[:, 1], pred_states[:, 2], color="crimson", alpha=0.8, lw=0.8)
    ax2.set_title("Identified State $\\hat{s}(t) = W z(t) + b$")
    ax2.set_xlabel("$\\hat{x}$")
    ax2.set_ylabel("$\\hat{y}$")
    ax2.set_zlabel("$\\hat{z}$")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_singular_spectrum(
    spectrum_dict: Dict[str, np.ndarray],
    title: str = "Latent Covariance Singular Value Decay Spectrum",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots singular value decay comparing SIGReg, VICReg, and No-Regularization (collapse).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, s_vals in spectrum_dict.items():
        s_norm = s_vals / (np.sum(s_vals) + 1e-12)
        ax.plot(np.arange(1, len(s_norm) + 1), s_norm, marker="o", markersize=4, label=label)

    ax.set_xlabel("Singular Value Index $i$")
    ax.set_ylabel("Normalized Singular Value $\\sigma_i / \\sum \\sigma$")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_noise_benchmark(
    nuisance_counts: List[int],
    jepa_r2_scores: List[float],
    recon_r2_scores: List[float],
    title: str = "World Model Predictive Performance vs Nuisance Noise Dimensions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots JEPA vs Reconstructive World Model downstream predictive fidelity as uninformative
    noise channels increase.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(nuisance_counts, jepa_r2_scores, "o-", color="darkgreen", lw=2, label="JEPA World Model (Ours)")
    ax.plot(nuisance_counts, recon_r2_scores, "s--", color="firebrick", lw=2, label="Reconstructive World Model (VAE/MAE)")

    ax.set_xlabel("Number of Nuisance Noise Channels $C_{nuisance}$")
    ax.set_ylabel("Downstream Linear Identifiability $R^2$ Score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
