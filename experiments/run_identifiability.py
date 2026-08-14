"""
Experiment 2: Linear Identifiability & Anti-Collapse Study across Regularizer Weights.

Empirical verification of Klindt, LeCun, Balestriero (2026) Linear Identifiability Theorem.
Compares representation rank and ground-truth recovery across SIGReg weights lambda.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_wm.core import LatentJEPAWorldModel
from jepa_wm.data import Lorenz63System, create_dataloaders
from jepa_wm.diagnostics import compute_spectral_diagnostics
from jepa_wm.training import JEPATrainer, evaluate_world_model_benchmarks
from jepa_wm.utils import plot_singular_spectrum, save_metrics_json


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Linear Identifiability Benchmark across SIGReg Weights")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dataset
    system = Lorenz63System(dt=0.01)
    num_trajectories = 20
    num_steps = 180
    states, actions = system.generate_trajectories(num_trajectories=num_trajectories, num_steps=num_steps, seed=42)
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1.0 / np.sqrt(3), size=(3, 8))
    observations = np.matmul(states, W) + rng.normal(0, 0.02, size=(num_trajectories, num_steps, 8)).astype(np.float32)

    train_loader, val_loader = create_dataloaders(
        observations=observations,
        actions=actions,
        states=states,
        context_len=10,
        horizon=5,
        batch_size=64,
    )

    lambda_weights = [0.0, 0.01, 0.1, 1.0]
    results_sweep = {}
    spectrum_dict = {}

    for lam in lambda_weights:
        print(f"\n--- Training with SIGReg weight lambda = {lam} ---")
        latent_dim = 32
        model = LatentJEPAWorldModel(
            in_channels=8,
            action_dim=3,
            latent_dim=latent_dim,
            encoder_d_model=64,
            sigreg_weight=lam,
        )

        trainer = JEPATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            device=device,
        )

        trainer.fit(epochs=8, verbose=False)

        metrics = evaluate_world_model_benchmarks(model, val_loader, device=device)
        results_sweep[f"lambda_{lam}"] = metrics

        # Extract validation latents for singular spectrum analysis
        model.eval()
        all_z = []
        with torch.no_grad():
            for batch in val_loader:
                z = model.encoder(batch["obs_ctx"].to(device))
                all_z.append(z.cpu().numpy())
        z_mat = np.concatenate(all_z, axis=0)
        z_c = z_mat - z_mat.mean(axis=0, keepdims=True)
        _, s_vals, _ = np.linalg.svd(z_c, full_matrices=False)
        spectrum_dict[f"lambda={lam}"] = s_vals

        print(f"  Result for lambda={lam}:")
        print(f"    * Effective Rank:             {metrics['effective_rank']:.2f}")
        print(f"    * Linear Identifiability R^2: {metrics['identifiability_r2']:.4f}")
        print(f"    * Procrustes Identifiability: {metrics['procrustes_identifiability']:.4f}")

    os.makedirs("results", exist_ok=True)
    save_metrics_json(results_sweep, "results/identifiability_sweep.json")

    fig = plot_singular_spectrum(
        spectrum_dict,
        title="Singular Spectrum Decay: Impact of SIGReg Anti-Collapse Regularization",
        save_path="results/identifiability_spectrum.png",
    )
    plt.close(fig)
    print("\nIdentifiability benchmark completed and saved to results/")


if __name__ == "__main__":
    main()
