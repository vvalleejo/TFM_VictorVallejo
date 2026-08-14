"""
Experiment 3: Nuisance Noise Robustness Benchmark — JEPA vs Reconstructive World Models.

Tests Hypothesis 1: Under uninformative stochastic nuisance channels, JEPA World Models
retain high predictive dynamical fidelity, whereas Reconstructive World Models suffer
capacity degradation due to pixel/sensor noise reconstruction.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jepa_wm.core import LatentJEPAWorldModel, ReconstructiveWorldModel
from jepa_wm.data import MultiSensorIndustrialBenchmark, create_dataloaders
from jepa_wm.diagnostics import LinearIdentifiabilityProbe
from jepa_wm.training import JEPATrainer
from jepa_wm.utils import plot_noise_benchmark, save_metrics_json


def train_reconstructive_baseline(
    model: ReconstructiveWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Trains Reconstructive World Model baseline and returns validation linear identifiability R2."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            obs_ctx = batch["obs_ctx"].to(device)
            act_fut = batch["act_fut"].to(device)
            obs_fut = batch["obs_fut"].to(device)

            optimizer.zero_grad()
            _, obs_pred, metrics = model(obs_ctx, act_fut, obs_fut)
            loss = nn.functional.mse_loss(obs_pred, obs_fut)
            loss.backward()
            optimizer.step()

    # Evaluate Linear Probe R2
    model.eval()
    all_z = []
    all_s = []
    with torch.no_grad():
        for batch in val_loader:
            z = model.encode(batch["obs_ctx"].to(device))
            all_z.append(z.cpu().numpy())
            all_s.append(batch["state_ctx"].numpy())

    z_mat = np.concatenate(all_z, axis=0)
    s_mat = np.concatenate(all_s, axis=0)
    n_train = int(0.7 * len(z_mat))

    probe = LinearIdentifiabilityProbe(alpha=1e-3).fit(z_mat[:n_train], s_mat[:n_train])
    res = probe.evaluate(z_mat[n_train:], s_mat[n_train:])
    return res["r2_score"]


def main():
    print("=" * 70)
    print("EXPERIMENT 3: JEPA vs Reconstructive World Model under Nuisance Noise")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nuisance_levels = [0, 4, 8, 16]

    jepa_r2_scores = []
    recon_r2_scores = []

    for n_nuis in nuisance_levels:
        print(f"\n--- Testing with C_nuisance = {n_nuis} Noise Channels ---")
        benchmark = MultiSensorIndustrialBenchmark(
            system_type="lorenz",
            num_physical_channels=6,
            num_nuisance_channels=n_nuis,
            noise_level=0.05,
        )
        obs, acts, states = benchmark.generate_dataset(num_trajectories=20, num_steps=180, seed=42)

        train_loader, val_loader = create_dataloaders(
            observations=obs,
            actions=acts,
            states=states,
            context_len=10,
            horizon=5,
            batch_size=64,
        )

        total_channels = 6 + n_nuis
        latent_dim = 32

        # 1. Train JEPA World Model
        print("  Training JEPA World Model (Ours)...")
        jepa = LatentJEPAWorldModel(
            in_channels=total_channels,
            action_dim=3,
            latent_dim=latent_dim,
            encoder_d_model=64,
            sigreg_weight=0.1,
        )
        jepa_trainer = JEPATrainer(jepa, train_loader, val_loader, lr=1e-3, device=device)
        jepa_trainer.fit(epochs=8, verbose=False)
        jepa_eval = jepa_trainer.evaluate()
        jepa_r2 = jepa_eval["val_identifiability_r2"]
        jepa_r2_scores.append(jepa_r2)
        print(f"    -> JEPA Linear Probe R^2: {jepa_r2:.4f}")

        # 2. Train Reconstructive World Model Baseline
        print("  Training Reconstructive Baseline (MAE/VAE style)...")
        recon = ReconstructiveWorldModel(
            in_channels=total_channels,
            action_dim=3,
            latent_dim=latent_dim,
            hidden_dim=64,
        )
        recon_r2 = train_reconstructive_baseline(recon, train_loader, val_loader, epochs=8, lr=1e-3, device=device)
        recon_r2_scores.append(recon_r2)
        print(f"    -> Reconstructive Linear Probe R^2: {recon_r2:.4f}")

    os.makedirs("results", exist_ok=True)
    summary_results = {
        "nuisance_channels": nuisance_levels,
        "jepa_r2_scores": jepa_r2_scores,
        "reconstructive_r2_scores": recon_r2_scores,
    }
    save_metrics_json(summary_results, "results/noise_benchmark_results.json")

    fig = plot_noise_benchmark(
        nuisance_counts=nuisance_levels,
        jepa_r2_scores=jepa_r2_scores,
        recon_r2_scores=recon_r2_scores,
        title="JEPA vs Reconstructive World Model: Downstream R^2 vs Nuisance Noise",
        save_path="results/noise_benchmark_results.png",
    )
    plt.close(fig)
    print("\nNoise benchmark completed and saved to results/")


if __name__ == "__main__":
    main()
