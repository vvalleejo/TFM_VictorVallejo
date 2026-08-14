"""
Experiment 1: End-to-End Training of Latent JEPA World Model on Chaotic Lorenz-63 Dynamics.

Validates:
1. Convergence of multi-step latent prediction.
2. Prevention of representation collapse via SIGReg.
3. Linear Identifiability recovery of the 3D Lorenz attractor manifold.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_wm.core import LatentJEPAWorldModel
from jepa_wm.data import Lorenz63System, create_dataloaders
from jepa_wm.diagnostics import LinearIdentifiabilityProbe, compute_spectral_diagnostics
from jepa_wm.training import JEPATrainer, evaluate_world_model_benchmarks
from jepa_wm.utils import plot_phase_space_attractor, save_metrics_json


def main():
    print("=" * 70)
    print("EXPERIMENT 1: Training Latent JEPA World Model on Lorenz-63 Attractor")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # 1. Simulate Lorenz-63 Trajectories
    system = Lorenz63System(dt=0.01)
    num_trajectories = 25
    num_steps = 200

    def smooth_control(s, t):
        # Action with periodic sinusoidal forcing
        return 0.5 * np.sin(0.05 * t + np.array([0.0, 1.0, 2.0]))

    print("Simulating chaotic trajectories via RK4...")
    states, actions = system.generate_trajectories(
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        action_fn=smooth_control,
        seed=42,
    )

    # Sensor observation mapping: embed 3D physical state into 8D observation channels
    rng = np.random.default_rng(42)
    W_obs = rng.normal(0, 1.0 / np.sqrt(3), size=(3, 8))
    observations = np.matmul(states, W_obs)
    # Add mild sensor noise
    observations += rng.normal(0, 0.02, size=observations.shape).astype(np.float32)

    # 2. Build DataLoaders
    context_len = 10
    horizon = 5
    train_loader, val_loader = create_dataloaders(
        observations=observations,
        actions=actions,
        states=states,
        context_len=context_len,
        horizon=horizon,
        train_ratio=0.8,
        batch_size=64,
    )

    print(f"Dataset ready: {len(train_loader.dataset)} training windows, {len(val_loader.dataset)} validation windows.")

    # 3. Instantiate Latent JEPA World Model
    latent_dim = 32
    model = LatentJEPAWorldModel(
        in_channels=8,
        action_dim=3,
        latent_dim=latent_dim,
        encoder_d_model=64,
        predictor_type="gru",
        sigreg_weight=0.1,
        discount=0.95,
    )

    # 4. Train Model
    trainer = JEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device=device,
    )

    epochs = 10
    print(f"Training LeJEPA for {epochs} epochs...")
    history = trainer.fit(epochs=epochs, verbose=True)

    # 5. Evaluate Benchmarks
    print("\nRunning comprehensive mathematical evaluation...")
    results = evaluate_world_model_benchmarks(model, val_loader, device=device)

    print("-" * 50)
    print("EVALUATION RESULTS:")
    print(f"  * Effective Rank:             {results['effective_rank']:.2f} / {latent_dim}")
    print(f"  * Linear Identifiability R^2: {results['identifiability_r2']:.4f}")
    print(f"  * Procrustes Discrepancy:     {results['procrustes_discrepancy']:.4f}")
    print(f"  * Canonical Correlation (CCA):{results['cca_mean']:.4f}")
    print(f"  * Multi-Step Pred MSE (step 1):{results['pred_mse_step1']:.6f}")
    print(f"  * Multi-Step Pred MSE (step K):{results['pred_mse_stepK']:.6f}")
    print("-" * 50)

    # 6. Generate 3D Phase Space Reconstruction Plot
    os.makedirs("results", exist_ok=True)
    save_metrics_json(results, "results/lorenz_jepa_metrics.json")

    # Fit linear probe on test trajectory to plot orbit
    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        z_val = model.encoder(val_batch["obs_ctx"].to(device)).cpu().numpy()
    s_val = val_batch["state_ctx"].numpy()

    probe = LinearIdentifiabilityProbe(alpha=1e-3).fit(z_val, s_val)
    s_pred = probe.probe.predict(z_val)

    fig = plot_phase_space_attractor(
        s_val,
        s_pred,
        title="Lorenz-63 Attractor Recovery via JEPA + SIGReg",
        save_path="results/lorenz_attractor_recovery.png",
    )
    plt.close(fig)
    print("Results and figures saved to results/")


if __name__ == "__main__":
    main()
