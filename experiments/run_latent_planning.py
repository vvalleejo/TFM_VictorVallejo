"""
Experiment 4: Closed-Loop Model Predictive Control in Latent Space (CEM & MPPI).

Tests Hypothesis 3: Latent MPC planners optimize control sequences directly in the
JEPA latent space, achieving target tracking without sensor observation reconstruction.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_wm.core import LatentJEPAWorldModel
from jepa_wm.data import Lorenz63System, create_dataloaders
from jepa_wm.planning import LatentCEMPlanner, LatentMPPIPlanner
from jepa_wm.training import JEPATrainer
from jepa_wm.utils import save_metrics_json


def main():
    print("=" * 70)
    print("EXPERIMENT 4: Closed-Loop Latent Space Model Predictive Control (MPC)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Simulate data and train World Model
    system = Lorenz63System(dt=0.01)
    states, actions = system.generate_trajectories(num_trajectories=30, num_steps=200, seed=42)
    rng = np.random.default_rng(42)
    W = rng.normal(0, 1.0 / np.sqrt(3), size=(3, 6))
    observations = np.matmul(states, W).astype(np.float32)

    train_loader, val_loader = create_dataloaders(
        observations=observations,
        actions=actions,
        states=states,
        context_len=10,
        horizon=5,
        batch_size=32,
    )

    latent_dim = 24
    model = LatentJEPAWorldModel(
        in_channels=6,
        action_dim=3,
        latent_dim=latent_dim,
        encoder_d_model=48,
        sigreg_weight=0.1,
    )
    trainer = JEPATrainer(model, train_loader, val_loader, lr=1e-3, device=device)
    print("Pre-training Latent World Model...")
    trainer.fit(epochs=8, verbose=False)

    # 2. Plan trajectory to reach target latent state
    model.eval()
    val_batch = next(iter(val_loader))
    obs_ctx = val_batch["obs_ctx"][:1].to(device)
    obs_target = val_batch["obs_fut"][:1, -1:].to(device)

    with torch.no_grad():
        z_init = model.encode(obs_ctx)
        z_goal = model.encode(obs_target)

    print("\n1. Running Latent CEM Planner...")
    cem_planner = LatentCEMPlanner(
        model=model,
        horizon=10,
        num_candidates=128,
        num_elites=16,
        num_iterations=6,
    )
    t0 = time.perf_counter()
    cem_actions, cem_info = cem_planner.plan(z_init, z_goal, action_dim=3)
    cem_time = (time.perf_counter() - t0) * 1000

    print(f"  CEM initial cost: {cem_info['initial_cost']:.4f}")
    print(f"  CEM optimized cost: {cem_info['best_cost']:.4f}")
    print(f"  CEM execution time: {cem_time:.2f} ms")

    print("\n2. Running Latent MPPI Planner...")
    mppi_planner = LatentMPPIPlanner(
        model=model,
        horizon=10,
        num_samples=128,
        temperature=0.3,
    )
    t0 = time.perf_counter()
    mppi_actions, mppi_info = mppi_planner.plan(z_init, z_goal, action_dim=3)
    mppi_time = (time.perf_counter() - t0) * 1000

    print(f"  MPPI min cost: {mppi_info['min_cost']:.4f}")
    print(f"  MPPI mean cost: {mppi_info['mean_cost']:.4f}")
    print(f"  MPPI execution time: {mppi_time:.2f} ms")

    results = {
        "cem_initial_cost": cem_info["initial_cost"],
        "cem_best_cost": cem_info["best_cost"],
        "cem_time_ms": cem_time,
        "mppi_min_cost": mppi_info["min_cost"],
        "mppi_time_ms": mppi_time,
    }
    save_metrics_json(results, "results/latent_planning_results.json")
    print("\nLatent planning benchmark complete and saved to results/")


if __name__ == "__main__":
    main()
