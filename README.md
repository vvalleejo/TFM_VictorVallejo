# Latent World Models for Multivariate Data using JEPA-based Architectures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Master's Thesis in Mathematical Engineering and Computer Science (TFM)**
> **Author:** Víctor Vallejo
> **Topic:** Joint-Embedding Predictive Architectures (JEPA) as Latent World Models for Continuous & Chaotic Multivariate Dynamical Systems.

---

## 🔬 Abstract & Research Motivation

Traditional World Models (such as Dreamer or PlaNet) rely heavily on **generative observation reconstruction** via Autoencoders, VAEs, or Diffusion Models. However, when applied to complex physical systems, sensor networks, or industrial telemetry, raw observations contain substantial high-entropy stochastic noise and non-predictable nuisance variables. Reconstructive objectives allocate disproportionate network capacity to model these irrelevant details, leading to suboptimal latent dynamics.

This work implements and studies **Joint-Embedding Predictive Architectures (JEPA)** as Latent World Models for multivariate time series and continuous dynamical systems. We leverage:

1. **Self-Supervised Latent Dynamics Prediction** ($P_\phi(z_t, a_{t:t+k}) \to \hat{z}_{t+k}$) without observation decoding.
2. **Sketched Isotropic Gaussian Regularization (SIGReg)** based on the Cramér-Wold projection device and exact closed-form 1D Epps-Pulley test statistic to prevent representation collapse.
3. **Linear Identifiability & Physical Recovery**: Theoretical and empirical validation that LeJEPA recovers ground-truth physical state variables up to an orthogonal rotation ($h(s) = Qs$).
4. **Direct Latent Model Predictive Control (MPC)**: Real-time action sequence optimization via Cross-Entropy Method (CEM) and Model Predictive Path Integral (MPPI) operating entirely in the latent space.

---

## 🏛️ Architecture Overview

```mermaid
flowchart LR
    subgraph Observations
        X_hist["Past Multivariate History\n x(t-L:t) ∈ R^(L×C)"]
        X_fut["Future Targets\n x(t+1:t+K) ∈ R^(K×C)"]
    end

    subgraph Encoder
        Enc["Multivariate Patch Encoder\n E_θ"]
    end

    subgraph Latent_Space
        Z_t["Latent State\n z_t = E_θ(X_hist)"]
        Z_targets["Target Latents\n z(t+1:t+K) = E_θ(X_fut)"]
    end

    subgraph Predictor
        Actions["Action Trajectory\n a(t:t+K-1)"]
        Pred["Latent Predictor P_ϕ"]
        Z_hat["Predicted Latents\n z_hat(t+1:t+K)"]
    end

    subgraph Objective
        Loss_Pred["L_pred = ||z_hat - z_target||²"]
        Loss_SIGReg["L_SIGReg = (1/M) ∑ T(Z·u_m)"]
        Total_Loss["L_total = L_pred + λ · L_SIGReg"]
    end

    X_hist --> Enc --> Z_t
    X_fut --> Enc --> Z_targets
    Z_t & Actions --> Pred --> Z_hat
    Z_hat & Z_targets --> Loss_Pred
    Z_t --> Loss_SIGReg
    Loss_Pred & Loss_SIGReg --> Total_Loss
```

---

## 📦 Project Structure

```
├── jepa_wm/
│   ├── core/                  # Multivariate Encoders, Predictors, World Models & Baselines
│   ├── losses/                # SIGReg (closed-form), VICReg, InfoNCE, Rollout Loss
│   ├── data/                  # Dynamical systems (Lorenz-63/96, Oscillators, Nuisance Noise)
│   ├── diagnostics/           # Linear Identifiability, Effective Rank, CCA, Procrustes
│   ├── planning/              # Latent MPC: CEM and MPPI Planners
│   ├── training/              # Trainers, Schedulers, and Evaluators
│   └── utils/                 # Logging, Metrics, and Visualizations
├── experiments/               # Reproducible benchmark experiments
├── tests/                     # Comprehensive pytest test suite
└── docs/                      # Mathematical derivations, thesis blueprint, and API documentation
```

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/vvalleejo/TFM_VictorVallejo.git
cd TFM_VictorVallejo

# Install package in editable mode with development dependencies
pip install -e .
pip install -e ".[dev]"

# Run tests
pytest tests/
```

---

## 📚 Mathematical Documentation

See [docs/theoretical_foundations.md](docs/theoretical_foundations.md) and [docs/thesis_blueprint.md](docs/thesis_blueprint.md) for full mathematical derivations, proofs of linear identifiability, and the complete Master's Thesis chapter plan.
