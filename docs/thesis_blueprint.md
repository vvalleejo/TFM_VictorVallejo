# Master's Thesis Blueprint (TFM)

**Degree:** Master in Mathematical Engineering (Máster en Ingeniería Matemática)  
**Title:** *Latent World Models for Multivariate Data using JEPA-based Architectures*  
**Author:** Víctor Vallejo  
**Purpose:** Foundational Thesis & Initial Publication Blueprint for PhD Research on World Models.

---

## 📑 Proposed Thesis Structure & Chapter Outline

### Chapter 1: Introduction & State of the Art
- **1.1 Motivation:** The challenge of physical dynamics modeling in high-dimensional noisy sensory environments.
- **1.2 Reconstructive World Models vs Non-Reconstructive World Models:**
  - Autoencoders, Variational Autoencoders (VAEs), Diffusion-based World Models (PlaNet, DreamerV1-V3, DINO-WM).
  - Limitations: Capacity allocation to uninformative stochastic nuisance channels, reconstruction bottleneck.
- **1.3 Joint-Embedding Predictive Architectures (JEPA):**
  - LeCun's Vision: Autonomous Machine Intelligence and Energy-Based Models.
  - I-JEPA (2023), V-JEPA / V-JEPA 2 (2024-2025), LeWorldModel (2026).
- **1.4 Research Objectives & Hypotheses:**
  - *Hypothesis 1 (Noise Invariance):* JEPA models outperform reconstructive models in presence of nuisance noise.
  - *Hypothesis 2 (Linear Identifiability):* Under SIGReg, JEPA representations recover true physical states up to an orthogonal rotation ($h(s) = Qs$).
  - *Hypothesis 3 (Optimal Latent Control):* Latent MPC enables fast, decoder-free closed-loop control.

---

### Chapter 2: Mathematical Foundations
- **2.1 Multivariate Continuous Dynamical Systems:**
  - ODE formalization, Hamiltonian mechanics, Lyapunov exponents, and chaotic attractors (Lorenz-63 / Lorenz-96).
- **2.2 Self-Supervised Learning & The Representation Collapse Problem:**
  - Contrastive methods (InfoNCE) vs. Non-contrastive regularizers (VICReg, Barlow Twins).
- **2.3 The Cramér-Wold Device & Closed-Form Epps-Pulley Characteristic Testing (SIGReg):**
  - Complete derivation of the exact 1D integral.
  - Gradient properties and algorithmic complexity $\mathcal{O}(N^2 M)$.
- **2.4 Information-Theoretic Perspectives:**
  - The Predictive Information Bottleneck (PIB): Maximizing $I(Z_t; Z_{t+k})$ under bounded rate.

---

### Chapter 3: Identifiability Theory in JEPA World Models
- **3.1 The Identifiability Problem in Unsupervised Representation Learning:**
  - Non-identifiability of vanilla non-linear autoencoders (Hyvärinen et al.).
- **3.2 Linear Identifiability Theorem for LeJEPA (Klindt, LeCun, Balestriero 2026):**
  - Hermite spectral decomposition of Gauss-Markov transition operators.
  - Mehler kernel decay analysis: why nonlinear degrees $k \ge 2$ are strictly penalized.
- **3.3 Uniqueness of the Gaussian Latent Distribution:**
  - Sturm-Liouville operator analysis of score functions.
- **3.4 Approximate Identifiability & Smooth Error Bounds:**
  - Degradation bounds under finite sample approximations.

---

### Chapter 4: Architecture & Engineering Design
- **4.1 Multivariate Patch Encoder:**
  - Cross-channel attention mechanisms for variable-channel industrial telemetry.
- **4.2 Action-Conditioned Latent Dynamics Predictors:**
  - Residual GRU Continuous-Time Transition Predictor vs Causal Transformer Predictor.
- **4.3 Multi-Step Rollout Loss & Discounting Schedules.**
- **4.4 Reconstructive World Model Baseline Architecture for Fair Scientific Comparison.**

---

### Chapter 5: Empirical Benchmarks & Experiments
- **5.1 Experiment 1: Recovery of Chaotic Dynamics (Lorenz-63 & Lorenz-96 Attractors):**
  - Phase space orbit reconstruction via linear probing ($R^2 > 0.90$).
  - Multi-step predictive error over long horizons.
- **5.2 Experiment 2: Empirical Validation of Identifiability & SIGReg Regularization Sweeps:**
  - Singular value decay spectra, Effective Rank ($\text{erank}$), and CCA correlations across $\lambda \in [0, 1.0]$.
- **5.3 Experiment 3: Nuisance Noise Invariance vs. Generative Reconstruction:**
  - Benchmark with $C_{nuis} \in [0, 16]$ noise channels demonstrating JEPA's superior robustness.
- **5.4 Experiment 4: Real-Time Latent Planning (CEM & MPPI):**
  - Target tracking and control execution times ($< 5\text{ ms}$ for MPPI).

---

### Chapter 6: Conclusion & PhD Research Roadmap
- **6.1 Summary of Contributions:**
  - First comprehensive software & mathematical framework applying LeJEPA + SIGReg to multivariate continuous dynamical systems.
  - Empirical verification of the Linear Identifiability theorem on chaotic physical manifolds.
- **6.2 Roadmap for PhD Thesis:**
  - *Extension 1:* Neural ODE-JEPAs for irregular, continuous-time asynchronous telemetry.
  - *Extension 2:* Relational & Hierarchical Partitioning Graph-JEPAs (HP-JEPA) for multi-body physical systems.
  - *Extension 3:* Test-Time Adaptation (AdaJEPA) for non-stationary industrial environments.
  - *Extension 4:* Hardware implementation and Zero-Shot Sim-to-Real transfer for agile robotics.

---

## 📊 Summary of Experimental Artifacts Produced in this Codebase

| Benchmark | Script | Metrics File | Generated Figure |
|---|---|---|---|
| **Lorenz-63 Chaos Recovery** | `experiments/train_lorenz_jepa.py` | `results/lorenz_jepa_metrics.json` | `results/lorenz_attractor_recovery.png` |
| **Identifiability & SIGReg Sweep** | `experiments/run_identifiability.py` | `results/identifiability_sweep.json` | `results/identifiability_spectrum.png` |
| **Noise Invariance vs Reconstructive** | `experiments/run_noise_benchmark.py` | `results/noise_benchmark_results.json` | `results/noise_benchmark_results.png` |
| **Latent MPC Planning** | `experiments/run_latent_planning.py` | `results/latent_planning_results.json` | Latent optimization log |
