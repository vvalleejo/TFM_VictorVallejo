# API & Module Reference: JEPA-WM

This document outlines the core classes, methods, and functions in the `jepa_wm` library.

---

## 1. `jepa_wm.core`

### `MultivariatePatchEncoder`
- **Location:** `jepa_wm.core.encoder`
- **Description:** Multi-channel patch-based Transformer encoder for multivariate time series.
- **Parameters:**
  - `in_channels` (int): Number of input telemetry channels $C$.
  - `latent_dim` (int): Dimension of latent output $d_z$.
  - `d_model` (int): Transformer embedding hidden dimension.
  - `num_layers` (int): Number of TransformerEncoder layers.
  - `num_heads` (int): Number of attention heads.
- **Methods:**
  - `forward(x, return_sequence=False)`: Encodes $(B, T, C)$ into $(B, d_z)$ or $(B, T, d_z)$.

### `ResidualGRUPredictor`
- **Location:** `jepa_wm.core.predictor`
- **Description:** Continuous-time inspired recurrent latent dynamics predictor with residual Euler steps.
- **Methods:**
  - `forward(z_init, action_seq)`: Rolls out latent predictions $(B, K, d_z)$ given initial state $z_0$ and future actions.

### `LatentJEPAWorldModel`
- **Location:** `jepa_wm.core.world_model`
- **Description:** End-to-end JEPA World Model with Context Encoder, Target Encoder, Predictor, and SIGReg loss.
- **Methods:**
  - `encode(obs)`: Encodes observation window to latent $z$.
  - `predict_rollout(z_init, actions)`: Rolls out future latent trajectory.
  - `forward(obs_ctx, act_fut, obs_fut=None)`: Full forward pass returning `(z_pred_seq, z_target_seq, loss, metrics)`.

### `ReconstructiveWorldModel`
- **Location:** `jepa_wm.core.baselines`
- **Description:** Reconstructive Autoencoder / VAE World Model baseline for comparative benchmarking.

---

## 2. `jepa_wm.losses`

### `SIGReg`
- **Location:** `jepa_wm.losses.sigreg`
- **Description:** Sketched Isotropic Gaussian Regularizer using Cramér-Wold hypersphere projections and exact closed-form 1D Epps-Pulley test statistic.
- **Parameters:**
  - `num_slices` (int): Number of random 1D projections $M$ (default: 64).
  - `bandwidth` (float): Gaussian weight bandwidth $\sigma_w$ (default: 1.0).

### `LeWMLoss`
- **Location:** `jepa_wm.losses.rollout_loss`
- **Description:** Combined discounted multi-step predictive loss + $\lambda \cdot \text{SIGReg}(Z)$.

### `VICRegLoss` & `InfoNCELoss`
- **Location:** `jepa_wm.losses.vicreg`, `jepa_wm.losses.info_nce`
- **Description:** Comparative self-supervised regularizers.

---

## 3. `jepa_wm.diagnostics`

### `LinearIdentifiabilityProbe`
- **Location:** `jepa_wm.diagnostics.identifiability`
- **Description:** Ridge regression probe testing linear recovery $s \approx W z + b$ of true physical states.
- **Methods:**
  - `fit(z_train, s_train)`: Fits linear probe.
  - `evaluate(z_test, s_test)`: Returns $R^2$, MSE, and per-dimension $R^2$.

### `OrthogonalProcrustesAlignment`
- **Location:** `jepa_wm.diagnostics.identifiability`
- **Description:** Finds optimal rotation $Q \in \mathcal{O}(n)$ minimizing $\|Z Q - S\|_F$.

### `compute_effective_rank`
- **Location:** `jepa_wm.diagnostics.spectral`
- **Description:** Computes Roy & Vetterli effective rank $\exp(H(p_\sigma))$ of representation matrix.

### `compute_noise_rejection_ratio`
- **Location:** `jepa_wm.diagnostics.energy`
- **Description:** Measures ratio of variance explained by physical vs nuisance channels.

---

## 4. `jepa_wm.planning`

### `LatentCEMPlanner`
- **Location:** `jepa_wm.planning.cem`
- **Description:** Cross-Entropy Method optimizing action sequences directly in latent space.

### `LatentMPPIPlanner`
- **Location:** `jepa_wm.planning.mppi`
- **Description:** Model Predictive Path Integral planner with Boltzmann softmax reweighting.
