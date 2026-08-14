# Mathematical Foundations of Latent World Models using JEPA Architectures

**Master's Thesis in Mathematical Engineering**  
**Author:** Víctor Vallejo  
**Title:** *Latent World Models for Multivariate Data using JEPA-based Architectures*

---

## 1. Problem Formulation & Dynamical System Formalization

Consider an autonomous or controlled physical dynamical system evolving in continuous time over a latent state manifold $\mathcal{S} \subseteq \mathbb{R}^{d_s}$:

$$\frac{d s(t)}{dt} = f(s(t), a(t)) + \eta(t)$$

where:
- $s(t) \in \mathcal{S}$ is the true, unobserved physical state (e.g., positions, momenta, fluid velocities, attractor coordinates).
- $a(t) \in \mathcal{A} \subseteq \mathbb{R}^{d_a}$ is the control action input.
- $\eta(t)$ represents internal process disturbances or stochastic thermal fluctuations.

The system is observed through a discrete-time measurement operator $g: \mathcal{S} \times \mathcal{V} \to \mathbb{R}^C$ sampled at period $\Delta t$:

$$x_t = g(s_t, \nu_t) + \epsilon_t, \quad t \in \{1, \dots, T\}$$

where:
- $x_t \in \mathbb{R}^C$ is a multivariate telemetry vector across $C$ sensory channels.
- $\nu_t \in \mathbb{R}^{d_\nu}$ represents **uninformative, high-entropy stochastic nuisance noise** (e.g., sensor flutter, electronic thermal hum, uncoupled environmental background noise).
- $\epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2 I)$ is additive measurement noise.

---

## 2. Reconstructive World Models vs. Joint-Embedding Predictive Architectures

### 2.1 The Failure Mode of Reconstructive Models (Autoencoders / VAEs / Diffusion)
Traditional world models (e.g., PlaNet, Dreamer, World Models) optimize an observation reconstruction objective:

$$\min_{\theta, \phi, \psi} \mathbb{E}_{(x, a)} \left[ \sum_{k=1}^K \left\| x_{t+k} - D_\psi(P_\phi(E_\theta(x_{\le t}), a_{t:t+k-1})) \right\|_2^2 \right]$$

By Shannon's source coding theorem and rate-distortion theory, minimizing MSE in observation space forces the encoder $E_\theta$ to allocate latent capacity proportional to the entropy rate of each channel:

$$H(x_t) = H(g(s_t)) + H(\nu_t) + I(g(s_t); \nu_t)$$

When nuisance noise $\nu_t$ has high entropy (e.g., white or colored noise), the decoder $D_\psi$ forces the latent code $z_t$ to encode $\nu_t$ rather than the subtle, lower-entropy deterministic dynamics of $s_t$. Consequently, downstream predictive planning degrades.

### 2.2 The JEPA Paradigm & Predictive Information Bottleneck (PIB)
Joint-Embedding Predictive Architectures (LeCun 2022/2023) eliminate the decoder $D_\psi$. The objective is formulated strictly in an abstract latent representation space $\mathcal{Z} \subseteq \mathbb{R}^{d_z}$:

$$\min_{\theta, \phi} \mathcal{L}_{JEPA} = \mathbb{E} \left[ \sum_{k=1}^K \gamma^{k-1} \mathcal{D}\left( P_\phi(E_\theta(x_{\le t}), a_{t:t+k-1}), E_{\bar{\theta}}(x_{t+k}) \right) \right] + \lambda \mathcal{R}(Z)$$

From an information-theoretic perspective, JEPA realizes a **Predictive Information Bottleneck (PIB)**:

$$\max_{\theta} I(Z_t; Z_{t+k}) \quad \text{subject to} \quad I(Z_t; X_t) \le I_c$$

Because nuisance noise $\nu_t$ is temporally uncorrelated across time steps ($I(\nu_t; \nu_{t+k}) \approx 0$), the optimal predictive representation filters out $\nu_t$ natively without supervision.

---

## 3. Representation Collapse & Sketched Isotropic Gaussian Regularization (SIGReg)

### 3.1 The Collapse Phenomenon
Without a reconstruction loss, unregularized optimization of $\mathcal{D}(P_\phi(z_t, a), z_{t+1}) = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$ has a trivial global minimum:

$$E_\theta(x) \equiv c \quad \forall x \in \mathcal{X}, \quad P_\phi(c, a) \equiv c \implies \mathcal{L}_{pred} = 0$$

where the latent space collapses to a zero-dimensional point or a rank-1 subspace.

### 3.2 Cramér-Wold Device and Closed-Form Epps-Pulley Statistic
To guarantee full-rank representations and prevent collapse, Balestriero & LeCun (2025) introduced **SIGReg**.

#### Theorem (Cramér-Wold Theorem)
*A multivariate random vector $Z \in \mathbb{R}^d$ is distributed according to standard isotropic Gaussian $\mathcal{N}(0, I_d)$ if and only if for all 1D unit projection directions $u \in \mathbb{S}^{d-1}$, the scalar projection $h = Z^T u$ is distributed according to univariate standard normal $\mathcal{N}(0, 1)$.*

SIGReg sketches the representation by sampling $M$ uniform directions $u^{(1)}, \dots, u^{(M)} \sim \text{Uniform}(\mathbb{S}^{d-1})$ and evaluates the unweighted 1D **Epps-Pulley empirical characteristic function test statistic**:

$$T(h) = \int_{-\infty}^\infty \left| \phi_N(t; h) - \phi_0(t) \right|^2 w(t) dt$$

where:
- $\phi_N(t; h) = \frac{1}{N} \sum_{j=1}^N e^{i t h_j}$ is the empirical characteristic function.
- $\phi_0(t) = e^{-\frac{t^2}{2}}$ is the characteristic function of $\mathcal{N}(0, 1)$.
- $w(t) = e^{-\frac{t^2}{2\sigma_w^2}}$ is the Gaussian kernel weight.

#### Analytical Closed-Form Derivation
Expanding the squared magnitude:

$$\left| \phi_N(t) - \phi_0(t) \right|^2 = \frac{1}{N^2}\sum_{j,l} \cos(t(h_j - h_l)) - \frac{2}{N}\sum_j \cos(t h_j) e^{-\frac{t^2}{2}} + e^{-t^2}$$

Using the standard Gaussian integral $\int_{-\infty}^\infty \cos(t \Delta) e^{-\alpha t^2} dt = \sqrt{\frac{\pi}{\alpha}} e^{-\frac{\Delta^2}{4\alpha}}$, we integrate each term with respect to $w(t)$:

1. **Empirical Pairwise Term:**
   $$\int_{-\infty}^\infty \cos(t(h_j - h_l)) e^{-\frac{t^2}{2\sigma_w^2}} dt = \sqrt{2\pi \sigma_w^2} \exp\left( -\frac{\sigma_w^2 (h_j - h_l)^2}{2} \right)$$

2. **Cross Term:**
   $$\int_{-\infty}^\infty \cos(t h_j) e^{-\frac{t^2(\sigma_w^2 + 1)}{2\sigma_w^2}} dt = \sqrt{\frac{2\pi \sigma_w^2}{\sigma_w^2 + 1}} \exp\left( -\frac{\sigma_w^2 h_j^2}{2(\sigma_w^2 + 1)} \right)$$

3. **Target Constant Term:**
   $$\int_{-\infty}^\infty e^{-\frac{t^2(2\sigma_w^2 + 1)}{2\sigma_w^2}} dt = \sqrt{\frac{2\pi \sigma_w^2}{2\sigma_w^2 + 1}}$$

Normalizing by $\int w(t) dt = \sqrt{2\pi \sigma_w^2}$, the exact scalar statistic is:

$$\boxed{T(h) = \frac{1}{N^2}\sum_{j=1}^N\sum_{l=1}^N \exp\left( -\frac{\sigma_w^2(h_j - h_l)^2}{2} \right) - \frac{2}{N\sqrt{\sigma_w^2 + 1}}\sum_{j=1}^N \exp\left( -\frac{\sigma_w^2 h_j^2}{2(\sigma_w^2 + 1)} \right) + \frac{1}{\sqrt{2\sigma_w^2 + 1}}}$$

---

## 4. Formal Theory of Linear Identifiability (Klindt, LeCun, Balestriero 2026)

### 4.1 Theorem 1: Linear Identifiability on Gauss-Markov Systems
Assume the underlying physical state evolves according to a stationary Gauss-Markov process (Ornstein-Uhlenbeck):

$$s_{t+1} = \rho s_t + \sqrt{1 - \rho^2} \eta_t, \quad \eta_t \sim \mathcal{N}(0, I), \quad |\rho| < 1$$

Let $h: \mathcal{S} \to \mathcal{Z}$ be an encoder trained with LeJEPA (alignment loss + SIGReg).

**Result:** Any representation $h(s)$ that minimizes the LeJEPA objective while satisfying Gaussian marginal constraints is an **exact orthogonal rotation of the true state**:

$$h(s) = Q s \quad \text{for some } Q \in \mathcal{O}(d_s)$$

#### Proof Sketch (Mehler Hermite Polynomial Spectral Decomposition)
The transition density of the OU process admits an expansion in Hermite polynomials $H_k(x)$ via Mehler's formula:

$$p(s_{t+1} \mid s_t) = p(s_{t+1}) \sum_{k=0}^\infty \rho^k H_k(s_t) H_k(s_{t+1})$$

The predictive alignment term decomposes over orthogonal Hermite modes:

$$\mathbb{E}[\|h(s_{t+1}) - \hat{h}(s_{t+1})\|^2] = \sum_{k=1}^\infty (1 - \rho^{2k}) \|c_k\|^2$$

Since $\rho^{2k} < \rho^2$ strictly for all non-linear degrees $k \ge 2$, non-linear representations are strictly penalized compared to the linear mode $k=1$. Hence, the global minimizer is uniquely linear.

### 4.2 Theorem 2: Uniqueness of the Gaussian Distribution
Among all stationary additive-noise Markov processes, the **Gaussian distribution is the UNIQUE distribution** for which the linear mode dominates all higher-order eigenfunctions under Sturm-Liouville spectral analysis.

---

## 5. Latent Space Optimal Control & Model Predictive Control (MPC)

### 5.1 Theorem (Equivalence of Latent Planning)
If $h(s) = Q s$ with $Q^T Q = I$, then Euclidean distances in latent space equal physical Euclidean distances in true state space:

$$\|z_t - z_{goal}\|_2^2 = \|Q s_t - Q s_{goal}\|_2^2 = (s_t - s_{goal})^T Q^T Q (s_t - s_{goal}) = \|s_t - s_{goal}\|_2^2$$

Thus, trajectory planning directly in latent space without decoding is mathematically isomorphic to planning with full ground-truth state knowledge.

### 5.2 Latent MPPI & CEM Algorithms
- **CEM:** Iteratively refines Gaussian candidate action distribution $\mathcal{N}(\mu_t, \Sigma_t)$ by taking the top elite trajectories:
  $$\mu^{(k+1)} = (1-\alpha)\mu_{elite} + \alpha \mu^{(k)}$$
- **MPPI:** Computes action updates via Boltzmann-reweighted path integrals:
  $$w_i = \frac{\exp\left(-\frac{1}{\lambda_{temp}} S(\tau_i)\right)}{\sum_j \exp\left(-\frac{1}{\lambda_{temp}} S(\tau_j)\right)}, \quad a_t^* = \sum_i w_i a_t^{(i)}$$
