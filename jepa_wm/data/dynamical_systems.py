"""
Continuous and Chaotic Multivariate Dynamical Systems Simulators.

Includes:
1. Lorenz-63 Chaotic Attractor (3D state with optional control forcing).
2. Lorenz-96 Atmospheric Multi-Scale Spatiotemporal Dynamics (N-dimensional).
3. Coupled Hamiltonian Non-Linear Oscillators.

Uses high-precision 4th-Order Runge-Kutta (RK4) integration.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import torch


class Lorenz63System:
    """
    Lorenz-63 Chaotic Dynamical System:
        dx/dt = sigma * (y - x) + u_x
        dy/dt = x * (rho - z) - y + u_y
        dz/dt = x * y - beta * z + u_z

    Attributes:
        sigma (float): Prandtl number (default: 10.0).
        rho (float): Rayleigh number (default: 28.0 for chaotic regime).
        beta (float): Geometric factor (default: 8.0 / 3.0).
        dt (float): Integration time step (default: 0.01).
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.state_dim = 3
        self.action_dim = 3

    def dynamics(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes time derivatives [dx/dt, dy/dt, dz/dt]."""
        x, y, z = state[..., 0], state[..., 1], state[..., 2]
        u = action if action is not None else np.zeros_like(state)

        dx = self.sigma * (y - x) + u[..., 0]
        dy = x * (self.rho - z) - y + u[..., 1]
        dz = x * y - self.beta * z + u[..., 2]

        return np.stack([dx, dy, dz], axis=-1)

    def step_rk4(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Advances state by one time step dt using 4th-Order Runge-Kutta."""
        dt = self.dt
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * dt * k1, action)
        k3 = self.dynamics(state + 0.5 * dt * k2, action)
        k4 = self.dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def generate_trajectories(
        self,
        num_trajectories: int,
        num_steps: int,
        action_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        seed: Optional[int] = None,
        burn_in: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates multiple trajectories.

        Args:
            num_trajectories (int): Batch of trajectories.
            num_steps (int): Length of each trajectory after burn-in.
            action_fn (Callable, optional): Returns action given (state, step_idx).
            seed (int, optional): Random seed.
            burn_in (int): Steps to discard to reach chaotic attractor basin.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (states of shape (B, T, 3), actions of shape (B, T, 3)).
        """
        rng = np.random.default_rng(seed)
        # Initialize near the attractor basin
        initial_states = rng.normal(loc=[0.0, 0.0, 25.0], scale=[2.0, 2.0, 2.0], size=(num_trajectories, 3))

        # Burn-in integration
        curr = initial_states
        for _ in range(burn_in):
            curr = self.step_rk4(curr)

        states = np.zeros((num_trajectories, num_steps, 3), dtype=np.float32)
        actions = np.zeros((num_trajectories, num_steps, 3), dtype=np.float32)

        for t in range(num_steps):
            states[:, t] = curr
            if action_fn is not None:
                act = action_fn(curr, t)
            else:
                act = np.zeros((num_trajectories, 3), dtype=np.float32)
            actions[:, t] = act
            curr = self.step_rk4(curr, act)

        return states, actions


class Lorenz96System:
    """
    Lorenz-96 Spatiotemporal Ring Dynamical System:
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F + u_i
    with periodic boundary conditions x_{-1} = x_{N-1}, x_0 = x_N, etc.

    Attributes:
        dim (int): Dimensionality N of the spatial ring (default: 8).
        forcing (float): Constant external forcing F (default: 8.0 for chaotic regime).
        dt (float): Integration step (default: 0.01).
    """

    def __init__(self, dim: int = 8, forcing: float = 8.0, dt: float = 0.01) -> None:
        self.dim = dim
        self.forcing = forcing
        self.dt = dt
        self.state_dim = dim
        self.action_dim = dim

    def dynamics(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes derivatives for N-dimensional Lorenz-96 ring."""
        N = self.dim
        u = action if action is not None else 0.0
        # Periodic shift roll
        x_plus1 = np.roll(state, -1, axis=-1)
        x_minus1 = np.roll(state, 1, axis=-1)
        x_minus2 = np.roll(state, 2, axis=-1)

        dstate = (x_plus1 - x_minus2) * x_minus1 - state + self.forcing + u
        return dstate

    def step_rk4(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Advances state by one time step dt."""
        dt = self.dt
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * dt * k1, action)
        k3 = self.dynamics(state + 0.5 * dt * k2, action)
        k4 = self.dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def generate_trajectories(
        self,
        num_trajectories: int,
        num_steps: int,
        action_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        seed: Optional[int] = None,
        burn_in: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates Lorenz-96 trajectories."""
        rng = np.random.default_rng(seed)
        # Small perturbation around equilibrium F
        initial_states = self.forcing + rng.normal(scale=1.0, size=(num_trajectories, self.dim))

        curr = initial_states
        for _ in range(burn_in):
            curr = self.step_rk4(curr)

        states = np.zeros((num_trajectories, num_steps, self.dim), dtype=np.float32)
        actions = np.zeros((num_trajectories, num_steps, self.dim), dtype=np.float32)

        for t in range(num_steps):
            states[:, t] = curr
            act = action_fn(curr, t) if action_fn is not None else np.zeros((num_trajectories, self.dim), dtype=np.float32)
            actions[:, t] = act
            curr = self.step_rk4(curr, act)

        return states, actions


class CoupledOscillatorsSystem:
    """
    Coupled Hamiltonian Non-linear Oscillators with control torques/forces:
        q_i: generalized positions
        p_i: generalized momenta
        dH/dp_i = p_i / m_i
        dH/dq_i = -k * (q_i - q_{i-1}) - k * (q_i - q_{i+1}) - alpha * q_i^3
    """

    def __init__(
        self,
        num_oscillators: int = 4,
        coupling_k: float = 1.5,
        nonlinearity_alpha: float = 0.5,
        damping: float = 0.05,
        dt: float = 0.02,
    ) -> None:
        self.num_oscillators = num_oscillators
        self.coupling_k = coupling_k
        self.alpha = nonlinearity_alpha
        self.damping = damping
        self.dt = dt
        self.state_dim = 2 * num_oscillators  # (q_1..q_P, p_1..p_P)
        self.action_dim = num_oscillators

    def dynamics(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes phase space derivatives [dq/dt, dp/dt]."""
        P = self.num_oscillators
        q = state[..., :P]
        p = state[..., P:]
        u = action if action is not None else np.zeros_like(q)

        # dq/dt = p
        dq = p

        # Coupling forces between nearest neighbors
        q_left = np.pad(q[..., :-1], ((0, 0), (1, 0)), mode="constant")
        q_right = np.pad(q[..., 1:], ((0, 0), (0, 1)), mode="constant")
        coupling_forces = self.coupling_k * (q_left + q_right - 2 * q)

        # Non-linear restoring force + damping + control force
        dp = coupling_forces - self.alpha * (q ** 3) - self.damping * p + u

        return np.concatenate([dq, dp], axis=-1)

    def step_rk4(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Advances state by dt using RK4."""
        dt = self.dt
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + 0.5 * dt * k1, action)
        k3 = self.dynamics(state + 0.5 * dt * k2, action)
        k4 = self.dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def generate_trajectories(
        self,
        num_trajectories: int,
        num_steps: int,
        action_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates coupled oscillator trajectories."""
        rng = np.random.default_rng(seed)
        q0 = rng.uniform(-1.5, 1.5, size=(num_trajectories, self.num_oscillators))
        p0 = rng.uniform(-0.5, 0.5, size=(num_trajectories, self.num_oscillators))
        curr = np.concatenate([q0, p0], axis=-1)

        states = np.zeros((num_trajectories, num_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros((num_trajectories, num_steps, self.action_dim), dtype=np.float32)

        for t in range(num_steps):
            states[:, t] = curr
            act = action_fn(curr, t) if action_fn is not None else np.zeros((num_trajectories, self.action_dim), dtype=np.float32)
            actions[:, t] = act
            curr = self.step_rk4(curr, act)

        return states, actions
