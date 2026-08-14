"""
Multivariate Sensor Telemetry Benchmark with Controlled Nuisance Noise Channels.

Simulates industrial sensory environments where a subset of channels capture physical state
dynamics (temperature, pressure, vibration, rpm) while other channels contain high-entropy
stochastic nuisance noise (electrical hum, telemetry jitter, sensor drift).

Used to benchmark JEPA's noise-invariance vs. generative reconstruction baselines.
"""

from typing import Optional, Tuple
import numpy as np
from jepa_wm.data.dynamical_systems import Lorenz63System, CoupledOscillatorsSystem


class MultiSensorIndustrialBenchmark:
    """
    Synthesizes C-channel multivariate time-series from physical dynamics:
        x_t = [ g_sensor(s_t), nu_t ] + eta_t

    Attributes:
        num_physical_channels (int): Number of informative physical telemetry channels.
        num_nuisance_channels (int): Number of uninformative stochastic noise channels.
        noise_level (float): Measurement additive Gaussian noise sigma.
        pink_noise_weight (float): Relative amplitude of 1/f drift noise.
    """

    def __init__(
        self,
        system_type: str = "lorenz",
        num_physical_channels: int = 8,
        num_nuisance_channels: int = 12,
        noise_level: float = 0.05,
        pink_noise_weight: float = 0.5,
    ) -> None:
        self.num_physical_channels = num_physical_channels
        self.num_nuisance_channels = num_nuisance_channels
        self.total_channels = num_physical_channels + num_nuisance_channels
        self.noise_level = noise_level
        self.pink_noise_weight = pink_noise_weight
        self.system_type = system_type

        if system_type == "lorenz":
            self.system = Lorenz63System()
        elif system_type == "oscillators":
            self.system = CoupledOscillatorsSystem(num_oscillators=4)
        else:
            raise ValueError(f"Unknown system_type: {system_type}")

        self.state_dim = self.system.state_dim
        self.action_dim = self.system.action_dim

    def _sensor_observation_map(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Maps physical states s in R^(B x T x d_s) to C_phys sensor telemetry channels
        via non-linear calibration functions (mixing, polynomial terms, saturations).
        """
        B, T, d_s = states.shape
        C_phys = self.num_physical_channels

        # Fixed projection matrix
        W = rng.normal(0.0, 1.0 / np.sqrt(d_s), size=(d_s, C_phys))

        # Linear mixture
        linear = np.matmul(states, W)  # (B, T, C_phys)

        # Non-linear sensor responses (e.g. thermocouple exponential/tanh response)
        nonlinear = np.tanh(0.1 * linear) + 0.05 * np.sin(2.0 * linear)

        # Sensor normalization
        mean = np.mean(nonlinear, axis=(0, 1), keepdims=True)
        std = np.std(nonlinear, axis=(0, 1), keepdims=True) + 1e-6
        norm_sensor = (nonlinear - mean) / std
        return norm_sensor

    def _generate_nuisance_noise(self, B: int, T: int, rng: np.random.Generator) -> np.ndarray:
        """Generates C_nuisance channels containing white + 1/f pink noise + drifts."""
        C_nuis = self.num_nuisance_channels
        if C_nuis == 0:
            return np.empty((B, T, 0), dtype=np.float32)

        # White Gaussian noise
        white = rng.normal(0.0, 1.0, size=(B, T, C_nuis))

        # Random walk / colored drift
        random_walk = np.cumsum(rng.normal(0.0, 0.05, size=(B, T, C_nuis)), axis=1)

        # High frequency sensor telemetry spikes (Poisson-distributed bursts)
        spikes = (rng.uniform(0, 1, size=(B, T, C_nuis)) > 0.98).astype(np.float32) * rng.normal(0, 2.0, size=(B, T, C_nuis))

        nuisance = white + self.pink_noise_weight * random_walk + spikes

        # Standardize
        nuisance = (nuisance - nuisance.mean(axis=(0, 1), keepdims=True)) / (nuisance.std(axis=(0, 1), keepdims=True) + 1e-6)
        return nuisance.astype(np.float32)

    def generate_dataset(
        self,
        num_trajectories: int,
        num_steps: int,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates full multi-sensor benchmark trajectories.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - observations: Shape (B, T, total_channels)
                - actions: Shape (B, T, action_dim)
                - ground_truth_states: Shape (B, T, state_dim)
        """
        rng = np.random.default_rng(seed)

        # Random smooth control policy (sinusoidal action inputs)
        def action_policy(s: np.ndarray, t: int) -> np.ndarray:
            freq = 0.05
            return 0.5 * np.sin(freq * t + np.arange(self.action_dim))

        states, actions = self.system.generate_trajectories(
            num_trajectories=num_trajectories,
            num_steps=num_steps,
            action_fn=action_policy,
            seed=seed,
        )

        phys_obs = self._sensor_observation_map(states, rng)
        nuis_obs = self._generate_nuisance_noise(num_trajectories, num_steps, rng)

        # Combine channels: [physical_channels, nuisance_channels]
        if self.num_nuisance_channels > 0:
            observations = np.concatenate([phys_obs, nuis_obs], axis=-1)
        else:
            observations = phys_obs

        # Add measurement noise to all sensors
        observations += rng.normal(0.0, self.noise_level, size=observations.shape).astype(np.float32)

        return observations.astype(np.float32), actions.astype(np.float32), states.astype(np.float32)
