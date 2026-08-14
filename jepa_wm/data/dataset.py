"""
PyTorch Datasets and DataLoaders for Multivariate Trajectory Sequences.

Provides windowed sampling of historical context sequences and future rollout prediction targets.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """
    Sliding window dataset over multivariate trajectory sequences.

    Given trajectories of shape (num_trajectories, T_total, channels), extracts:
    - Context observations: x_{t - T_ctx + 1 : t} in R^(T_ctx x channels)
    - Future target observations: x_{t + 1 : t + K} in R^(K x channels)
    - Future actions: a_{t : t + K - 1} in R^(K x action_dim)
    - Ground-truth latent states (for diagnostics): s_t in R^(state_dim) and s_{t+1:t+K} in R^(K x state_dim)

    Attributes:
        context_len (int): Length of historical context window T_ctx.
        horizon (int): Multi-step rollout prediction horizon K.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        states: Optional[np.ndarray] = None,
        context_len: int = 10,
        horizon: int = 5,
        stride: int = 1,
    ) -> None:
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.states = (
            torch.as_tensor(states, dtype=torch.float32)
            if states is not None
            else None
        )
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride

        num_trajectories, T_total, _ = self.observations.shape
        window_size = context_len + horizon
        self.valid_steps_per_traj = max(0, (T_total - window_size) // stride + 1)
        self.total_samples = num_trajectories * self.valid_steps_per_traj

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.valid_steps_per_traj
        step_idx = (idx % self.valid_steps_per_traj) * self.stride

        # Context window: [step_idx, step_idx + context_len)
        ctx_start = step_idx
        ctx_end = step_idx + self.context_len

        # Future rollout window: [ctx_end, ctx_end + horizon)
        fut_start = ctx_end
        fut_end = ctx_end + self.horizon

        # Actions: conditioned from [ctx_end - 1, ctx_end + horizon - 1)
        # i.e., action taken at step t to predict t+1, etc.
        act_start = ctx_end - 1
        act_end = ctx_end + self.horizon - 1

        obs_ctx = self.observations[traj_idx, ctx_start:ctx_end]  # (T_ctx, C)
        obs_fut = self.observations[traj_idx, fut_start:fut_end]  # (K, C)
        act_fut = self.actions[traj_idx, act_start:act_end]      # (K, d_a)

        item = {
            "obs_ctx": obs_ctx,
            "obs_fut": obs_fut,
            "act_fut": act_fut,
        }

        if self.states is not None:
            item["state_ctx"] = self.states[traj_idx, ctx_end - 1]     # (d_s,) state at time t
            item["state_fut"] = self.states[traj_idx, fut_start:fut_end] # (K, d_s) future states

        return item


def create_dataloaders(
    observations: np.ndarray,
    actions: np.ndarray,
    states: Optional[np.ndarray] = None,
    context_len: int = 10,
    horizon: int = 5,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Splits trajectory data into train and validation sets and returns DataLoaders."""
    num_traj = observations.shape[0]
    n_train = int(num_traj * train_ratio)

    obs_train, obs_val = observations[:n_train], observations[n_train:]
    act_train, act_val = actions[:n_train], actions[n_train:]
    states_train = states[:n_train] if states is not None else None
    states_val = states[n_train:] if states is not None else None

    train_ds = TrajectoryDataset(obs_train, act_train, states_train, context_len, horizon)
    val_ds = TrajectoryDataset(obs_val, act_val, states_val, context_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
