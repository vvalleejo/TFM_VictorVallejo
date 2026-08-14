"""
Training Engine and Optimization Loops for Latent JEPA World Models.
"""

from typing import Callable, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_wm.diagnostics.identifiability import LinearIdentifiabilityProbe
from jepa_wm.diagnostics.spectral import compute_effective_rank


class JEPATrainer:
    """
    Trainer for Latent JEPA World Models.

    Attributes:
        model (nn.Module): LatentJEPAWorldModel.
        train_loader (DataLoader): DataLoader for training batches.
        val_loader (DataLoader): DataLoader for validation batches.
        lr (float): Peak learning rate.
        weight_decay (float): L2 weight decay for AdamW.
        device (str): Device to run training on ('cuda' or 'cpu').
        grad_clip (float): Maximum gradient norm.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        grad_clip: float = 1.0,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_pred": [],
            "train_loss_sigreg": [],
            "val_loss": [],
            "val_loss_pred": [],
            "val_loss_sigreg": [],
            "val_erank": [],
            "val_identifiability_r2": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        epoch_losses = []
        epoch_pred_losses = []
        epoch_sigreg_losses = []

        for batch in self.train_loader:
            obs_ctx = batch["obs_ctx"].to(self.device)
            act_fut = batch["act_fut"].to(self.device)
            obs_fut = batch["obs_fut"].to(self.device)

            self.optimizer.zero_grad()

            z_pred_seq, z_target_seq, loss, metrics = self.model(obs_ctx, act_fut, obs_fut)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update EMA target encoder if active
            if hasattr(self.model, "update_ema_target"):
                self.model.update_ema_target()

            epoch_losses.append(loss.item())
            epoch_pred_losses.append(metrics.get("loss_pred", 0.0))
            epoch_sigreg_losses.append(metrics.get("loss_sigreg", 0.0))

        return {
            "loss": float(np.mean(epoch_losses)),
            "loss_pred": float(np.mean(epoch_pred_losses)),
            "loss_sigreg": float(np.mean(epoch_sigreg_losses)),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Runs full validation loop including mathematical diagnostics."""
        self.model.eval()
        val_losses = []
        val_pred_losses = []
        val_sigreg_losses = []

        all_latents = []
        all_states = []

        for batch in self.val_loader:
            obs_ctx = batch["obs_ctx"].to(self.device)
            act_fut = batch["act_fut"].to(self.device)
            obs_fut = batch["obs_fut"].to(self.device)

            z_pred_seq, z_target_seq, loss, metrics = self.model(obs_ctx, act_fut, obs_fut)
            z_ctx = self.model.encoder(obs_ctx)

            val_losses.append(loss.item())
            val_pred_losses.append(metrics.get("loss_pred", 0.0))
            val_sigreg_losses.append(metrics.get("loss_sigreg", 0.0))

            all_latents.append(z_ctx.cpu().numpy())
            if "state_ctx" in batch:
                all_states.append(batch["state_ctx"].numpy())

        # Concatenate evaluation latents
        all_z = np.concatenate(all_latents, axis=0)
        erank = compute_effective_rank(all_z)

        eval_metrics = {
            "val_loss": float(np.mean(val_losses)),
            "val_loss_pred": float(np.mean(val_pred_losses)),
            "val_loss_sigreg": float(np.mean(val_sigreg_losses)),
            "val_erank": float(erank),
        }

        # Compute linear identifiability R2 if ground-truth states exist
        if len(all_states) > 0:
            all_s = np.concatenate(all_states, axis=0)
            n_split = int(0.7 * len(all_z))
            probe = LinearIdentifiabilityProbe(alpha=1e-3).fit(all_z[:n_split], all_s[:n_split])
            probe_results = probe.evaluate(all_z[n_split:], all_s[n_split:])
            eval_metrics["val_identifiability_r2"] = probe_results["r2_score"]
        else:
            eval_metrics["val_identifiability_r2"] = 0.0

        return eval_metrics

    def fit(
        self,
        epochs: int,
        verbose: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Runs full training and validation schedule.

        Args:
            epochs (int): Number of training epochs.
            verbose (bool): Whether to display progress bar.
            save_path (str, optional): Path to save best model checkpoint.

        Returns:
            Dict[str, List[float]]: History dictionary.
        """
        best_val_loss = float("inf")
        pbar = tqdm(range(epochs), disable=not verbose, desc="Training LeJEPA-WM")

        for epoch in pbar:
            train_m = self.train_epoch()
            val_m = self.evaluate()

            self.history["train_loss"].append(train_m["loss"])
            self.history["train_loss_pred"].append(train_m["loss_pred"])
            self.history["train_loss_sigreg"].append(train_m["loss_sigreg"])

            self.history["val_loss"].append(val_m["val_loss"])
            self.history["val_loss_pred"].append(val_m["val_loss_pred"])
            self.history["val_loss_sigreg"].append(val_m["val_loss_sigreg"])
            self.history["val_erank"].append(val_m["val_erank"])
            self.history["val_identifiability_r2"].append(val_m["val_identifiability_r2"])

            pbar.set_postfix(
                train_loss=f"{train_m['loss']:.4f}",
                val_loss=f"{val_m['val_loss']:.4f}",
                erank=f"{val_m['val_erank']:.2f}",
                r2=f"{val_m['val_identifiability_r2']:.3f}",
            )

            if val_m["val_loss"] < best_val_loss and save_path is not None:
                best_val_loss = val_m["val_loss"]
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

        return self.history
