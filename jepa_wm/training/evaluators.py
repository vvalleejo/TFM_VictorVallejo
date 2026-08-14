"""
Evaluation Routines and Benchmark Metrics for World Models.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jepa_wm.diagnostics.identifiability import (
    LinearIdentifiabilityProbe,
    OrthogonalProcrustesAlignment,
    CanonicalCorrelationMetrics,
)
from jepa_wm.diagnostics.spectral import compute_spectral_diagnostics


@torch.no_grad()
def evaluate_world_model_benchmarks(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Computes complete benchmark diagnostics across validation trajectory set:
    - Multi-step predictive MSE
    - Effective rank and condition number
    - Linear probe R2 and MSE on physical states
    - Orthogonal Procrustes discrepancy
    - Canonical correlation (mean and top-1)
    """
    model.eval()
    model.to(device)

    all_z = []
    all_s = []
    pred_errors_by_horizon = []

    for batch in val_loader:
        obs_ctx = batch["obs_ctx"].to(device)
        act_fut = batch["act_fut"].to(device)
        obs_fut = batch["obs_fut"].to(device)

        z_pred_seq, z_target_seq, loss, metrics = model(obs_ctx, act_fut, obs_fut)
        z_ctx = model.encoder(obs_ctx)

        all_z.append(z_ctx.cpu().numpy())
        if "state_ctx" in batch:
            all_s.append(batch["state_ctx"].numpy())

        # Step-by-step MSE error: (K,)
        step_err = (z_pred_seq - z_target_seq).pow(2).mean(dim=(0, 2)).cpu().numpy()
        pred_errors_by_horizon.append(step_err)

    z_mat = np.concatenate(all_z, axis=0)
    spectral = compute_spectral_diagnostics(z_mat)

    results = {
        "effective_rank": spectral["effective_rank"],
        "spectral_entropy": spectral["spectral_entropy"],
        "condition_number": spectral["condition_number"],
    }

    if len(pred_errors_by_horizon) > 0:
        mean_step_err = np.mean(pred_errors_by_horizon, axis=0)
        results["pred_mse_step1"] = float(mean_step_err[0])
        results["pred_mse_stepK"] = float(mean_step_err[-1])
        results["pred_mse_mean"] = float(np.mean(mean_step_err))

    if len(all_s) > 0:
        s_mat = np.concatenate(all_s, axis=0)
        n_train = int(0.7 * len(z_mat))

        # Linear Probe
        probe = LinearIdentifiabilityProbe(alpha=1e-3).fit(z_mat[:n_train], s_mat[:n_train])
        probe_res = probe.evaluate(z_mat[n_train:], s_mat[n_train:])
        results["identifiability_r2"] = probe_res["r2_score"]
        results["identifiability_mse"] = probe_res["mse"]

        # Procrustes Alignment
        _, _, proc_metrics = OrthogonalProcrustesAlignment.align(z_mat, s_mat)
        results["procrustes_discrepancy"] = proc_metrics["procrustes_discrepancy"]
        results["procrustes_identifiability"] = proc_metrics["identifiability_score"]

        # CCA
        cca = CanonicalCorrelationMetrics.compute(z_mat, s_mat)
        results["cca_mean"] = cca["mean_cca"]
        results["cca_top1"] = cca["top1_cca"]

    return results
