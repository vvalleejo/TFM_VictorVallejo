"""
InfoNCE Contrastive Loss Baseline for Dynamics Modeling.

Reference:
Oord, Li, Vinyals (2018): "Representation Learning with Contrastive Predictive Coding"
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for predictive representations.

    Treats (z_pred_i, z_target_i) as positive pairs and other elements in the batch as negative pairs.

    Attributes:
        temperature (float): Softmax temperature scale parameter tau.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z_pred (torch.Tensor): Predicted latents of shape (N, d).
            z_target (torch.Tensor): Target latents of shape (N, d).

        Returns:
            Tuple[torch.Tensor, dict]: Contrastive loss and dictionary.
        """
        if z_pred.ndim > 2:
            z_pred = z_pred.reshape(-1, z_pred.size(-1))
        if z_target.ndim > 2:
            z_target = z_target.reshape(-1, z_target.size(-1))

        # Normalize representations to unit sphere
        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        z_target_norm = F.normalize(z_target, p=2, dim=-1)

        # Similarity matrix: (N, N)
        similarity = torch.matmul(z_pred_norm, z_target_norm.T) / self.temperature

        # Target class indices: positive pairs are along the diagonal
        labels = torch.arange(z_pred.size(0), device=z_pred.device)

        loss = F.cross_entropy(similarity, labels)

        # Accuracy of positive retrieval
        with torch.no_grad():
            preds = similarity.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        return loss, {"loss": loss.item(), "top1_acc": acc}
