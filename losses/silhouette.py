"""Silhouette losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def silhouette_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy on silhouettes with numerical stability."""

    pred = pred.clamp(1e-6, 1 - 1e-6)
    target = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred, target)

__all__ = ["silhouette_loss"]
