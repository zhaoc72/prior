"""Regularization terms for Gaussian templates."""
from __future__ import annotations

import torch

def scale_regularizer(scales: torch.Tensor, minimum: float = 0.01, maximum: float = 0.5) -> torch.Tensor:
    """Encourage scale values to remain within a bounded interval."""

    lower = torch.relu(minimum - scales)
    upper = torch.relu(scales - maximum)
    return (lower + upper).mean()

def repulsion_regularizer(centers: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Discourage Gaussian centers from collapsing to the same location."""

    diff = centers[:, :, None, :] - centers[:, None, :, :]
    dist2 = (diff**2).sum(dim=-1)
    mask = 1.0 - torch.eye(centers.size(1), device=centers.device)[None]
    penalties = torch.exp(-dist2 / (sigma**2 + 1e-8)) * mask
    return penalties.mean()

def volume_regularizer(scales: torch.Tensor) -> torch.Tensor:
    """Penalize overly voluminous Gaussians."""

    volume = (scales.prod(dim=-1))
    return volume.mean()

__all__ = ["scale_regularizer", "repulsion_regularizer", "volume_regularizer"]
