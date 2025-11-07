"""3D occupancy losses for Gaussian templates."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def gaussian_occupancy(
    samples: torch.Tensor, centers: torch.Tensor, inv_scales: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Compute occupancy probability induced by a set of Gaussian kernels."""

    diff = samples[:, :, None, :] - centers[:, None, :, :]
    quad = (diff**2 * inv_scales[:, None, :, :]).sum(dim=-1)
    gaussian = torch.exp(-0.5 * quad) * alpha.squeeze(-1)
    occupancy = 1.0 - torch.prod(1.0 - gaussian, dim=-1)
    return occupancy

def occupancy_loss(
    samples: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    inv_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Binary cross entropy occupancy loss."""

    preds = gaussian_occupancy(samples, centers, inv_scales, alpha)
    preds = preds.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(preds, labels)

__all__ = ["gaussian_occupancy", "occupancy_loss"]
