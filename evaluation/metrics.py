"""Offline evaluation metrics for Gaussian category priors."""
from __future__ import annotations

from typing import Iterable

import torch


def _pairwise_min_distances(
    source: torch.Tensor, target: torch.Tensor, chunk_size: int = 2048
) -> torch.Tensor:
    """Return nearest neighbour distances from source to target."""

    distances = []
    for start in range(0, source.size(0), chunk_size):
        end = min(start + chunk_size, source.size(0))
        chunk = source[start:end]
        dist = torch.cdist(chunk.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
        distances.append(dist.min(dim=1).values)
    return torch.cat(distances, dim=0)


def chamfer_L2(
    template_points: torch.Tensor,
    reference_points: torch.Tensor,
    chunk_size: int = 2048,
) -> float:
    """Symmetric L2 Chamfer distance between two point sets."""

    d_template = _pairwise_min_distances(template_points, reference_points, chunk_size)
    d_reference = _pairwise_min_distances(reference_points, template_points, chunk_size)
    distance = (d_template.square().mean() + d_reference.square().mean()).item()
    return float(distance)


def fscore_at_tau(
    template_points: torch.Tensor,
    reference_points: torch.Tensor,
    tau: float,
    chunk_size: int = 2048,
) -> float:
    """F-score at threshold tau for two point clouds."""

    d_template = _pairwise_min_distances(template_points, reference_points, chunk_size)
    d_reference = _pairwise_min_distances(reference_points, template_points, chunk_size)

    precision = (d_template <= tau).float().mean()
    recall = (d_reference <= tau).float().mean()
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return float((2 * precision * recall / (denom + 1e-6)).item())


def sample_gaussian_template(
    centers: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
    num_samples: int = 5000,
) -> torch.Tensor:
    """Sample points from a diagonal Gaussian template."""

    weights = alpha.squeeze(-1).clamp(min=1e-6)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, num_samples, replacement=True)
    noise = torch.randn(num_samples, 3, device=centers.device)
    samples = centers[indices] + noise * scales[indices]
    return samples


def volume_iou(pred_occ: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU on occupancy predictions."""

    pred_binary = (pred_occ >= threshold)
    label_binary = (labels >= 0.5)
    intersection = (pred_binary & label_binary).float().sum(dim=-1)
    union = (pred_binary | label_binary).float().sum(dim=-1).clamp_min(1.0)
    iou = (intersection / union).mean().item()
    return float(iou)


def aggregate(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item())


__all__ = [
    "aggregate",
    "chamfer_L2",
    "fscore_at_tau",
    "sample_gaussian_template",
    "volume_iou",
]
