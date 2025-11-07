"""Differentiable silhouette rendering for Gaussian templates."""
from __future__ import annotations

import torch

def render_silhouette(
    centers_2d: torch.Tensor,
    scales_2d: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    alpha: torch.Tensor,
    tile: int = 32,
) -> torch.Tensor:
    """Rasterise Gaussian blobs into a probabilistic silhouette image."""

    device = centers_2d.device
    batch, num_gaussians, _ = centers_2d.shape
    image = torch.zeros(batch, 1, height, width, device=device)

    for b in range(batch):
        for start in range(0, num_gaussians, tile):
            end = min(start + tile, num_gaussians)
            mu = centers_2d[b, start:end]
            sigma = scales_2d[b, start:end].clamp(min=0.5)
            a = alpha[b, start:end]

            x0 = (mu[:, 0] - 4.0 * sigma[:, 0]).floor().clamp(0, width - 1)
            x1 = (mu[:, 0] + 4.0 * sigma[:, 0]).ceil().clamp(0, width - 1)
            y0 = (mu[:, 1] - 4.0 * sigma[:, 1]).floor().clamp(0, height - 1)
            y1 = (mu[:, 1] + 4.0 * sigma[:, 1]).ceil().clamp(0, height - 1)

            for i in range(mu.size(0)):
                xs = torch.arange(x0[i], x1[i] + 1, device=device)
                ys = torch.arange(y0[i], y1[i] + 1, device=device)
                grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
                dx = (grid_x - mu[i, 0]) / (sigma[i, 0] + 1e-6)
                dy = (grid_y - mu[i, 1]) / (sigma[i, 1] + 1e-6)
                gaussian = torch.exp(-0.5 * (dx**2 + dy**2)) * a[i]
                patch = 1.0 - (1.0 - gaussian).clamp(0.0, 1.0)
                image[b, 0, grid_y.long(), grid_x.long()] = 1.0 - (
                    1.0 - image[b, 0, grid_y.long(), grid_x.long()]
                ) * (1.0 - patch)
    return image

__all__ = ["render_silhouette"]
