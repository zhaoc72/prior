"""Core Gaussian representations and projection utilities."""
from __future__ import annotations

import torch

def se3_project(points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Project 3D points from canonical space to camera coordinates."""

    if points.ndim != 3:
        raise ValueError("points should be of shape [B, N, 3]")
    if intrinsics.ndim != 3 or extrinsics.ndim != 3:
        raise ValueError("Camera parameters must be batched")

    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    cam = torch.einsum("bij,bnj->bni", rotation, points) + translation[:, None, :]
    x = cam[..., 0] / (cam[..., 2] + 1e-6)
    y = cam[..., 1] / (cam[..., 2] + 1e-6)
    u = intrinsics[:, 0, 0][:, None] * x + intrinsics[:, 0, 2][:, None]
    v = intrinsics[:, 1, 1][:, None] * y + intrinsics[:, 1, 2][:, None]
    return torch.stack([u, v, cam[..., 2]], dim=-1)

def gaussian_2d_params(
    centers_3d: torch.Tensor,
    scales_3d: torch.Tensor,
    rotations_3d: torch.Tensor | None,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate 2D Gaussian parameters from anisotropic 3D Gaussians."""

    uvz = se3_project(centers_3d, intrinsics, extrinsics)
    depth = uvz[..., 2].clamp(min=1e-3)
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]

    if rotations_3d is not None:
        # full projection via linearisation of rotation and scaling
        R = rotations_3d
        S = scales_3d
        sigma_x = fx * (S[..., 0] / depth)
        sigma_y = fy * (S[..., 1] / depth)
    else:
        sigma_x = fx * (scales_3d[..., 0] / depth)
        sigma_y = fy * (scales_3d[..., 1] / depth)

    sigmas = torch.stack([sigma_x, sigma_y], dim=-1)
    return uvz[..., :2], sigmas, depth

__all__ = ["se3_project", "gaussian_2d_params"]
