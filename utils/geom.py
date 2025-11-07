"""Geometry utilities for canonicalization and Gaussian initialization."""
from __future__ import annotations

import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors


def canonicalize(mesh: trimesh.Trimesh, unit: str = "sphere") -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Center and normalize a mesh into canonical space.

    Args:
        mesh: Input mesh in arbitrary object space.
        unit: Either "sphere" (default) or "cube" normalisation.

    Returns:
        A tuple of the canonicalized mesh and a dictionary containing the
        translation and scale applied during normalization.
    """

    vertices = mesh.vertices
    center = (vertices.max(0) + vertices.min(0)) / 2.0
    centered = vertices - center

    if unit == "sphere":
        scale = np.linalg.norm(centered, axis=1).max() + 1e-8
    elif unit == "cube":
        scale = (centered.max(0) - centered.min(0)).max() + 1e-8
    else:
        raise ValueError(f"Unsupported unit '{unit}'")

    normalized = centered / scale
    canon_mesh = trimesh.Trimesh(normalized, mesh.faces, process=False)
    transform = {"translation": -center.astype(np.float32), "scale": float(scale)}
    return canon_mesh, transform


def fps(points: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    """Farthest point sampling for dense point clouds."""

    if len(points) < k:
        raise ValueError("Point cloud smaller than requested sample count")

    rng = np.random.default_rng(seed)
    selected = [rng.integers(len(points))]
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(1, k):
        diff = points - points[selected[-1]]
        dist = np.linalg.norm(diff, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))
    return np.asarray(selected, dtype=np.int64)


def pca_cov(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute principal components and eigenvalues of a local neighbourhood."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points should be of shape [N, 3]")
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    return vals.astype(np.float32), vecs.astype(np.float32)


def init_gaussians_from_pointcloud(
    points: np.ndarray, K: int = 2048, knn: int = 64, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialise Gaussian parameters from a canonical point cloud."""

    if points.shape[0] < K:
        raise ValueError("Point cloud must contain at least K points")

    idx = fps(points, K, seed=seed)
    centers = points[idx]
    nbrs = NearestNeighbors(n_neighbors=knn).fit(points)
    _, indices = nbrs.kneighbors(centers)

    scales = []
    rotations = []
    for neigh in indices:
        vals, vecs = pca_cov(points[neigh])
        scales.append(np.sqrt(vals))
        rotations.append(vecs)

    scales = np.stack(scales).astype(np.float32)
    rotations = np.stack(rotations).astype(np.float32)
    alphas = np.full((K, 1), 0.9, dtype=np.float32)
    return centers.astype(np.float32), scales, rotations, alphas


__all__ = [
    "canonicalize",
    "fps",
    "pca_cov",
    "init_gaussians_from_pointcloud",
]
