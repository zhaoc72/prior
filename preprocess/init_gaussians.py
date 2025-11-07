"""Initialise Gaussian templates from canonical point clouds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.geom import init_gaussians_from_pointcloud

def process_directory(points_dir: Path, out_dir: Path, K: int, knn: int, seed: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    point_files = sorted(points_dir.glob("*.npz"))
    for npz_path in tqdm(point_files, desc="gaussian_init"):
        data = np.load(npz_path)
        points = data["xyz"]
        centers, scales, rotations, alpha = init_gaussians_from_pointcloud(points, K=K, knn=knn, seed=seed)
        np.savez(
            out_dir / npz_path.name,
            centers=centers,
            scales=scales,
            rotations=rotations,
            alpha=alpha,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_directory(args.points_dir, args.out, args.K, args.knn, args.seed)
