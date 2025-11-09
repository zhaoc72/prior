"""Initialise Gaussian templates from canonical point clouds."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.geom import init_gaussians_from_pointcloud

def _process_single(
    npz_path: Path,
    out_dir: Path,
    K: int,
    knn: int,
    seed: int | None,
    nn_jobs: int | None,
) -> None:
    data = np.load(npz_path)
    points = data["xyz"]
    centers, scales, rotations, alpha = init_gaussians_from_pointcloud(
        points, K=K, knn=knn, seed=seed, n_jobs=nn_jobs
    )
    np.savez(
        out_dir / npz_path.name,
        centers=centers,
        scales=scales,
        rotations=rotations,
        alpha=alpha,
    )


def process_directory(
    points_dir: Path,
    out_dir: Path,
    K: int,
    knn: int,
    seed: int | None,
    nn_jobs: int | None,
    workers: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    point_files = sorted(points_dir.glob("*.npz"))
    if not point_files:
        return

    worker_count = 1 if not workers or workers <= 1 else workers
    if worker_count <= 1:
        for npz_path in tqdm(point_files, desc="gaussian_init"):
            _process_single(npz_path, out_dir, K, knn, seed, nn_jobs)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _process_single, npz_path, out_dir, K, knn, seed, nn_jobs
            )
            for npz_path in point_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="gaussian_init"):
            future.result()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--nn_jobs",
        type=int,
        default=None,
        help="Thread count passed to sklearn NearestNeighbors (None keeps library default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel worker processes (set 1 to disable)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_directory(
        args.points_dir,
        args.out,
        args.K,
        args.knn,
        args.seed,
        args.nn_jobs,
        args.workers,
    )
