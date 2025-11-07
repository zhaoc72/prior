"""Sample occupancy and surface points from canonical meshes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

def sample_points(
    mesh: trimesh.Trimesh, n_surface: int, n_uniform: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface, _ = trimesh.sample.sample_surface(mesh, n_surface)
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    uniform = np.random.uniform(bounds_min, bounds_max, size=(n_uniform, 3))
    inside = mesh.contains(uniform)
    points = np.concatenate([surface, uniform], axis=0)
    labels = np.concatenate([np.ones(len(surface)), inside.astype(np.float32)], axis=0)
    return (
        points.astype(np.float32),
        labels.astype(np.float32),
        surface.astype(np.float32),
        uniform.astype(np.float32),
    )

def process_directory(mesh_dir: Path, out_dir: Path, n_surface: int, n_uniform: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = sorted(mesh_dir.rglob("*.obj")) + sorted(mesh_dir.rglob("*.ply"))
    for mesh_path in tqdm(meshes, desc="occ_sampling"):
        mesh = trimesh.load(mesh_path, process=False)
        pts, lbl, surf, uniform = sample_points(mesh, n_surface, n_uniform)
        rel_path = mesh_path.relative_to(mesh_dir)
        out_path = (out_dir / rel_path).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            xyz=pts,
            lbl=lbl,
            surface=surf,
            uniform=uniform,
            n_surface=int(len(surf)),
            n_uniform=int(len(uniform)),
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_surf", type=int, default=40000)
    parser.add_argument("--n_uniform", type=int, default=60000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_directory(args.mesh_dir, args.out, args.n_surf, args.n_uniform)
