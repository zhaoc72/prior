"""Canonicalize meshes to a shared coordinate frame."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from utils.geom import canonicalize

def canonicalize_directory(source: Path, destination: Path, unit: str = "sphere") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    meshes = sorted(source.glob("*.obj")) + sorted(source.glob("*.ply"))
    for mesh_path in tqdm(meshes, desc="canonicalize"):
        mesh = trimesh.load(mesh_path, process=False)
        canon_mesh, transform = canonicalize(mesh, unit=unit)
        out_path = destination / mesh_path.name
        canon_mesh.export(out_path)
        meta_path = destination / f"{mesh_path.stem}_transform.npz"
        np.savez(
            meta_path,
            translation=transform["translation"],
            scale=transform["scale"],
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Directory with raw meshes")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory")
    parser.add_argument("--unit", type=str, default="sphere", choices=["sphere", "cube"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    canonicalize_directory(args.src, args.dst, unit=args.unit)
