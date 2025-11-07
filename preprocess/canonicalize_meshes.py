"""Canonicalize meshes to a shared coordinate frame."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.geom import canonicalize

def canonicalize_directory(source: Path, destination: Path, unit: str = "sphere") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    meshes = sorted(source.rglob("*.obj")) + sorted(source.rglob("*.ply"))
    for mesh_path in tqdm(meshes, desc="canonicalize"):
        mesh = trimesh.load(mesh_path, process=False)
        # Some files load as a Scene (multiple geometries). Convert to a single Trimesh.
        if isinstance(mesh, trimesh.Scene):
            try:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            except Exception:
                dumped = mesh.dump()
                if isinstance(dumped, (list, tuple)) and len(dumped) > 0:
                    mesh = trimesh.util.concatenate(
                        [m for m in dumped if isinstance(m, trimesh.Trimesh)]
                    )
                else:
                    raise
        canon_mesh, transform = canonicalize(mesh, unit=unit)
        rel_path = mesh_path.relative_to(source)
        out_path = destination / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canon_mesh.export(out_path)
        meta_path = out_path.with_name(f"{out_path.stem}_transform.npz")
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
