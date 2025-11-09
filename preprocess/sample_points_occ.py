"""Sample occupancy and surface points from canonical meshes."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# Limit the number of BLAS/OpenMP threads each worker can spawn when the caller
# has not provided explicit values. This reduces the likelihood of native
# library crashes that manifest as ``BrokenProcessPool`` when many workers are
# active simultaneously.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

def _process_single(
    mesh_path: Path,
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
) -> None:
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


def _run_with_executor(
    executor_type,
    worker_count: int,
    meshes: list[Path],
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
    *,
    desc: str,
    mp_context: mp.context.BaseContext | None = None,
) -> None:
    """Execute sampling tasks with the provided executor implementation."""

    executor_kwargs = {"max_workers": worker_count}
    if mp_context is not None and executor_type is ProcessPoolExecutor:
        executor_kwargs["mp_context"] = mp_context

    with executor_type(**executor_kwargs) as executor:
        futures = {
            executor.submit(
                _process_single,
                mesh_path,
                mesh_dir,
                out_dir,
                n_surface,
                n_uniform,
            ): mesh_path
            for mesh_path in meshes
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            mesh_path = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - surfaces errors are rare
                raise RuntimeError(f"Failed to process {mesh_path}") from exc


def process_directory(
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
    workers: int | None = None,
    skip_existing: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes = sorted(mesh_dir.rglob("*.obj")) + sorted(mesh_dir.rglob("*.ply"))
    if not meshes:
        return

    if skip_existing:
        filtered: list[Path] = []
        for mesh_path in meshes:
            rel_path = mesh_path.relative_to(mesh_dir)
            out_path = (out_dir / rel_path).with_suffix(".npz")
            if out_path.exists():
                continue
            filtered.append(mesh_path)
        meshes = filtered
        if not meshes:
            return

    worker_count = 1 if not workers or workers <= 1 else workers
    if worker_count <= 1:
        for mesh_path in tqdm(meshes, desc="occ_sampling"):
            _process_single(mesh_path, mesh_dir, out_dir, n_surface, n_uniform)
        return

    attempt_workers = worker_count
    last_exc: BrokenProcessPool | None = None
    ctx = mp.get_context("spawn")
    while attempt_workers > 1:
        try:
            _run_with_executor(
                ProcessPoolExecutor,
                attempt_workers,
                meshes,
                mesh_dir,
                out_dir,
                n_surface,
                n_uniform,
                desc="occ_sampling",
                mp_context=ctx,
            )
            return
        except BrokenProcessPool as exc:
            last_exc = exc
            print(
                "Process pool crashed (likely due to native library fork-safety). "
                f"Retrying with {max(1, attempt_workers // 2)} workers instead of {attempt_workers}."
            )
            attempt_workers = max(1, attempt_workers // 2)

    if last_exc is not None and worker_count > 1:
        print(
            "Process-based parallelism repeatedly failed. Switching to a thread "
            f"pool with {worker_count} workers. Last error: {last_exc}"
        )
        try:
            _run_with_executor(
                ThreadPoolExecutor,
                worker_count,
                meshes,
                mesh_dir,
                out_dir,
                n_surface,
                n_uniform,
                desc="occ_sampling-threads",
            )
            return
        except Exception as thread_exc:  # pragma: no cover - rare failure path
            print(
                "Thread pool execution also failed; falling back to sequential "
                f"processing. Error: {thread_exc}"
            )

    for mesh_path in tqdm(meshes, desc="occ_sampling-sequential"):
        _process_single(mesh_path, mesh_dir, out_dir, n_surface, n_uniform)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_surf", type=int, default=40000)
    parser.add_argument("--n_uniform", type=int, default=60000)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes (set 1 to disable)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute outputs even if the corresponding .npz file is already present.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_directory(
        args.mesh_dir,
        args.out,
        args.n_surf,
        args.n_uniform,
        args.workers,
        skip_existing=not args.no_skip_existing,
    )
