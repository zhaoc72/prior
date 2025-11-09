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

import gc
from functools import partial

import numpy as np
import trimesh
from numpy.random import default_rng
from tqdm import tqdm


_RNG = default_rng()

def _as_float32(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as float32 without copying when already the right dtype."""

    if array.dtype == np.float32:
        return array
    return array.astype(np.float32, copy=False)


def sample_points(
    mesh: trimesh.Trimesh, n_surface: int, n_uniform: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample surface and occupancy points with a low-memory footprint."""

    surface, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface = _as_float32(surface)

    bounds_min = _as_float32(mesh.bounds[0])
    bounds_max = _as_float32(mesh.bounds[1])
    uniform = _RNG.uniform(bounds_min, bounds_max, size=(n_uniform, 3), dtype=np.float32)
    inside = mesh.contains(uniform.astype(np.float64))

    total_points = n_surface + n_uniform
    points = np.empty((total_points, 3), dtype=np.float32)
    points[:n_surface] = surface
    points[n_surface:] = uniform

    labels = np.empty(total_points, dtype=np.float32)
    labels[:n_surface] = 1.0
    labels[n_surface:] = inside.astype(np.float32, copy=False)

    return points, labels, surface, uniform

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
    del mesh, pts, lbl, surf, uniform


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
            except BrokenProcessPool as exc:
                raise exc
            except Exception as exc:  # pragma: no cover - surfaces errors are rare
                raise RuntimeError(f"Failed to process {mesh_path}") from exc


def _run_sequential(
    meshes: list[Path],
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
) -> None:
    process_one = partial(
        _process_single,
        mesh_dir=mesh_dir,
        out_dir=out_dir,
        n_surface=n_surface,
        n_uniform=n_uniform,
    )
    for mesh_path in tqdm(meshes, desc="occ_sampling-sequential"):
        process_one(mesh_path)
        gc.collect()


def _run_thread_pool(
    worker_count: int,
    meshes: list[Path],
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
) -> None:
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


def _run_process_pool(
    worker_count: int,
    meshes: list[Path],
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
    *,
    fail_fast: bool,
) -> None:
    ctx = mp.get_context("spawn")
    attempt_workers = worker_count
    last_exc: BrokenProcessPool | None = None
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
            if fail_fast:
                raise
            last_exc = exc
            next_workers = max(1, attempt_workers // 2)
            print(
                "Process pool crashed (likely due to native library fork-safety). "
                f"Retrying with {next_workers} workers instead of {attempt_workers}."
            )
            attempt_workers = next_workers

    if last_exc is not None:
        raise last_exc


def process_directory(
    mesh_dir: Path,
    out_dir: Path,
    n_surface: int,
    n_uniform: int,
    workers: int | None = None,
    skip_existing: bool = True,
    executor: str = "auto",
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

    worker_count = 1 if not workers or workers <= 1 else min(workers, len(meshes))

    if worker_count <= 1:
        _run_sequential(meshes, mesh_dir, out_dir, n_surface, n_uniform)
        return

    exec_mode = executor.lower()
    if exec_mode not in {"auto", "process", "thread"}:
        raise ValueError(f"Unsupported executor mode: {executor}")

    if exec_mode == "thread":
        print(f"Using thread pool with {worker_count} workers for occupancy sampling.")
        _run_thread_pool(worker_count, meshes, mesh_dir, out_dir, n_surface, n_uniform)
        return

    if exec_mode == "process":
        try:
            print(f"Using process pool with {worker_count} workers for occupancy sampling.")
            _run_process_pool(
                worker_count,
                meshes,
                mesh_dir,
                out_dir,
                n_surface,
                n_uniform,
                fail_fast=False,
            )
            return
        except BrokenProcessPool as exc:
            fallback_workers = min(worker_count, len(meshes))
            if fallback_workers > 1:
                print(
                    "Process-based parallelism repeatedly failed. Switching to a "
                    f"thread pool with {fallback_workers} workers. Last error: {exc}"
                )
                _run_thread_pool(
                    fallback_workers, meshes, mesh_dir, out_dir, n_surface, n_uniform
                )
                return
            print(
                "Process-based parallelism repeatedly failed and no parallel "
                f"fallback is available. Running sequentially instead. Last error: {exc}"
            )
            _run_sequential(meshes, mesh_dir, out_dir, n_surface, n_uniform)
            return

    # Auto mode: try process pool once, fall back to threads, then sequential.
    try:
        print(
            f"Trying process pool with {worker_count} workers for occupancy sampling (auto executor)."
        )
        _run_process_pool(
            worker_count,
            meshes,
            mesh_dir,
            out_dir,
            n_surface,
            n_uniform,
            fail_fast=True,
        )
        return
    except BrokenProcessPool as exc:
        print(
            "Process pool crashed (likely due to native library fork-safety). "
            f"Switching to a thread pool with {worker_count} workers. Error: {exc}"
        )
        try:
            _run_thread_pool(worker_count, meshes, mesh_dir, out_dir, n_surface, n_uniform)
            return
        except Exception as thread_exc:  # pragma: no cover - rare failure path
            print(
                "Thread pool execution also failed; falling back to sequential "
                f"processing. Error: {thread_exc}"
            )
            _run_sequential(meshes, mesh_dir, out_dir, n_surface, n_uniform)

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
        help="Number of parallel workers (set 1 to disable parallelism)",
    )
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help="Parallel executor type: process (multiprocessing), thread, or auto fallback.",
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
        executor=args.executor,
    )
