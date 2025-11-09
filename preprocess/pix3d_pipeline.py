"""Pix3D preprocessing orchestration.

This module consolidates the Pix3D preprocessing flow into a single Python
entrypoint.  It mirrors the behaviour of the previous shell script while adding
structured logging, better diagnostics for worker/executor selection, and the
ability to override settings via CLI flags.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("PyYAML is required to run the Pix3D preprocessing pipeline. Install it with 'pip install pyyaml'.") from exc

from preprocess.build_index import build_index
from preprocess.canonicalize_meshes import canonicalize_directory
from preprocess.prepare_pix3d_metadata import prepare_metadata
from preprocess.sample_points_occ import process_directory
from utils.io import save_index

_ALLOWED_EXECUTORS = {"auto", "process", "thread"}


@dataclass
class Pix3DPaths:
    """Collection of resolved filesystem locations required for preprocessing."""

    dataset_root: Optional[Path]
    raw_mesh_dir: Path
    mask_dir: Path
    annotations_json: Path
    index_file: Path

    @property
    def output_root(self) -> Path:
        return self.index_file.parent

    @property
    def canonical_mesh_dir(self) -> Path:
        return self.output_root / "meshes"

    @property
    def occupancy_dir(self) -> Path:
        return self.output_root / "occ_npz"

    @property
    def cameras_json(self) -> Path:
        return self.output_root / "cameras.json"


@dataclass
class WorkerOverrides:
    canonicalize: Optional[int]
    occupancy: Optional[int]
    index: Optional[int]


@dataclass
class ExecutorSelection:
    value: str
    source: str


def _read_yaml(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dataset_root(paths_cfg: dict) -> Optional[Path]:
    root_value = paths_cfg.get("dataset_root")
    if not root_value:
        return None
    root_path = Path(root_value).expanduser()
    return root_path.resolve()


def _resolve_path(
    paths_cfg: dict,
    aliases: Iterable[str],
    dataset_root: Optional[Path],
    *,
    description: str,
) -> Path:
    for key in aliases:
        value = paths_cfg.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        if dataset_root is None:
            raise ValueError(
                f"paths.dataset_root must be provided when using a relative path for '{key}'."
            )
        return (dataset_root / path).resolve()
    alias_list = ", ".join(f"paths.{name}" for name in aliases)
    raise ValueError(f"Missing required path keys in config: [{alias_list}] for {description}")


def _resolve_pix3d_paths(config_path: Path) -> Pix3DPaths:
    cfg = _read_yaml(config_path)
    paths_cfg = cfg.get("paths") or {}
    data_cfg = cfg.get("data") or {}

    index_value = data_cfg.get("index_file")
    if not index_value:
        raise ValueError("data.index_file must be specified in the config.")
    index_path = Path(index_value).expanduser()
    if not index_path.is_absolute():
        index_path = (Path.cwd() / index_path).resolve()
    else:
        index_path = index_path.resolve()

    dataset_root = _resolve_dataset_root(paths_cfg)
    raw_mesh_dir = _resolve_path(
        paths_cfg,
        ("raw_mesh_dir", "mesh_dir", "model_dir"),
        dataset_root,
        description="raw mesh directory",
    )
    mask_dir = _resolve_path(
        paths_cfg,
        ("mask_dir", "masks_dir"),
        dataset_root,
        description="mask directory",
    )
    annotations_json = _resolve_path(
        paths_cfg,
        (
            "annotations_json",
            "annotation_json",
            "pix3d_json",
            "cameras_json",
            "camera_json",
            "metadata_json",
        ),
        dataset_root,
        description="annotations json",
    )

    return Pix3DPaths(
        dataset_root=dataset_root,
        raw_mesh_dir=raw_mesh_dir,
        mask_dir=mask_dir,
        annotations_json=annotations_json,
        index_file=index_path,
    )


def _parse_worker_value(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"Environment variable {name} must be >= 1")
    return value


def _resolve_workers(args: argparse.Namespace) -> WorkerOverrides:
    def pick(
        cli_value: Optional[int],
        *env_candidates: tuple[str, str],
    ) -> Optional[int]:
        if cli_value is not None:
            return cli_value
        for env_name, display in env_candidates:
            raw = os.environ.get(env_name)
            if not raw:
                continue
            value = _parse_worker_value(raw, env_name)
            print(f"Using {display}={value} from environment")
            return value
        return None

    canonicalize = pick(
        args.canon_workers,
        ("PIX3D_CANON_WORKERS", "PIX3D_CANON_WORKERS"),
        ("PREPROCESS_CANON_WORKERS", "PREPROCESS_CANON_WORKERS"),
        ("PREPROCESS_WORKERS", "PREPROCESS_WORKERS"),
    )
    occupancy = pick(
        args.occ_workers,
        ("PIX3D_OCC_WORKERS", "PIX3D_OCC_WORKERS"),
        ("PREPROCESS_OCC_WORKERS", "PREPROCESS_OCC_WORKERS"),
        ("PREPROCESS_WORKERS", "PREPROCESS_WORKERS"),
    )
    index = pick(
        args.index_workers,
        ("PIX3D_INDEX_WORKERS", "PIX3D_INDEX_WORKERS"),
        ("PREPROCESS_INDEX_WORKERS", "PREPROCESS_INDEX_WORKERS"),
        ("PREPROCESS_WORKERS", "PREPROCESS_WORKERS"),
    )
    return WorkerOverrides(canonicalize=canonicalize, occupancy=occupancy, index=index)


def _normalise_executor(value: str, *, source: str) -> ExecutorSelection:
    lowered = value.strip().lower()
    if lowered not in _ALLOWED_EXECUTORS:
        allowed = ", ".join(sorted(_ALLOWED_EXECUTORS))
        raise ValueError(f"Unsupported executor '{value}' from {source}. Allowed: {allowed}")
    return ExecutorSelection(value=lowered, source=source)


def _resolve_executor(args: argparse.Namespace) -> ExecutorSelection:
    if args.occ_executor:
        return _normalise_executor(args.occ_executor, source="CLI --occ-executor")

    dataset_override = os.environ.get("PIX3D_OCC_EXECUTOR")
    if dataset_override:
        selection = _normalise_executor(dataset_override, source="PIX3D_OCC_EXECUTOR")
        print(
            "Using Pix3D-specific occupancy executor from environment: "
            f"{selection.value}"
        )
        return selection

    global_override = os.environ.get("PREPROCESS_OCC_EXECUTOR")
    if global_override:
        selection = _normalise_executor(global_override, source="PREPROCESS_OCC_EXECUTOR")
        print(
            "Using global occupancy executor override from PREPROCESS_OCC_EXECUTOR: "
            f"{selection.value}"
        )
        return selection

    legacy_override = os.environ.get("PREPROCESS_EXECUTOR")
    if legacy_override:
        selection = _normalise_executor(legacy_override, source="PREPROCESS_EXECUTOR")
        print(
            "Using legacy PREPROCESS_EXECUTOR override for Pix3D occupancy sampling: "
            f"{selection.value}"
        )
        return selection

    return ExecutorSelection(value="process", source="default")


def _ensure_output_dirs(paths: Pix3DPaths) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.canonical_mesh_dir.mkdir(parents=True, exist_ok=True)
    paths.occupancy_dir.mkdir(parents=True, exist_ok=True)


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    paths = _resolve_pix3d_paths(config_path)
    overrides = _resolve_workers(args)
    executor = _resolve_executor(args)

    _ensure_output_dirs(paths)

    print("Pix3D preprocessing configuration:")
    print(f"  Config: {config_path}")
    print(f"  Dataset root: {paths.dataset_root if paths.dataset_root else '(not set)'}")
    print(f"  Raw meshes: {paths.raw_mesh_dir}")
    print(f"  Canonical meshes: {paths.canonical_mesh_dir}")
    print(f"  Occupancy outputs: {paths.occupancy_dir}")
    print(f"  Cameras JSON: {paths.cameras_json}")
    print(f"  Index file: {paths.index_file}")
    if overrides.canonicalize:
        print(f"  Canonicalize workers: {overrides.canonicalize}")
    if overrides.occupancy:
        print(f"  Occupancy workers: {overrides.occupancy}")
    if overrides.index:
        print(f"  Index workers: {overrides.index} (build_index runs single-threaded)")
    print(f"  Occupancy executor: {executor.value} (source: {executor.source})")
    print(f"  Skip existing occupancy files: {not args.recompute_occ}")

    canonicalize_directory(
        paths.raw_mesh_dir,
        paths.canonical_mesh_dir,
        workers=overrides.canonicalize,
    )
    process_directory(
        paths.canonical_mesh_dir,
        paths.occupancy_dir,
        args.n_surf,
        args.n_uniform,
        overrides.occupancy,
        skip_existing=not args.recompute_occ,
        executor=executor.value,
    )

    metadata_args = SimpleNamespace(
        config=config_path,
        annotations=paths.annotations_json,
        mask_dir=paths.mask_dir,
        mesh_dir=paths.raw_mesh_dir,
        dataset_root=paths.dataset_root,
        out=paths.cameras_json,
    )
    prepare_metadata(metadata_args)

    index_data = build_index(paths.mask_dir, paths.occupancy_dir, paths.cameras_json, args.split)
    save_index(index_data, paths.index_file)
    print(f"Saved Pix3D index to {paths.index_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pix3D preprocessing pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to the Pix3D YAML config")
    parser.add_argument("--canon-workers", type=int, default=None, help="Override canonicalization workers")
    parser.add_argument("--occ-workers", type=int, default=None, help="Override occupancy workers")
    parser.add_argument("--index-workers", type=int, default=None, help="Override index workers (reserved)")
    parser.add_argument(
        "--occ-executor",
        choices=sorted(_ALLOWED_EXECUTORS),
        default=None,
        help="Occupancy executor to use (overrides environment defaults)",
    )
    parser.add_argument("--n-surf", type=int, default=40000, help="Number of surface points to sample")
    parser.add_argument("--n-uniform", type=int, default=60000, help="Number of uniform occupancy samples")
    parser.add_argument(
        "--recompute-occ",
        action="store_true",
        help="Force recomputing occupancy .npz files even if they already exist",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to build (train/val/test)",
    )
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
