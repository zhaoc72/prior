"""Convert Pix3D official annotations to GCP metadata format."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


def _ensure_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not parse {name} value '{value}' as float") from exc


def _parse_focal(entry: dict[str, Any]) -> tuple[float, float]:
    focal = entry.get("focal_length") or entry.get("focal")
    if isinstance(focal, (int, float)):
        f = _ensure_float(focal, "focal_length")
        return f, f
    if isinstance(focal, (list, tuple)):
        if len(focal) >= 2:
            fx = _ensure_float(focal[0], "focal_length[0]")
            fy = _ensure_float(focal[1], "focal_length[1]")
            return fx, fy
        if len(focal) == 1:
            f = _ensure_float(focal[0], "focal_length[0]")
            return f, f
    if isinstance(focal, dict):
        fx = focal.get("fx") or focal.get("f_x") or focal.get("fx1") or focal.get("fx0")
        fy = focal.get("fy") or focal.get("f_y") or focal.get("fy1") or focal.get("fy0")
        if fx is None and fy is None:
            if "value" in focal:
                f = _ensure_float(focal["value"], "focal_length")
                return f, f
            raise RuntimeError("focal_length dictionary missing fx/fy values")
        if fx is None:
            fx = fy
        if fy is None:
            fy = fx
        return _ensure_float(fx, "focal_length.fx"), _ensure_float(fy, "focal_length.fy")
    raise RuntimeError("Unsupported focal length format in annotation entry")


def _parse_principal(entry: dict[str, Any]) -> tuple[float, float]:
    principal = entry.get("principal") or entry.get("principal_point") or entry.get("principal_pt")
    if isinstance(principal, (list, tuple)) and len(principal) >= 2:
        return _ensure_float(principal[0], "principal[0]"), _ensure_float(principal[1], "principal[1]")
    if isinstance(principal, dict):
        cx = principal.get("cx") or principal.get("c_x") or principal.get("x")
        cy = principal.get("cy") or principal.get("c_y") or principal.get("y")
        if cx is not None and cy is not None:
            return _ensure_float(cx, "principal.cx"), _ensure_float(cy, "principal.cy")
    viewport = entry.get("viewport") or entry.get("image_size") or entry.get("size")
    if isinstance(viewport, (list, tuple)) and len(viewport) >= 2:
        return float(viewport[0]) * 0.5, float(viewport[1]) * 0.5
    raise RuntimeError("Principal point not provided in annotation entry")


def _matrix3(value: Any) -> list[list[float]]:
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value):
            return [[_ensure_float(x, "rot_mat") for x in row] for row in value]
        if len(value) == 9:
            flat = [_ensure_float(x, "rot_mat") for x in value]
            return [flat[0:3], flat[3:6], flat[6:9]]
    raise RuntimeError("Rotation matrix must be 3x3 or length-9 list")


def _vector3(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [_ensure_float(v, "trans_mat") for v in value[:3]]
    if isinstance(value, dict):
        components = [value.get("x"), value.get("y"), value.get("z")]
        if all(component is not None for component in components):
            return [_ensure_float(component, f"trans_mat.{axis}") for component, axis in zip(components, "xyz")]
    raise RuntimeError("Translation vector must contain three components")


def _matrix4(value: Any) -> list[list[float]]:
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in value):
            return [[_ensure_float(x, "extrinsics") for x in row] for row in value]
        if len(value) == 16:
            flat = [_ensure_float(x, "extrinsics") for x in value]
            return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    raise RuntimeError("Extrinsics must be 4x4 or length-16 list")


def _parse_extrinsics(entry: dict[str, Any]) -> list[list[float]]:
    if "extrinsics" in entry:
        return _matrix4(entry["extrinsics"])
    rotation = entry.get("rot_mat") or entry.get("rotation") or entry.get("R")
    translation = entry.get("trans_mat") or entry.get("translation") or entry.get("T")
    if rotation is None or translation is None:
        raise RuntimeError("Annotation entry is missing rotation/translation")
    rot = _matrix3(rotation)
    trans = _vector3(translation)
    extr = [
        [rot[0][0], rot[0][1], rot[0][2], trans[0]],
        [rot[1][0], rot[1][1], rot[1][2], trans[1]],
        [rot[2][0], rot[2][1], rot[2][2], trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return extr


def _normalise_intrinsics(entry: dict[str, Any]) -> list[list[float]]:
    if "intrinsics" in entry:
        intr = entry["intrinsics"]
        if isinstance(intr, (list, tuple)):
            if len(intr) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in intr):
                return [[_ensure_float(x, "intrinsics") for x in row] for row in intr]
            if len(intr) == 9:
                flat = [_ensure_float(x, "intrinsics") for x in intr]
                return [flat[0:3], flat[3:6], flat[6:9]]
        raise RuntimeError("Intrinsics matrix must be 3x3")
    fx, fy = _parse_focal(entry)
    cx, cy = _parse_principal(entry)
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compute_relative(path: Path, candidates: Iterable[Path | None]) -> str:
    for root in candidates:
        if root is None:
            continue
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _strip_suffix(path_str: str) -> str:
    path = Path(path_str)
    suffix = path.suffix
    if suffix:
        return (path.parent / path.stem).as_posix()
    return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Pix3D metadata for preprocessing")
    parser.add_argument("--config", type=Path, required=True, help="Path to Pix3D config YAML")
    parser.add_argument("--annotations", type=Path, required=True, help="Official pix3d.json path")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing masks")
    parser.add_argument("--mesh-dir", type=Path, required=True, help="Directory containing meshes")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Pix3D dataset root (optional)")
    parser.add_argument("--out", type=Path, required=True, help="Output cameras.json path")
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        raise RuntimeError("Unexpected annotation file format; expected a list of entries")
    return data


def determine_split(entry: dict[str, Any]) -> str:
    split = entry.get("split") or entry.get("subset") or entry.get("set")
    if isinstance(split, str) and split:
        return split.lower()
    for key in ("train", "val", "test"):
        flag = entry.get(key)
        if isinstance(flag, bool) and flag:
            return key
        if isinstance(flag, (int, float)) and flag:
            return key
    return "train"


def prepare_metadata(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    cfg = _load_yaml(args.config)
    entries = load_annotations(args.annotations.resolve())
    categories = cfg.get("data", {}).get("categories") or []
    if not categories:
        categories = sorted({ann.get("category") or ann.get("cat") for ann in entries if ann.get("category") or ann.get("cat")})
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}

    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    mask_dir = args.mask_dir.resolve()
    mesh_dir = args.mesh_dir.resolve()
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for ann in entries:
        category = ann.get("category") or ann.get("cat")
        if not category:
            skipped += 1
            continue
        if cat_to_id and category not in cat_to_id:
            skipped += 1
            continue
        try:
            intrinsics = _normalise_intrinsics(ann)
            extrinsics = _parse_extrinsics(ann)
        except RuntimeError:
            skipped += 1
            continue

        mask_value = ann.get("mask") or ann.get("mask_file")
        model_value = ann.get("model") or ann.get("model_file")
        if not mask_value or not model_value:
            skipped += 1
            continue

        mask_path = Path(mask_value)
        if not mask_path.is_absolute():
            if dataset_root:
                mask_path = dataset_root / mask_path
            else:
                mask_path = mask_dir / mask_path
        mesh_path = Path(model_value)
        if not mesh_path.is_absolute():
            if dataset_root:
                mesh_path = dataset_root / mesh_path
            else:
                mesh_path = mesh_dir / mesh_path

        mask_rel = _compute_relative(mask_path, (mask_dir, dataset_root))
        mesh_rel = _compute_relative(mesh_path, (mesh_dir, dataset_root))
        mesh_id = _strip_suffix(mesh_rel)

        split = determine_split(ann)
        cat_id = cat_to_id.get(category, 0)
        splits[split].append(
            {
                "mask_file": mask_rel,
                "mesh_id": mesh_id,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "category": category,
                "category_id": cat_id,
            }
        )

    if not splits:
        raise RuntimeError("No valid Pix3D annotations were processed; please check the dataset paths")

    result = {split: items for split, items in splits.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f)

    total = sum(len(items) for items in result.values())
    print(f"Wrote {total} Pix3D entries to {args.out} (skipped {skipped})")
    return result


if __name__ == "__main__":
    prepare_metadata(parse_args())

