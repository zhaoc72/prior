#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONFIG=${1:-configs/pix3d.yaml}

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  echo "Usage: bash $0 <path/to/config.yaml>" >&2
  exit 1
fi

if ! mapfile -t CFG < <(
  "$PYTHON" - <<'PY' "$CONFIG"
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
paths = cfg.get("paths") or {}
data_cfg = cfg.get("data") or {}

index_file = data_cfg.get("index_file")
if not index_file:
    raise SystemExit("data.index_file must be specified in the config.")

root_value = paths.get("dataset_root")
root_path = Path(root_value).resolve() if root_value else None

def resolve_path(aliases):
    for key in aliases:
        value = paths.get(key)
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                if root_path is None:
                    raise SystemExit(
                        f"paths.dataset_root must be provided when using a relative path for '{key}'."
                    )
                candidate = root_path / candidate
            return str(candidate.resolve())
    alias_list = ", ".join(f"paths.{name}" for name in aliases)
    raise SystemExit(f"Missing required path keys in config: [{alias_list}]")

raw_mesh_dir = resolve_path(("raw_mesh_dir", "mesh_dir", "model_dir"))
mask_dir = resolve_path(("mask_dir", "masks_dir"))
annotations_json = resolve_path(
    (
        "annotations_json",
        "annotation_json",
        "pix3d_json",
        "cameras_json",
        "camera_json",
        "metadata_json",
    )
)

print(str(root_path) if root_path else "")
print(raw_mesh_dir)
print(mask_dir)
print(annotations_json)
print(index_file)
PY
); then
  exit 1
fi

if ((${#CFG[@]} < 5)); then
  echo "Failed to load configuration values from $CONFIG. Ensure the config is valid and that PyYAML is installed." >&2
  exit 1
fi

DATASET_ROOT=${CFG[0]}
RAW_MESH_DIR=${CFG[1]}
MASK_DIR=${CFG[2]}
ANNOTATIONS_JSON=${CFG[3]}
INDEX_FILE=${CFG[4]}
OUT_DIR=$(dirname "$INDEX_FILE")

mkdir -p "$OUT_DIR"

"$PYTHON" preprocess/canonicalize_meshes.py \
  --src "$RAW_MESH_DIR" \
  --dst "$OUT_DIR/meshes"

"$PYTHON" preprocess/sample_points_occ.py \
  --mesh_dir "$OUT_DIR/meshes" \
  --out "$OUT_DIR/occ_npz" \
  --n_surf 40000 \
  --n_uniform 60000

GENERATED_CAMERAS="$OUT_DIR/cameras.json"
CONVERT_ARGS=(
  --config "$CONFIG"
  --annotations "$ANNOTATIONS_JSON"
  --mask-dir "$MASK_DIR"
  --mesh-dir "$RAW_MESH_DIR"
  --out "$GENERATED_CAMERAS"
)
if [[ -n "$DATASET_ROOT" ]]; then
  CONVERT_ARGS+=(--dataset-root "$DATASET_ROOT")
fi

"$PYTHON" preprocess/prepare_pix3d_metadata.py "${CONVERT_ARGS[@]}"

"$PYTHON" preprocess/build_index.py \
  --mask_dir "$MASK_DIR" \
  --occ_dir "$OUT_DIR/occ_npz" \
  --cams "$GENERATED_CAMERAS" \
  --out "$INDEX_FILE" \
  --split train
