#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONFIG=${1:-configs/vkitti2.yaml}

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
root_path = Path(root_value) if root_value else None

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
            return str(candidate)
    alias_list = ", ".join(f"paths.{name}" for name in aliases)
    raise SystemExit(f"Missing required path keys in config: [{alias_list}]")

raw_mesh_dir = resolve_path(("raw_mesh_dir", "mesh_dir", "model_dir"))
mask_dir = resolve_path(("mask_dir", "masks_dir"))
cameras_json = resolve_path(("cameras_json", "camera_json", "metadata_json"))

print(raw_mesh_dir)
print(mask_dir)
print(cameras_json)
print(index_file)
PY
); then
  exit 1
fi

RAW_MESH_DIR=${CFG[0]}
MASK_DIR=${CFG[1]}
CAMERA_JSON=${CFG[2]}
INDEX_FILE=${CFG[3]}
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

"$PYTHON" preprocess/build_index.py \
  --mask_dir "$MASK_DIR" \
  --occ_dir "$OUT_DIR/occ_npz" \
  --cams "$CAMERA_JSON" \
  --out "$INDEX_FILE" \
  --split train
