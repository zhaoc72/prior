#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONFIG=${1:-configs/scannetv2.yaml}

mapfile -t CFG < <(
  "$PYTHON" - <<'PY' "$CONFIG"
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
paths = cfg.get('paths') or {}
required = ['raw_mesh_dir', 'mask_dir', 'cameras_json']
missing = [key for key in required if key not in paths or not paths[key]]
if missing:
    raise SystemExit(f"Missing required path keys in config: {missing}")

index_file = cfg.get('data', {}).get('index_file')
if not index_file:
    raise SystemExit('data.index_file must be specified in the config.')

print(paths['raw_mesh_dir'])
print(paths['mask_dir'])
print(paths['cameras_json'])
print(index_file)
PY
)

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
