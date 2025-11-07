#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
CONFIG=${1:-configs/pix3d.yaml}

mapfile -t CFG < <(
  "$PYTHON" - <<'PY' "$CONFIG"
import sys, yaml
from pathlib import Path

cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
train_cfg = cfg.get('train') or {}
data_cfg = cfg.get('data') or {}

out_dir = train_cfg.get('out_dir')
if not out_dir:
    raise SystemExit('train.out_dir must be set in the config.')

dataset_name = data_cfg.get('dataset_name')
if not dataset_name:
    index_file = data_cfg.get('index_file')
    if index_file:
        dataset_name = Path(index_file).parent.name
if not dataset_name:
    raise SystemExit('data.dataset_name must be set in the config.')

print(out_dir)
print(dataset_name)
PY
)

OUT_DIR=${CFG[0]}
DATASET_NAME=${CFG[1]}
CKPT_PATH="$OUT_DIR/gcp_final_${DATASET_NAME}.pt"
PRIOR_DIR="outputs/priors/${DATASET_NAME}"

"$PYTHON" train/train_gcp.py --config "$CONFIG"
"$PYTHON" train/export_priors.py \
  --ckpt "$CKPT_PATH" \
  --config "$CONFIG" \
  --out "$PRIOR_DIR"
