#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
if (($# > 0)); then
  CONFIG=$1
  shift
else
  CONFIG=configs/pix3d.yaml
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  echo "Usage: bash $0 <path/to/config.yaml> [additional options]" >&2
  exit 1
fi

exec "$PYTHON" -m preprocess.pix3d_pipeline --config "$CONFIG" "$@"
