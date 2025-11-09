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

# Unless the caller explicitly provides an executor override, force the
# Pix3D occupancy sampling step to use the process pool.  This keeps the
# behaviour aligned with the original shell pipeline even when environment
# variables request the thread executor.
has_occ_executor=false
for arg in "$@"; do
  if [[ $arg == --occ-executor ]]; then
    has_occ_executor=true
    break
  elif [[ $arg == --occ-executor=* ]]; then
    has_occ_executor=true
    break
  fi
done

if [[ $has_occ_executor == false ]]; then
  set -- "$@" --occ-executor process
fi

exec "$PYTHON" -m preprocess.pix3d_pipeline --config "$CONFIG" "$@"
