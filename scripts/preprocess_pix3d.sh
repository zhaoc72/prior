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

# Default to the process executor when neither the CLI nor environment provide
# an explicit override so the behaviour stays aligned with the historical shell
# pipeline.
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

# Only append a process executor override when the caller has not provided one
# via CLI *and* no environment variables have already requested a specific
# executor. This allows PIX3D_OCC_EXECUTOR/PREPROCESS_* overrides (for example,
# forcing threads on fork-unsafe systems) to take effect while keeping the
# default aligned with the historical process-pool behaviour.
if [[ $has_occ_executor == false ]]; then
  if [[ -z "${PIX3D_OCC_EXECUTOR:-}" && -z "${PREPROCESS_OCC_EXECUTOR:-}" && -z "${PREPROCESS_EXECUTOR:-}" ]]; then
    set -- "$@" --occ-executor process
  fi
fi

exec "$PYTHON" -m preprocess.pix3d_pipeline --config "$CONFIG" "$@"
