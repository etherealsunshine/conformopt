#!/usr/bin/env bash
# Detached controller for the five independent leakage-corrected A' folds.
set -euo pipefail

: "${OUTPUT:?set OUTPUT to a new PVC result directory}"
: "${PYTHON:=/home/dev/qfit_unet_data/.venv-qfit-audit/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_d1_aprime_leakage_corrected_cv.py"

if [[ -e "$OUTPUT/status.txt" ]]; then
  echo "refusing to overwrite existing controller status: $OUTPUT/status.txt" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids"
printf 'launching\n' > "$OUTPUT/status.txt"

declare -a pids=()
for fold in 0 1 2 3 4; do
  nohup "$PYTHON" "$SCRIPT" --output "$OUTPUT/split_$fold" --fold "$fold" \
    > "$OUTPUT/logs/split_$fold.log" 2>&1 < /dev/null &
  pid=$!
  pids[$fold]=$pid
  printf '%s\n' "$pid" > "$OUTPUT/pids/split_$fold.pid"
done

failed=0
for fold in 0 1 2 3 4; do
  if wait "${pids[$fold]}"; then
    printf 'complete\n' > "$OUTPUT/pids/split_$fold.status"
  else
    printf 'failed\n' > "$OUTPUT/pids/split_$fold.status"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  printf 'failed\n' > "$OUTPUT/status.txt"
  exit 1
fi

"$PYTHON" "$SCRIPT" --output "$OUTPUT" --aggregate \
  > "$OUTPUT/logs/aggregate.log" 2>&1
printf 'complete\n' > "$OUTPUT/status.txt"
