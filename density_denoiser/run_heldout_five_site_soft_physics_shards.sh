#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/dev/qfit_unet_data/.venv/bin/python
SELECTION=/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_soft_physics}
FRAME=${FRAME:-crystal}

mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards" "$OUTPUT/calibration"

# Gate the shared weights against deposited A and kinematic B on all five sites
# before spending any compute on the 250-start production run.
"$PYTHON" -m density_denoiser.five_site_optimizer \
  --selection "$SELECTION" \
  --frame "$FRAME" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT/calibration" \
  --target denoised \
  --soft-physics \
  --calibration-only \
  > "$OUTPUT/logs/calibration.log" 2>&1

sites=(
  4C16_A_MET258
  7F72_A_MET103
  3A1C_B_ARG447
  6H59_B_ARG144
  8Q6Q_B_ASP81
)

pids=()
for site in "${sites[@]}"; do
  nohup "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$SELECTION" \
    --frame "$FRAME" \
    --site "$site" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/shards/$site" \
    --target denoised \
    --n-starts 50 \
    --n-steps 500 \
    --soft-physics \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  printf '%s\n' "$pid" > "$OUTPUT/pids/$site.pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed )); then
  printf '%s\n' failed > "$OUTPUT/status.txt"
  exit 1
fi
printf '%s\n' complete > "$OUTPUT/status.txt"
