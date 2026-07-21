#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
SELECTION=${SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_two_stage_prospective_v1}
FRAME=${FRAME:-crystal}

sites=(
  1ZV8_E_ASN1
  6Y4G_B_CYS260
  7UO8_A_GLN53
  3GMI_A_GLU5
  2V05_A_HIS168
  8FBE_B_ILE92
  7T7A_A_LEU396
  3NY7_B_LYS19
  5KWB_A_PHE591
  3K8W_A_SER337
  4MKM_A_THR77
  5DBA_A_TRP325
  2VFP_A_TYR417
  8DJ2_A_VAL893
  5Z8H_A_MET730
)

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing prospective run: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards" "$OUTPUT/calibration" "$OUTPUT/audit"
printf '%s\n' calibrating > "$OUTPUT/status.txt"

"$PYTHON" -m density_denoiser.five_site_optimizer \
  --selection "$SELECTION" \
  --frame "$FRAME" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT/calibration" \
  --target denoised \
  --physics-refinement-steps 200 \
  --physics-refinement-lr-scale 0.1 \
  --calibration-only \
  > "$OUTPUT/logs/calibration.log" 2>&1

printf '%s\n' optimizing > "$OUTPUT/status.txt"
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
    --lr 1.0 \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  printf '%s\n' "$pid" > "$OUTPUT/pids/$site.pid"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s\n' complete > "$OUTPUT/pids/${sites[$index]}.status"
  else
    printf '%s\n' failed > "$OUTPUT/pids/${sites[$index]}.status"
    failed=1
  fi
done

if (( failed )); then
  printf '%s\n' optimizer_failed > "$OUTPUT/status.txt"
  exit 1
fi

printf '%s\n' geometry_audit > "$OUTPUT/status.txt"
"$PYTHON" -m density_denoiser.audit_five_site_endpoints \
  --selection "$SELECTION" \
  --results-root "$OUTPUT/shards" \
  --output "$OUTPUT/audit" \
  --target denoised \
  > "$OUTPUT/logs/geometry_audit.log" 2>&1

printf '%s\n' tmol_calibration > "$OUTPUT/status.txt"
"$TMOL_PYTHON" five_site_tmol_audit.py \
  --input-root "$OUTPUT/audit" \
  --output "$OUTPUT/audit" \
  --calibrate-only \
  > "$OUTPUT/logs/tmol_calibration.log" 2>&1

printf '%s\n' tmol_audit > "$OUTPUT/status.txt"
"$TMOL_PYTHON" five_site_tmol_audit.py \
  --input-root "$OUTPUT/audit" \
  --output "$OUTPUT/audit" \
  --resume \
  > "$OUTPUT/logs/tmol_audit.log" 2>&1

printf '%s\n' summarizing > "$OUTPUT/status.txt"
"$PYTHON" -m density_denoiser.summarize_endpoint_audit \
  --audit-root "$OUTPUT/audit" \
  > "$OUTPUT/logs/strict_summary.log" 2>&1

printf '%s\n' complete > "$OUTPUT/status.txt"
