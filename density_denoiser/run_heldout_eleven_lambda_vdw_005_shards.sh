#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_eleven_lambda_vdw_005_v1}
FRAME=${FRAME:-crystal}
LAMBDA_VDW=${LAMBDA_VDW:-0.05}

original_sites=(
  3A1C_B_ARG447
)

expanded_sites=(
  1ZV8_E_ASN1
  2V05_A_HIS168
  2VFP_A_TYR417
  3GMI_A_GLU5
  3K8W_A_SER337
  3NY7_B_LYS19
  5DBA_A_TRP325
  7UO8_A_GLN53
  8DJ2_A_VAL893
  8FBE_B_ILE92
)

write_atomic() {
  local path=$1
  local value=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

write_status() {
  write_atomic "$OUTPUT/status.txt" "$1"
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT/logs" "$OUTPUT/pids" \
  "$OUTPUT/shards/original1" "$OUTPUT/shards/expanded10" \
  "$OUTPUT/calibration/original1" "$OUTPUT/calibration/expanded10" \
  "$OUTPUT/audit/original1" "$OUTPUT/audit/expanded10"

run_calibration() {
  local panel=$1
  local selection=$2
  shift 2
  local command=(
    "$PYTHON" -m density_denoiser.five_site_optimizer
    --selection "$selection"
    --frame "$FRAME"
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT/calibration/$panel"
    --target denoised
    --per-residue-class-schedule
    --physics-refinement-steps 200
    --physics-refinement-lr-scale 0.1
    --lambda-vdw "$LAMBDA_VDW"
    --lambda-rot 0.5
    --lambda-clash 5.0
    --calibration-only
  )
  local site
  for site in "$@"; do
    command+=(--site "$site")
  done
  "${command[@]}" > "$OUTPUT/logs/calibration_$panel.log" 2>&1
}

write_status calibrating
run_calibration original1 "$ORIGINAL_SELECTION" "${original_sites[@]}"
run_calibration expanded10 "$EXPANDED_SELECTION" "${expanded_sites[@]}"

write_status optimizing
pids=()
labels=()

launch_site() {
  local panel=$1
  local selection=$2
  local site=$3
  nohup "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --frame "$FRAME" \
    --site "$site" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/shards/$panel/$site" \
    --target denoised \
    --n-starts 50 \
    --n-steps 500 \
    --lr 1.0 \
    --per-residue-class-schedule \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw "$LAMBDA_VDW" \
    --lambda-rot 0.5 \
    --lambda-clash 5.0 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  local pid=$!
  pids+=("$pid")
  labels+=("$site")
  write_atomic "$OUTPUT/pids/$site.pid" "$pid"
  write_atomic "$OUTPUT/pids/$site.status" running
}

for site in "${original_sites[@]}"; do
  launch_site original1 "$ORIGINAL_SELECTION" "$site"
done
for site in "${expanded_sites[@]}"; do
  launch_site expanded10 "$EXPANDED_SELECTION" "$site"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    write_atomic "$OUTPUT/pids/${labels[$index]}.status" complete
  else
    write_atomic "$OUTPUT/pids/${labels[$index]}.status" failed
    failed=1
  fi
done

if (( failed )); then
  write_status optimizer_failed
  exit 1
fi

write_status geometry_audit
for panel in original1 expanded10; do
  if [[ "$panel" == original1 ]]; then
    selection=$ORIGINAL_SELECTION
  else
    selection=$EXPANDED_SELECTION
  fi
  "$PYTHON" -m density_denoiser.audit_five_site_endpoints \
    --selection "$selection" \
    --results-root "$OUTPUT/shards/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --target denoised \
    > "$OUTPUT/logs/geometry_audit_$panel.log" 2>&1
done

write_status tmol_calibration
for panel in original1 expanded10; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --calibrate-only \
    > "$OUTPUT/logs/tmol_calibration_$panel.log" 2>&1
done

write_status tmol_audit
for panel in original1 expanded10; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --resume \
    > "$OUTPUT/logs/tmol_audit_$panel.log" 2>&1
done

write_status summarizing
for panel in original1 expanded10; do
  "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
    --audit-root "$OUTPUT/audit/$panel" \
    > "$OUTPUT/logs/strict_summary_$panel.log" 2>&1
done

write_status complete
