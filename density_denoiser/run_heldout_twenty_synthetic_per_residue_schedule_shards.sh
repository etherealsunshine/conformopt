#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_per_residue_schedule_v1}
FRAME=${FRAME:-crystal}

original_sites=(
  4C16_A_MET258
  7F72_A_MET103
  3A1C_B_ARG447
  6H59_B_ARG144
  8Q6Q_B_ASP81
)

expanded_sites=(
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
  printf 'Refusing to overwrite existing run: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT/logs" "$OUTPUT/pids" \
  "$OUTPUT/shards/original5" "$OUTPUT/shards/expanded15" \
  "$OUTPUT/calibration/original5" "$OUTPUT/calibration/expanded15" \
  "$OUTPUT/audit/original5" "$OUTPUT/audit/expanded15" \
  "$OUTPUT/analysis"
printf '%s\n' calibrating > "$OUTPUT/status.txt"

for panel in original5 expanded15; do
  if [[ "$panel" == original5 ]]; then
    selection=$ORIGINAL_SELECTION
  else
    selection=$EXPANDED_SELECTION
  fi
  "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --frame "$FRAME" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/calibration/$panel" \
    --target synthetic \
    --per-residue-class-schedule \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --calibration-only \
    > "$OUTPUT/logs/calibration_$panel.log" 2>&1
done

printf '%s\n' optimizing > "$OUTPUT/status.txt"
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
    --target synthetic \
    --n-starts 50 \
    --n-steps 500 \
    --lr 1.0 \
    --per-residue-class-schedule \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  local pid=$!
  pids+=("$pid")
  labels+=("$site")
  printf '%s\n' "$pid" > "$OUTPUT/pids/$site.pid"
  printf '%s\n' running > "$OUTPUT/pids/$site.status"
}

for site in "${original_sites[@]}"; do
  launch_site original5 "$ORIGINAL_SELECTION" "$site"
done
for site in "${expanded_sites[@]}"; do
  launch_site expanded15 "$EXPANDED_SELECTION" "$site"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '%s\n' complete > "$OUTPUT/pids/${labels[$index]}.status"
  else
    printf '%s\n' failed > "$OUTPUT/pids/${labels[$index]}.status"
    failed=1
  fi
done

if (( failed )); then
  printf '%s\n' optimizer_failed > "$OUTPUT/status.txt"
  exit 1
fi

printf '%s\n' geometry_audit > "$OUTPUT/status.txt"
for panel in original5 expanded15; do
  if [[ "$panel" == original5 ]]; then
    selection=$ORIGINAL_SELECTION
  else
    selection=$EXPANDED_SELECTION
  fi
  "$PYTHON" -m density_denoiser.audit_five_site_endpoints \
    --selection "$selection" \
    --results-root "$OUTPUT/shards/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --target synthetic \
    > "$OUTPUT/logs/geometry_audit_$panel.log" 2>&1
done

printf '%s\n' tmol_calibration > "$OUTPUT/status.txt"
for panel in original5 expanded15; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --calibrate-only \
    > "$OUTPUT/logs/tmol_calibration_$panel.log" 2>&1
done

printf '%s\n' tmol_audit > "$OUTPUT/status.txt"
for panel in original5 expanded15; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --resume \
    > "$OUTPUT/logs/tmol_audit_$panel.log" 2>&1
done

printf '%s\n' summarizing > "$OUTPUT/status.txt"
for panel in original5 expanded15; do
  "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
    --audit-root "$OUTPUT/audit/$panel" \
    > "$OUTPUT/logs/strict_summary_$panel.log" 2>&1
done

printf '%s\n' occupancy_sensitivity > "$OUTPUT/status.txt"
"$PYTHON" -m density_denoiser.recompute_strict_occupancy_tolerances \
  --ensemble-table "$OUTPUT/audit/original5/ensemble_strict_audit.csv" \
  --ensemble-table "$OUTPUT/audit/expanded15/ensemble_strict_audit.csv" \
  --tolerance 0.20 \
  --tolerance 0.10 \
  --tolerance 0.05 \
  --output "$OUTPUT/analysis/occupancy_tolerance_sensitivity_v1" \
  > "$OUTPUT/logs/occupancy_sensitivity.log" 2>&1

printf '%s\n' complete > "$OUTPUT/status.txt"
