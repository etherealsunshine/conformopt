#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_v1}
FRAME=${FRAME:-crystal}

original_sites=(3A1C_B_ARG447 6H59_B_ARG144)
expanded_sites=(3NY7_B_LYS19)

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

filter_selection() {
  local source=$1
  local destination=$2
  shift 2
  "$PYTHON" - "$source" "$destination" "$@" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
wanted = set(sys.argv[3:])
payload = json.loads(source.read_text())
payload["sites"] = [site for site in payload["sites"] if site["key"] in wanted]
found = {site["key"] for site in payload["sites"]}
if found != wanted:
    raise SystemExit(f"selection mismatch: wanted={sorted(wanted)} found={sorted(found)}")
payload["subset_source"] = str(source)
payload["subset_rule"] = "predeclared four-chi schedule ablation; no outcome filtering"
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, destination)
PY
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/selections" \
  "$OUTPUT/shards/original2" "$OUTPUT/shards/expanded1" \
  "$OUTPUT/calibration/original2" "$OUTPUT/calibration/expanded1" \
  "$OUTPUT/audit/original2" "$OUTPUT/audit/expanded1"

ORIGINAL_SUBSET="$OUTPUT/selections/original2.json"
EXPANDED_SUBSET="$OUTPUT/selections/expanded1.json"
filter_selection "$ORIGINAL_SELECTION" "$ORIGINAL_SUBSET" "${original_sites[@]}"
filter_selection "$EXPANDED_SELECTION" "$EXPANDED_SUBSET" "${expanded_sites[@]}"

run_calibration() {
  local panel=$1
  local selection=$2
  "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --frame "$FRAME" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/calibration/$panel" \
    --target denoised \
    --n-steps 500 \
    --lr 1.0 \
    --per-residue-class-schedule \
    --four-chi-stage-steps 200 \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw 1.0 \
    --lambda-rot 0.5 \
    --lambda-clash 5.0 \
    --calibration-only \
    > "$OUTPUT/logs/calibration_$panel.log" 2>&1
}

write_status calibrating
run_calibration original2 "$ORIGINAL_SUBSET"
run_calibration expanded1 "$EXPANDED_SUBSET"

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
    --four-chi-stage-steps 200 \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw 1.0 \
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
  launch_site original2 "$ORIGINAL_SUBSET" "$site"
done
for site in "${expanded_sites[@]}"; do
  launch_site expanded1 "$EXPANDED_SUBSET" "$site"
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
for panel in original2 expanded1; do
  if [[ "$panel" == original2 ]]; then
    selection=$ORIGINAL_SUBSET
  else
    selection=$EXPANDED_SUBSET
  fi
  "$PYTHON" -m density_denoiser.audit_five_site_endpoints \
    --selection "$selection" \
    --results-root "$OUTPUT/shards/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --target denoised \
    > "$OUTPUT/logs/geometry_audit_$panel.log" 2>&1
done

write_status tmol_calibration
for panel in original2 expanded1; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --calibrate-only \
    > "$OUTPUT/logs/tmol_calibration_$panel.log" 2>&1
done

write_status tmol_audit
for panel in original2 expanded1; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --resume \
    > "$OUTPUT/logs/tmol_audit_$panel.log" 2>&1
done

write_status summarizing
for panel in original2 expanded1; do
  "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
    --audit-root "$OUTPUT/audit/$panel" \
    > "$OUTPUT/logs/strict_summary_$panel.log" 2>&1
done

write_status complete
