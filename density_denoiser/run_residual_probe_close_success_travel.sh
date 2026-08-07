#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_close_success_residual_probe_travel_v1}

FROZEN_BASE=/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_v2_single_rule_v1
CONTROL=/home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2
CONTROL_WATER2=/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1

sites=(
  2VFP_A_TYR417
  4C16_A_MET258
  5KWB_A_PHE591
  5Z8H_A_MET730
  7F72_A_MET103
  7T7A_A_LEU396
  8FBE_B_ILE92
)

write_atomic() {
  local path=$1
  local value=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing replay: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/sites"
write_atomic "$OUTPUT/status.txt" "launching"

common_args=(
  --frame crystal
  --checkpoint "$CHECKPOINT"
  --target synthetic
  --n-steps 500
  --lr 1.0
  --per-residue-class-schedule
  --four-chi-stage-steps 200
  --physics-refinement-steps 200
  --physics-refinement-lr-scale 0.1
  --lambda-vdw 1.0
  --lambda-rot 0.5
  --lambda-clash 5.0
  --vdw-threshold 3.0
  --clash-threshold 2.5
  --symmetry-hard-threshold 2.0
  --symmetry-barrier-buffer 0.25
  --symmetry-barrier-scale 0.0
  --seed 41
  --residual-minor-probe
  --residual-probe-lobe-radius 1.0
)

launch_site() {
  local site=$1
  local selection panel endpoint ensembles
  local -a starts
  case "$site" in
    2VFP_A_TYR417)
      selection=$EXPANDED_SELECTION
      endpoint="$CONTROL_WATER2/shards/$site/synthetic/${site}_starts.csv"
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/water2/ensemble_strict_audit.csv"
      starts=(12 19 27 28 34 42 46 47 49)
      ;;
    4C16_A_MET258)
      selection=$ORIGINAL_SELECTION
      panel=original5
      starts=(31 40 45)
      ;;
    5KWB_A_PHE591)
      selection=$EXPANDED_SELECTION
      panel=expanded15
      starts=(13)
      ;;
    5Z8H_A_MET730)
      selection=$EXPANDED_SELECTION
      panel=expanded15
      starts=(0 3 15 27 33 41 43)
      ;;
    7F72_A_MET103)
      selection=$ORIGINAL_SELECTION
      panel=original5
      starts=(45)
      ;;
    7T7A_A_LEU396)
      selection=$EXPANDED_SELECTION
      panel=expanded15
      starts=(8 38 44 46)
      ;;
    8FBE_B_ILE92)
      selection=$EXPANDED_SELECTION
      panel=expanded15
      starts=(46)
      ;;
  esac
  if [[ "$site" != 2VFP_A_TYR417 ]]; then
    endpoint="$CONTROL/shards/$panel/$site/synthetic/${site}_starts.csv"
    if [[ "$panel" == original5 ]]; then
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/original5/ensemble_strict_audit.csv"
    else
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/expanded13/ensemble_strict_audit.csv"
    fi
  fi
  local -a start_args=()
  local start
  for start in "${starts[@]}"; do
    start_args+=(--residual-probe-start "$start")
  done
  (
    if "$PYTHON" -m density_denoiser.five_site_optimizer \
      --selection "$selection" \
      --site "$site" \
      --output "$OUTPUT/sites/$site" \
      --residual-probe-endpoints "$endpoint" \
      --residual-probe-ensembles "$ensembles" \
      "${start_args[@]}" \
      "${common_args[@]}" \
      > "$OUTPUT/logs/$site.log" 2>&1; then
      write_atomic "$OUTPUT/pids/$site.status" "complete"
    else
      write_atomic "$OUTPUT/pids/$site.status" "failed"
      exit 1
    fi
  ) &
  printf '%s\n' "$!" > "$OUTPUT/pids/$site.pid"
}

for site in "${sites[@]}"; do
  launch_site "$site"
done
write_atomic "$OUTPUT/status.txt" "running"

failed=0
for site in "${sites[@]}"; do
  if ! wait "$(cat "$OUTPUT/pids/$site.pid")"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  write_atomic "$OUTPUT/status.txt" "failed"
  exit 1
fi

"$PYTHON" - "$OUTPUT" <<'PY'
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("sites/*/residual_minor_probe.csv")):
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
fixed = [
    row for row in rows
    if row["occupancy_mode"] == "fixed_minor"
    and row["recovered_minor_lt_1A"].lower() == "true"
]
if len(fixed) != 26:
    raise RuntimeError(f"expected 26 reproduced fixed successes, got {len(fixed)}")

with tempfile.NamedTemporaryFile(
    "w", dir=root, delete=False, newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    temporary = handle.name
os.replace(temporary, root / "replay_all.csv")

payload = {
    "status": "complete",
    "diagnostic_only": True,
    "metric_changed": False,
    "replayed_close_fixed_successes": len(fixed),
}
with tempfile.NamedTemporaryFile(
    "w", dir=root, delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, root / "summary.json")
PY

write_atomic "$OUTPUT/status.txt" "complete"
