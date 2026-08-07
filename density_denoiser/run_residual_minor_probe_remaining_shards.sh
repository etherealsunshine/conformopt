#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_remaining_residual_minor_probe_v1}

FROZEN_BASE=/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_v2_single_rule_v1
CONTROL=/home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2
TAIL=/home/dev/qfit_unet_data/density_denoiser/heldout_five_tail_residual_minor_probe_v1

sites=(
  2V05_A_HIS168
  3A1C_B_ARG447
  3GMI_A_GLU5
  3NY7_B_LYS19
  5DBA_A_TRP325
  5KWB_A_PHE591
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
  printf 'Refusing to overwrite existing diagnostic: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/sites"
write_atomic "$OUTPUT/status.txt" "launching"

"$PYTHON" - "$OUTPUT/run_manifest.json" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

output = Path(sys.argv[1])
sources = [
    Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py"),
    Path(
        "/home/dev/workspace/density_denoiser/"
        "run_residual_minor_probe_remaining_shards.sh"
    ),
    Path("/home/dev/workspace/scripts/analyze_residual_minor_probe.py"),
]
payload = {
    "diagnostic": "residual_minor_probe_remaining_v1",
    "diagnostic_only": True,
    "production_changed": False,
    "metric_changed": False,
    "frozen_metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
    "extends_tail_run": (
        "/home/dev/qfit_unet_data/density_denoiser/"
        "heldout_five_tail_residual_minor_probe_v1"
    ),
    "sites": [
        "2V05_A_HIS168", "3A1C_B_ARG447", "3GMI_A_GLU5",
        "3NY7_B_LYS19", "5DBA_A_TRP325", "5KWB_A_PHE591",
        "7F72_A_MET103", "7T7A_A_LEU396", "8FBE_B_ILE92",
    ],
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
}
with tempfile.NamedTemporaryFile(
    "w", dir=output.parent, delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
PY

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
  case "$site" in
    3A1C_B_ARG447|7F72_A_MET103)
      selection=$ORIGINAL_SELECTION
      panel=original5
      ;;
    *)
      selection=$EXPANDED_SELECTION
      panel=expanded15
      ;;
  esac
  endpoint="$CONTROL/shards/$panel/$site/synthetic/${site}_starts.csv"
  if [[ "$panel" == original5 ]]; then
    ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/original5/ensemble_strict_audit.csv"
  else
    ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/expanded13/ensemble_strict_audit.csv"
  fi
  (
    if "$PYTHON" -m density_denoiser.five_site_optimizer \
      --selection "$selection" \
      --site "$site" \
      --output "$OUTPUT/sites/$site" \
      --residual-probe-endpoints "$endpoint" \
      --residual-probe-ensembles "$ensembles" \
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

"$PYTHON" - "$OUTPUT" "$TAIL" <<'PY'
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
tail = Path(sys.argv[2])
remaining = []
for path in sorted(root.glob("sites/*/residual_minor_probe.csv")):
    with path.open(newline="") as handle:
        remaining.extend(csv.DictReader(handle))
with (tail / "residual_minor_probe_all.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle)) + remaining

expected_remaining = {
    "2V05_A_HIS168": 4,
    "3A1C_B_ARG447": 1,
    "3GMI_A_GLU5": 9,
    "3NY7_B_LYS19": 1,
    "5DBA_A_TRP325": 5,
    "5KWB_A_PHE591": 1,
    "7F72_A_MET103": 3,
    "7T7A_A_LEU396": 5,
    "8FBE_B_ILE92": 1,
}
counts = defaultdict(lambda: defaultdict(int))
for row in remaining:
    counts[row["site"]][row["occupancy_mode"]] += 1
for site, expected in expected_remaining.items():
    for mode in ("fixed_minor", "free_sigmoid"):
        if counts[site][mode] != expected:
            raise RuntimeError(
                f"{site} {mode}: expected {expected}, got {counts[site][mode]}"
            )
for mode in ("fixed_minor", "free_sigmoid"):
    total = sum(row["occupancy_mode"] == mode for row in rows)
    if total != 129:
        raise RuntimeError(f"{mode}: expected 129 combined rows, got {total}")

with tempfile.NamedTemporaryFile(
    "w", dir=root, delete=False, newline=""
) as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    temporary = handle.name
os.replace(temporary, root / "residual_minor_probe_all_20site.csv")
PY

"$PYTHON" scripts/analyze_residual_minor_probe.py \
  --input "$OUTPUT/residual_minor_probe_all_20site.csv" \
  --separations \
    /home/dev/qfit_unet_data/density_denoiser/deposited_panel_separation_diagnostic_v4/deposited_panel_separations.csv \
  --output "$OUTPUT/analysis_all_20site"

write_atomic "$OUTPUT/status.txt" "complete"
