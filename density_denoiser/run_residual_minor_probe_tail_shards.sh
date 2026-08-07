#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_tail_residual_minor_probe_v1}

FROZEN_BASE=/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_v2_single_rule_v1
CONTROL_EIGHTEEN=/home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2
CONTROL_WATER2=/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1

sites=(
  1ZV8_E_ASN1
  2VFP_A_TYR417
  5Z8H_A_MET730
  7UO8_A_GLN53
  4C16_A_MET258
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
        "run_residual_minor_probe_tail_shards.sh"
    ),
    Path("/home/dev/workspace/scripts/analyze_residual_minor_probe.py"),
]
payload = {
    "diagnostic": "residual_minor_probe_tail_v1",
    "diagnostic_only": True,
    "production_changed": False,
    "metric_changed": False,
    "frozen_metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
    "sites": [
        "1ZV8_E_ASN1", "2VFP_A_TYR417", "5Z8H_A_MET730",
        "7UO8_A_GLN53", "4C16_A_MET258",
    ],
    "occupancy_modes": ["fixed_minor", "free_sigmoid"],
    "initialization": "deposited-A-centered N(0,1) radians; seed 41+start",
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
  local selection endpoint ensembles
  case "$site" in
    4C16_A_MET258)
      selection=$ORIGINAL_SELECTION
      endpoint="$CONTROL_EIGHTEEN/shards/original5/$site/synthetic/${site}_starts.csv"
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/original5/ensemble_strict_audit.csv"
      ;;
    2VFP_A_TYR417|7UO8_A_GLN53)
      selection=$EXPANDED_SELECTION
      endpoint="$CONTROL_WATER2/shards/$site/synthetic/${site}_starts.csv"
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/water2/ensemble_strict_audit.csv"
      ;;
    *)
      selection=$EXPANDED_SELECTION
      endpoint="$CONTROL_EIGHTEEN/shards/expanded15/$site/synthetic/${site}_starts.csv"
      ensembles="$FROZEN_BASE/analysis/metric_v3_protected_merge_sweep/0p5/expanded13/ensemble_strict_audit.csv"
      ;;
  esac
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

"$PYTHON" - "$OUTPUT" <<'PY'
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("sites/*/residual_minor_probe.csv")):
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))

expected = {
    "1ZV8_E_ASN1": 20,
    "2VFP_A_TYR417": 31,
    "5Z8H_A_MET730": 26,
    "7UO8_A_GLN53": 15,
    "4C16_A_MET258": 7,
}
counts = defaultdict(lambda: defaultdict(int))
for row in rows:
    site = row["site"]
    mode = row["occupancy_mode"]
    counts[site][mode] += 1
    counts[site][f"{mode}_recovered"] += (
        row["recovered_minor_lt_1A"].lower() == "true"
    )
for site, count in expected.items():
    for mode in ("fixed_minor", "free_sigmoid"):
        if counts[site][mode] != count:
            raise RuntimeError(
                f"{site} {mode}: expected {count}, got {counts[site][mode]}"
            )

summary_rows = []
for site in sorted(expected):
    for mode in ("fixed_minor", "free_sigmoid"):
        summary_rows.append({
            "site": site,
            "occupancy_mode": mode,
            "eligible": counts[site][mode],
            "recovered": counts[site][f"{mode}_recovered"],
            "recovery_rate": (
                counts[site][f"{mode}_recovered"] / counts[site][mode]
            ),
        })
for mode in ("fixed_minor", "free_sigmoid"):
    eligible = sum(counts[site][mode] for site in expected)
    recovered = sum(
        counts[site][f"{mode}_recovered"] for site in expected
    )
    summary_rows.append({
        "site": "TOTAL",
        "occupancy_mode": mode,
        "eligible": eligible,
        "recovered": recovered,
        "recovery_rate": recovered / eligible,
    })

def atomic_csv(path, values):
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=values[0].keys())
        writer.writeheader()
        writer.writerows(values)
        temporary = handle.name
    os.replace(temporary, path)

atomic_csv(root / "residual_minor_probe_all.csv", rows)
atomic_csv(root / "summary.csv", summary_rows)
payload = {
    "status": "complete",
    "diagnostic_only": True,
    "metric_changed": False,
    "eligible_starts": sum(expected.values()),
    "summary": summary_rows,
}
with tempfile.NamedTemporaryFile(
    "w", dir=root, delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, root / "summary.json")
PY

"$PYTHON" scripts/analyze_residual_minor_probe.py \
  --input "$OUTPUT/residual_minor_probe_all.csv" \
  --separations \
    /home/dev/qfit_unet_data/density_denoiser/deposited_panel_separation_diagnostic_v4/deposited_panel_separations.csv \
  --output "$OUTPUT/analysis"

write_atomic "$OUTPUT/status.txt" "complete"
