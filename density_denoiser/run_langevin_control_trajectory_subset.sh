#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/langevin_control_trajectory_subset_v1}
CONTROL_BASELINE_ROOT=${CONTROL_BASELINE_ROOT:-/home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2}
CONTROL_REPLACEMENT_ROOT=${CONTROL_REPLACEMENT_ROOT:-/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1}

sites=(
  "original5|$ORIGINAL_SELECTION|4C16_A_MET258"
  "expanded15|$EXPANDED_SELECTION|1ZV8_E_ASN1"
  "expanded15|$EXPANDED_SELECTION|7UO8_A_GLN53"
  "expanded15|$EXPANDED_SELECTION|2VFP_A_TYR417"
  "expanded15|$EXPANDED_SELECTION|5Z8H_A_MET730"
)

write_atomic() {
  local path=$1
  local value=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite control trajectory subset: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards"
write_atomic "$OUTPUT/status.txt" optimizing

common_args=(
  --frame crystal
  --checkpoint "$CHECKPOINT"
  --target synthetic
  --n-steps 500
  --lr 1.0
  --per-residue-class-schedule
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
  --stage1-chi-noise-initial-degrees 0
  --record-stage1-trajectories
)

pids=()
labels=()
for item in "${sites[@]}"; do
  IFS='|' read -r panel selection site <<< "$item"
  nohup "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --site "$site" \
    "${common_args[@]}" \
    --output "$OUTPUT/shards/$panel/$site" \
    --K 4 \
    --n-starts 10 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  pids+=("$!")
  labels+=("$site")
  write_atomic "$OUTPUT/pids/$site.pid" "$!"
  write_atomic "$OUTPUT/pids/$site.status" running
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
  write_atomic "$OUTPUT/status.txt" optimizer_failed
  exit 1
fi

"$PYTHON" - "$OUTPUT" "$CONTROL_BASELINE_ROOT" \
  "$CONTROL_REPLACEMENT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
control_roots = [Path(sys.argv[2]), Path(sys.argv[3])]
rows = []
for path in sorted(root.glob("shards/*/*/synthetic/*_starts.csv")):
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
trajectories = list(root.glob("shards/*/*/trajectories/*.npz"))
if len(rows) != 50 or len(trajectories) != 50:
    raise RuntimeError(
        f"expected 50 rows/trajectories, got {len(rows)}/{len(trajectories)}"
    )
control = {}
for control_root in control_roots:
    for path in control_root.glob("shards/**/synthetic/*_starts.csv"):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                control[(row["site"], int(row["start"]))] = row
for row in rows:
    reference = control[(row["site"], int(row["start"]))]
    mismatches = [
        key for key in set(row) & set(reference)
        if row[key] != reference[key]
    ]
    if mismatches:
        raise RuntimeError(
            f"zero-noise parity failed for {row['site']} start "
            f"{row['start']}: {mismatches}"
        )
for path in root.glob("shards/*/*/run_config.json"):
    config = json.loads(path.read_text())
    expected = {
        "seed": 41,
        "n_starts": 10,
        "stage1_chi_noise_initial_degrees": 0.0,
        "record_stage1_trajectories": True,
        "respawn_cadence": 0,
        "fixed_occupancy_steps": 0,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"{key} mismatch in {path}")
PY

write_atomic "$OUTPUT/status.txt" complete
