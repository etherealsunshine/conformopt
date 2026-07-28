#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
SELECTION=${SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1}
FRAME=${FRAME:-crystal}
RULE=2026-07-24-altloc-minstate-water-minstate-v2

sites=(
  7UO8_A_GLN53
  2VFP_A_TYR417
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
  "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards" \
  "$OUTPUT/calibration" "$OUTPUT/audit" "$OUTPUT/analysis"

"$PYTHON" - "$SELECTION" "$OUTPUT/selection.json" "$OUTPUT/run_manifest.json" "$RULE" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

source_selection = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
rule = sys.argv[4]
wanted = {"7UO8_A_GLN53", "2VFP_A_TYR417"}
selection = json.loads(source_selection.read_text())
selection["sites"] = [row for row in selection["sites"] if row["key"] in wanted]
if {row["key"] for row in selection["sites"]} != wanted:
    raise RuntimeError("Could not resolve both requested sites in the selection")

def atomic_json(path: Path, payload: object) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)

atomic_json(subset_path, selection)
source_files = [
    Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py"),
    Path("/home/dev/workspace/density_denoiser/clash_environment.py"),
    Path("/home/dev/workspace/density_denoiser/residue_geometry.py"),
    Path("/home/dev/workspace/density_denoiser/audit_five_site_endpoints.py"),
    Path("/home/dev/workspace/five_site_tmol_audit.py"),
]
atomic_json(
    manifest_path,
    {
        "single_factor": (
            "extend min-over-altloc-state soft physics to labeled waters"
        ),
        "optimizer_physics_environment_rule": rule,
        "sites": sorted(wanted),
        "target": "synthetic",
        "starts_per_site": 50,
        "K": 4,
        "seed": 41,
        "density_schedule": "500 full-resolution steps, Adam lr 1.0",
        "physics_schedule": "200 full-resolution steps, reset Adam, lr 0.1",
        "lambda_vdw": 1.0,
        "lambda_rot": 0.5,
        "lambda_clash": 5.0,
        "vdw_threshold_A": 3.0,
        "symmetry_soft_threshold_A": 2.5,
        "symmetry_hard_threshold_A": 2.0,
        "symmetry_barrier_scale": 0.0,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    },
)
PY

common_args=(
  --selection "$OUTPUT/selection.json"
  --frame "$FRAME"
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
)

write_status calibrating
"$PYTHON" -m density_denoiser.five_site_optimizer \
  "${common_args[@]}" \
  --output "$OUTPUT/calibration" \
  --calibration-only \
  > "$OUTPUT/logs/calibration.log" 2>&1

write_status optimizing
pids=()
for site in "${sites[@]}"; do
  nohup "$PYTHON" -m density_denoiser.five_site_optimizer \
    "${common_args[@]}" \
    --site "$site" \
    --output "$OUTPUT/shards/$site" \
    --K 4 \
    --n-starts 50 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  write_atomic "$OUTPUT/pids/$site.pid" "$pid"
  write_atomic "$OUTPUT/pids/$site.status" running
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    write_atomic "$OUTPUT/pids/${sites[$index]}.status" complete
  else
    write_atomic "$OUTPUT/pids/${sites[$index]}.status" failed
    failed=1
  fi
done
if (( failed )); then
  write_status optimizer_failed
  exit 1
fi

write_status geometry_audit
"$PYTHON" -m density_denoiser.audit_five_site_endpoints \
  --selection "$OUTPUT/selection.json" \
  --results-root "$OUTPUT/shards" \
  --output "$OUTPUT/audit" \
  --target synthetic \
  > "$OUTPUT/logs/geometry_audit.log" 2>&1

write_status tmol_calibration
"$TMOL_PYTHON" five_site_tmol_audit.py \
  --input-root "$OUTPUT/audit" \
  --output "$OUTPUT/audit" \
  --calibrate-only \
  > "$OUTPUT/logs/tmol_calibration.log" 2>&1

write_status tmol_audit
"$TMOL_PYTHON" five_site_tmol_audit.py \
  --input-root "$OUTPUT/audit" \
  --output "$OUTPUT/audit" \
  --resume \
  > "$OUTPUT/logs/tmol_audit.log" 2>&1

write_status summarizing
"$PYTHON" -m density_denoiser.summarize_endpoint_audit \
  --audit-root "$OUTPUT/audit" \
  --tmol-max-delta 0.0 \
  > "$OUTPUT/logs/strict_summary.log" 2>&1

write_status complete
