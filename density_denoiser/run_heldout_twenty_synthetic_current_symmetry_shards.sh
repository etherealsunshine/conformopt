#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
STALE_BASELINE_TABLE=${STALE_BASELINE_TABLE:-/home/dev/workspace/results/figures/twenty_site_synthetic_200step_composite.csv}
COMPLETED_CURRENT_TABLE=${COMPLETED_CURRENT_TABLE:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1/analysis/tmol_margin_sweep_v1/per_site_cascade_and_tmol_sweep.csv}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1}
FRAME=${FRAME:-crystal}
EXCLUDE_SITES=${EXCLUDE_SITES:-}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-20}
SKIP_COMPOSITE_ANALYSIS=${SKIP_COMPOSITE_ANALYSIS:-0}
GRID_RADIUS=${GRID_RADIUS:-4.0}
DENSITY_MASK_MODE=${DENSITY_MASK_MODE:-sphere}
REACHABLE_MASK_PADDING=${REACHABLE_MASK_PADDING:-1.0}
DENSITY_WEIGHT_MODE=${DENSITY_WEIGHT_MODE:-uniform}
TMOL_MAX_DELTA=${TMOL_MAX_DELTA:-0.0}
FOUND_OCCUPANCY=${FOUND_OCCUPANCY:-0.05}
MERGE_RMSD_THRESHOLD=${MERGE_RMSD_THRESHOLD:-0.0}
export GRID_RADIUS DENSITY_MASK_MODE REACHABLE_MASK_PADDING
export DENSITY_WEIGHT_MODE TMOL_MAX_DELTA FOUND_OCCUPANCY
export MERGE_RMSD_THRESHOLD

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

tuned_four_chi_sites=(
  3A1C_B_ARG447
  6H59_B_ARG144
  3NY7_B_LYS19
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

is_tuned_four_chi() {
  local query=$1
  local site
  for site in "${tuned_four_chi_sites[@]}"; do
    if [[ "$site" == "$query" ]]; then
      return 0
    fi
  done
  return 1
}

is_excluded() {
  local query=$1
  [[ " $EXCLUDE_SITES " == *" $query "* ]]
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/analysis" \
  "$OUTPUT/shards/original5" "$OUTPUT/shards/expanded15" \
  "$OUTPUT/calibration/original5" "$OUTPUT/calibration/expanded15" \
  "$OUTPUT/audit/original5" "$OUTPUT/audit/expanded15"

"$PYTHON" - "$OUTPUT/run_manifest.json" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from density_denoiser.clash_environment import (
    OPTIMIZER_PHYSICS_ENVIRONMENT_RULE,
)

output = Path(sys.argv[1])
source_files = [
    Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py"),
    Path("/home/dev/workspace/density_denoiser/clash_environment.py"),
    Path("/home/dev/workspace/density_denoiser/residue_geometry.py"),
    Path("/home/dev/workspace/density_denoiser/audit_five_site_endpoints.py"),
    Path("/home/dev/workspace/five_site_tmol_audit.py"),
]
payload = {
    "objective_change": (
        "single-rule altloc min-state soft environment with occupancy-weighted "
        "partial labeled waters; no barrier"
    ),
    "optimizer_physics_environment_rule": OPTIMIZER_PHYSICS_ENVIRONMENT_RULE,
    "comparison_constraint": (
        "reproduce the stale 673/653 composite schedules; do not interpret "
        "deltas as model progress because the optimizer objective changed"
    ),
    "target": "synthetic",
    "density_mask_mode": os.environ.get("DENSITY_MASK_MODE", "sphere"),
    "grid_radius_A": float(os.environ.get("GRID_RADIUS", "4.0")),
    "reachable_mask_padding_A": float(
        os.environ.get("REACHABLE_MASK_PADDING", "1.0")
    ),
    "density_weight_mode": os.environ.get(
        "DENSITY_WEIGHT_MODE", "uniform"
    ),
    "merge_rmsd_threshold_A": float(
        os.environ.get("MERGE_RMSD_THRESHOLD", "0.0")
    ),
    "starts_per_site": 50,
    "K": 4,
    "seed": 41,
    "one_to_three_chi_density_schedule": (
        "500 full-resolution steps, Adam lr 1.0"
    ),
    "tuned_four_chi_sites": [
        "3A1C_B_ARG447", "6H59_B_ARG144", "3NY7_B_LYS19"
    ],
    "tuned_four_chi_density_schedule": (
        "200 steps at 4A/lr1.0, 200 at 2A/lr0.1, "
        "200 full-resolution/lr0.01"
    ),
    "physics_schedule": "200 full-resolution steps, reset Adam, lr 0.1",
    "lambda_vdw": 1.0,
    "lambda_rot": 0.5,
    "lambda_clash": 5.0,
    "vdw_threshold_A": 3.0,
    "symmetry_soft_threshold_A": 2.5,
    "symmetry_hard_threshold_A": 2.0,
    "symmetry_barrier_scale": 0.0,
    "tmol_tolerance_is_post_hoc": True,
    "tmol_tolerance_promoted": False,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
PY

run_calibration() {
  local panel=$1
  local selection=$2
  "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --frame "$FRAME" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/calibration/$panel" \
    --target synthetic \
    --grid-radius "$GRID_RADIUS" \
    --density-mask-mode "$DENSITY_MASK_MODE" \
    --reachable-mask-padding "$REACHABLE_MASK_PADDING" \
    --density-weight-mode "$DENSITY_WEIGHT_MODE" \
    --n-steps 500 \
    --lr 1.0 \
    --per-residue-class-schedule \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw 1.0 \
    --lambda-rot 0.5 \
    --lambda-clash 5.0 \
    --vdw-threshold 3.0 \
    --clash-threshold 2.5 \
    --symmetry-hard-threshold 2.0 \
    --symmetry-barrier-buffer 0.25 \
    --symmetry-barrier-scale 0.0 \
    --seed 41 \
    --calibration-only \
    > "$OUTPUT/logs/calibration_$panel.log" 2>&1
}

write_status calibrating
run_calibration original5 "$ORIGINAL_SELECTION"
run_calibration expanded15 "$EXPANDED_SELECTION"

write_status optimizing
pids=()
labels=()

launch_site() {
  local panel=$1
  local selection=$2
  local site=$3
  local extra_schedule=()
  if is_tuned_four_chi "$site"; then
    extra_schedule=(--four-chi-stage-steps 200)
  fi
  nohup "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    --frame "$FRAME" \
    --site "$site" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT/shards/$panel/$site" \
    --target synthetic \
    --grid-radius "$GRID_RADIUS" \
    --density-mask-mode "$DENSITY_MASK_MODE" \
    --reachable-mask-padding "$REACHABLE_MASK_PADDING" \
    --density-weight-mode "$DENSITY_WEIGHT_MODE" \
    --K 4 \
    --n-starts 50 \
    --n-steps 500 \
    --lr 1.0 \
    --per-residue-class-schedule \
    "${extra_schedule[@]}" \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw 1.0 \
    --lambda-rot 0.5 \
    --lambda-clash 5.0 \
    --vdw-threshold 3.0 \
    --clash-threshold 2.5 \
    --symmetry-hard-threshold 2.0 \
    --symmetry-barrier-buffer 0.25 \
    --symmetry-barrier-scale 0.0 \
    --seed 41 \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  local pid=$!
  pids+=("$pid")
  labels+=("$site")
  write_atomic "$OUTPUT/pids/$site.pid" "$pid"
  write_atomic "$OUTPUT/pids/$site.status" running
}

for site in "${original_sites[@]}"; do
  if ! is_excluded "$site"; then
    launch_site original5 "$ORIGINAL_SELECTION" "$site"
  fi
done
for site in "${expanded_sites[@]}"; do
  if ! is_excluded "$site"; then
    launch_site expanded15 "$EXPANDED_SELECTION" "$site"
  fi
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

write_status verifying_rule_provenance
"$PYTHON" - "$OUTPUT" "$EXPECTED_SHARDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_shards = int(sys.argv[2])
manifest = json.loads((output / "run_manifest.json").read_text())
expected_rule = manifest["optimizer_physics_environment_rule"]
optimizer_path = Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py")
expected_hash = manifest["source_sha256"][str(optimizer_path)]
if hashlib.sha256(optimizer_path.read_bytes()).hexdigest() != expected_hash:
    raise RuntimeError("optimizer source changed after controller launch")
configs = sorted((output / "shards").glob("*/*/run_config.json"))
if len(configs) != expected_shards:
    raise RuntimeError(
        f"expected {expected_shards} shard run_config files, found {len(configs)}"
    )
for path in configs:
    config = json.loads(path.read_text())
    for key in ("density_mask_mode", "density_weight_mode"):
        if config[key] != manifest[key]:
            raise RuntimeError(f"{key} mismatch in {path}")
    if config["grid_radius"] != manifest["grid_radius_A"]:
        raise RuntimeError(f"grid radius mismatch in {path}")
    if (
        config["reachable_mask_padding"]
        != manifest["reachable_mask_padding_A"]
    ):
        raise RuntimeError(f"reachable padding mismatch in {path}")
rules = {
    json.loads(path.read_text()).get("optimizer_physics_environment_rule")
    for path in configs
}
if rules != {expected_rule}:
    raise RuntimeError(f"mixed optimizer environment rules: {sorted(rules)}")
PY

write_status geometry_audit
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
    --merge-rmsd-threshold "$MERGE_RMSD_THRESHOLD" \
    > "$OUTPUT/logs/geometry_audit_$panel.log" 2>&1
done

write_status tmol_calibration
for panel in original5 expanded15; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --calibrate-only \
    > "$OUTPUT/logs/tmol_calibration_$panel.log" 2>&1
done

write_status tmol_audit
for panel in original5 expanded15; do
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --resume \
    > "$OUTPUT/logs/tmol_audit_$panel.log" 2>&1
done

write_status summarizing_zero_tolerance
for panel in original5 expanded15; do
  "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
    --audit-root "$OUTPUT/audit/$panel" \
    --tmol-max-delta "$TMOL_MAX_DELTA" \
    --found-occupancy "$FOUND_OCCUPANCY" \
    > "$OUTPUT/logs/strict_summary_$panel.log" 2>&1
done

if [[ "$SKIP_COMPOSITE_ANALYSIS" == 1 ]]; then
  write_status complete
  exit 0
fi

write_status tmol_margin_sweep
"$PYTHON" -m density_denoiser.summarize_tmol_margin_sweep \
  --audit-root "$OUTPUT/audit/original5" \
  --audit-root "$OUTPUT/audit/expanded15" \
  --stale-baseline-table "$STALE_BASELINE_TABLE" \
  --comparison-table "$COMPLETED_CURRENT_TABLE" \
  --comparison-label completed_current_728 \
  --output "$OUTPUT/analysis/tmol_margin_sweep_v1" \
  > "$OUTPUT/logs/tmol_margin_sweep.log" 2>&1

write_status assignment_diagnostic
"$PYTHON" -m density_denoiser.analyze_tmol_assignments \
  --strict-table "$OUTPUT/audit/original5/active_conformer_strict_audit.csv" \
  --strict-table "$OUTPUT/audit/expanded15/active_conformer_strict_audit.csv" \
  --site 3K8W_A_SER337 \
  --site 8Q6Q_B_ASP81 \
  --output "$OUTPUT/analysis/3k8w_8q6q_assignment_tmol_v1" \
  > "$OUTPUT/logs/assignment_diagnostic.log" 2>&1

write_status complete
