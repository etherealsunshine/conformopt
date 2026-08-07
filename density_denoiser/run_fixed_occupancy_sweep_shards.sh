#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_fixed_occupancy_sweep_v1}
FRAME=${FRAME:-crystal}
FROZEN_METRIC=qfit-synth20-merge050-one-to-one-tmol044-v3
FROZEN_BASELINE_STRICT=626
FROZEN_BASELINE_FOUND=742
FREEZE_ARMS=(100 200 300)

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

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing sweep: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p "$OUTPUT/logs" "$OUTPUT/calibration"

"$PYTHON" - "$OUTPUT/run_manifest.json" \
  "$FROZEN_METRIC" "$FROZEN_BASELINE_STRICT" "$FROZEN_BASELINE_FOUND" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

output = Path(sys.argv[1])
source_files = [
    Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py"),
    Path("/home/dev/workspace/density_denoiser/clash_environment.py"),
    Path("/home/dev/workspace/density_denoiser/residue_geometry.py"),
    Path("/home/dev/workspace/density_denoiser/audit_five_site_endpoints.py"),
    Path("/home/dev/workspace/density_denoiser/summarize_endpoint_audit.py"),
    Path("/home/dev/workspace/scripts/five_site_tmol_audit.py"),
    Path(
        "/home/dev/workspace/density_denoiser/"
        "run_fixed_occupancy_sweep_shards.sh"
    ),
]
payload = {
    "experiment": "stage1_fixed_occupancy_prefix_sweep_v1",
    "single_factor": "fixed_occupancy_steps",
    "arms": [100, 200, 300],
    "control": {
        "fixed_occupancy_steps": 0,
        "reused_frozen_run": True,
        "metric": sys.argv[2],
        "strict": int(sys.argv[3]),
        "both_found": int(sys.argv[4]),
    },
    "occupancy_freeze_implementation": "adam_group_lr_zero_warm_state_v1",
    "seed": 41,
    "seed_per_shard_start": "41 + start",
    "starts_per_site": 50,
    "K": 4,
    "one_to_three_chi_density_schedule": (
        "500 full-resolution steps, Adam chi lr 1.0"
    ),
    "tuned_four_chi_sites": [
        "3A1C_B_ARG447", "6H59_B_ARG144", "3NY7_B_LYS19"
    ],
    "tuned_four_chi_density_schedule": (
        "200 steps at 4A/lr1.0, 200 at 2A/lr0.1, "
        "200 full-resolution/lr0.01; Adam reset at schedule boundaries"
    ),
    "physics_schedule": "200 full-resolution steps, reset Adam, lr 0.1",
    "physics_weights": {"vdw": 1.0, "rotamer": 0.5, "symmetry": 5.0},
    "metric_unchanged": True,
    "frozen_metric": sys.argv[2],
    "merge_rmsd_threshold_A": 0.5,
    "found_occupancy_threshold": 0.10,
    "tmol_tolerance": 0.44,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    },
}
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
PY

common_args=(
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
for panel in original5 expanded15; do
  if [[ "$panel" == original5 ]]; then
    selection=$ORIGINAL_SELECTION
  else
    selection=$EXPANDED_SELECTION
  fi
  "$PYTHON" -m density_denoiser.five_site_optimizer \
    --selection "$selection" \
    "${common_args[@]}" \
    --output "$OUTPUT/calibration/$panel" \
    --calibration-only \
    > "$OUTPUT/logs/calibration_$panel.log" 2>&1
done

run_arm() {
  local fixed_steps=$1
  local arm="$OUTPUT/fixed_${fixed_steps}"
  mkdir -p \
    "$arm/logs" "$arm/pids" \
    "$arm/shards/original5" "$arm/shards/expanded15" \
    "$arm/audit/original5" "$arm/audit/expanded15"
  write_atomic "$arm/status.txt" optimizing
  write_status "fixed_${fixed_steps}_optimizing"

  local pids=()
  local labels=()
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
      --site "$site" \
      "${common_args[@]}" \
      "${extra_schedule[@]}" \
      --output "$arm/shards/$panel/$site" \
      --K 4 \
      --n-starts 50 \
      --fixed-occupancy-steps "$fixed_steps" \
      --record-stage1-trajectories \
      > "$arm/logs/$site.log" 2>&1 < /dev/null &
    local pid=$!
    pids+=("$pid")
    labels+=("$site")
    write_atomic "$arm/pids/$site.pid" "$pid"
    write_atomic "$arm/pids/$site.status" running
  }

  local site
  for site in "${original_sites[@]}"; do
    launch_site original5 "$ORIGINAL_SELECTION" "$site"
  done
  for site in "${expanded_sites[@]}"; do
    launch_site expanded15 "$EXPANDED_SELECTION" "$site"
  done

  local failed=0
  local index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      write_atomic "$arm/pids/${labels[$index]}.status" complete
    else
      write_atomic "$arm/pids/${labels[$index]}.status" failed
      failed=1
    fi
  done
  if (( failed )); then
    write_atomic "$arm/status.txt" optimizer_failed
    write_status "fixed_${fixed_steps}_optimizer_failed"
    return 1
  fi

  write_atomic "$arm/status.txt" verifying_provenance
  "$PYTHON" - "$OUTPUT/run_manifest.json" "$arm" "$fixed_steps" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
arm = Path(sys.argv[2])
fixed_steps = int(sys.argv[3])
optimizer = Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py")
expected_hash = manifest["source_sha256"][str(optimizer)]
if hashlib.sha256(optimizer.read_bytes()).hexdigest() != expected_hash:
    raise RuntimeError("optimizer source changed after sweep launch")
configs = sorted((arm / "shards").glob("*/*/run_config.json"))
if len(configs) != 20:
    raise RuntimeError(f"expected 20 shard configs, found {len(configs)}")
for path in configs:
    config = json.loads(path.read_text())
    if config["fixed_occupancy_steps"] != fixed_steps:
        raise RuntimeError(f"wrong fixed steps in {path}")
    if config["seed"] != 41 or config["K"] != 4 or config["n_starts"] != 50:
        raise RuntimeError(f"frozen launch mismatch in {path}")
    if not config["record_stage1_trajectories"]:
        raise RuntimeError(f"trajectory logging disabled in {path}")
    if (
        config["occupancy_freeze_implementation"]
        != "adam_group_lr_zero_warm_state_v1"
    ):
        raise RuntimeError(f"wrong freeze implementation in {path}")
PY

  local panel
  for panel in original5 expanded15; do
    local selection
    if [[ "$panel" == original5 ]]; then
      selection=$ORIGINAL_SELECTION
    else
      selection=$EXPANDED_SELECTION
    fi

    write_atomic "$arm/status.txt" "geometry_audit_$panel"
    "$PYTHON" -m density_denoiser.audit_five_site_endpoints \
      --selection "$selection" \
      --results-root "$arm/shards/$panel" \
      --output "$arm/audit/$panel" \
      --target synthetic \
      --merge-rmsd-threshold 0.5 \
      > "$arm/logs/geometry_audit_$panel.log" 2>&1

    write_atomic "$arm/status.txt" "tmol_calibration_$panel"
    "$TMOL_PYTHON" five_site_tmol_audit.py \
      --input-root "$arm/audit/$panel" \
      --output "$arm/audit/$panel" \
      --calibrate-only \
      > "$arm/logs/tmol_calibration_$panel.log" 2>&1

    write_atomic "$arm/status.txt" "tmol_audit_$panel"
    "$TMOL_PYTHON" five_site_tmol_audit.py \
      --input-root "$arm/audit/$panel" \
      --output "$arm/audit/$panel" \
      --resume \
      > "$arm/logs/tmol_audit_$panel.log" 2>&1

    write_atomic "$arm/status.txt" "summarizing_$panel"
    "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
      --audit-root "$arm/audit/$panel" \
      --tmol-max-delta 0.44 \
      --found-occupancy 0.10 \
      > "$arm/logs/strict_summary_$panel.log" 2>&1
  done

  write_atomic "$arm/status.txt" complete
}

for fixed_steps in "${FREEZE_ARMS[@]}"; do
  run_arm "$fixed_steps"
done

write_status complete
