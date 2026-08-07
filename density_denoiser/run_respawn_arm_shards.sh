#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
TMOL_PYTHON=${TMOL_PYTHON:-/home/dev/qfit_unet_data/.venv-tmol/bin/python}
ORIGINAL_SELECTION=${ORIGINAL_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json}
EXPANDED_SELECTION=${EXPANDED_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
CHECKPOINT=${CHECKPOINT:-/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_respawn_R1_v1}
ARM_LABEL=${ARM_LABEL:-R1}
RESPAWN_CADENCE=${RESPAWN_CADENCE:-100}
RESPAWN_MERGE_RMSD=${RESPAWN_MERGE_RMSD:-0.5}
CHI_NOISE_INITIAL_DEGREES=${CHI_NOISE_INITIAL_DEGREES:-0}
RECORD_STAGE1_TRAJECTORIES=${RECORD_STAGE1_TRAJECTORIES:-0}
FROZEN_METRIC=qfit-synth20-merge050-one-to-one-tmol044-v3

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

if [[ "$RESPAWN_CADENCE" != 0 && "$CHI_NOISE_INITIAL_DEGREES" != 0 ]]; then
  printf 'Respawn and chi noise are mutually exclusive in this launcher\n' >&2
  exit 2
fi

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing intervention arm: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/calibration" \
  "$OUTPUT/shards/original5" "$OUTPUT/shards/expanded15" \
  "$OUTPUT/audit/original5" "$OUTPUT/audit/expanded15"

export ARM_LABEL RESPAWN_CADENCE RESPAWN_MERGE_RMSD
export CHI_NOISE_INITIAL_DEGREES RECORD_STAGE1_TRAJECTORIES FROZEN_METRIC
"$PYTHON" - "$OUTPUT/run_manifest.json" <<'PY'
import hashlib
import json
import os
import tempfile
from pathlib import Path

output = Path(__import__("sys").argv[1])
source_files = [
    Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py"),
    Path("/home/dev/workspace/density_denoiser/clash_environment.py"),
    Path("/home/dev/workspace/density_denoiser/residue_geometry.py"),
    Path("/home/dev/workspace/density_denoiser/audit_five_site_endpoints.py"),
    Path("/home/dev/workspace/density_denoiser/summarize_endpoint_audit.py"),
    Path("/home/dev/workspace/five_site_tmol_audit.py"),
    Path("/home/dev/workspace/density_denoiser/run_respawn_arm_shards.sh"),
]
payload = {
    "experiment": (
        "stage1_annealed_chi_langevin_v1"
        if float(os.environ["CHI_NOISE_INITIAL_DEGREES"]) > 0
        else "stage1_merge_and_residual_peak_respawn_v1"
    ),
    "arm": os.environ["ARM_LABEL"],
    "single_factor": (
        "linearly annealed chi-only Gaussian noise after each Stage-1 Adam step"
        if float(os.environ["CHI_NOISE_INITIAL_DEGREES"]) > 0
        else "Stage-1 merge-and-respawn placement"
    ),
    "stage1_chi_noise_initial_sd_degrees": float(
        os.environ["CHI_NOISE_INITIAL_DEGREES"]
    ),
    "stage1_chi_noise_schedule": (
        "linear_first_step_initial_to_final_step_zero"
    ),
    "record_stage1_trajectories": (
        os.environ["RECORD_STAGE1_TRAJECTORIES"] == "1"
    ),
    "respawn_cadence": int(os.environ["RESPAWN_CADENCE"]),
    "respawn_merge_rmsd_A": float(os.environ["RESPAWN_MERGE_RMSD"]),
    "control": {
        "reused": True,
        "rerun": False,
        "metric": os.environ["FROZEN_METRIC"],
        "cascade": [742, 714, 710, 710, 710, 626],
        "raw_greedy_minor_major_misses": [142, 45],
    },
    "seed": 41,
    "seed_per_shard_start": "41 + start",
    "starts_per_site": 50,
    "K": 4,
    "merge_trigger": "closest pair symmetry-aware conventional RMSD",
    "merge_pair_eligibility": (
        "both slots active above frozen nontrivial occupancy 0.05"
    ),
    "gram_diagnostic_threshold": 100.0,
    "gram_changes_trigger": False,
    "respawn_peak": "argmax(target - rendered) on current masked Stage-1 grid",
    "torsion_inversion": {
        "method": "parallel per-heavy-atom local Adam on chi",
        "steps": 50,
        "learning_rate": 0.1,
        "reached_threshold_A": 0.5,
    },
    "merge_bookkeeping_occupancy_floor": 1e-6,
    "respawn_occupancy_rule": (
        "after measuring the bookkeeping merge, reuse the freed slot's "
        "pre-merge occupancy, transferred back from the keeper"
    ),
    "adam_reset": (
        "freed chi moment row and freed+keeper occupancy-logit moment "
        "elements; shared Adam scalar step retained"
    ),
    "terminal_stage1_check_excluded": True,
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
    "physics_weights": {"vdw": 1.0, "rotamer": 0.5, "symmetry": 5.0},
    "metric_changed": False,
    "audit_merge_rmsd_A": 0.5,
    "found_occupancy_threshold": 0.10,
    "tmol_tolerance": 0.44,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    },
}
with tempfile.NamedTemporaryFile(
    "w", dir=output.parent, delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
Path(temporary).replace(output)
PY

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
  --stage1-chi-noise-initial-degrees "$CHI_NOISE_INITIAL_DEGREES"
)
trajectory_args=()
if [[ "$RECORD_STAGE1_TRAJECTORIES" == 1 ]]; then
  trajectory_args=(--record-stage1-trajectories)
fi

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
    "${trajectory_args[@]}" \
    --output "$OUTPUT/calibration/$panel" \
    --calibration-only \
    > "$OUTPUT/logs/calibration_$panel.log" 2>&1
done

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
    --site "$site" \
    "${common_args[@]}" \
    "${extra_schedule[@]}" \
    "${trajectory_args[@]}" \
    --output "$OUTPUT/shards/$panel/$site" \
    --K 4 \
    --n-starts 50 \
    --respawn-cadence "$RESPAWN_CADENCE" \
    --respawn-merge-rmsd "$RESPAWN_MERGE_RMSD" \
    > "$OUTPUT/logs/$site.log" 2>&1 < /dev/null &
  local pid=$!
  pids+=("$pid")
  labels+=("$site")
  write_atomic "$OUTPUT/pids/$site.pid" "$pid"
  write_atomic "$OUTPUT/pids/$site.status" running
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

write_status verifying_provenance
"$PYTHON" - "$OUTPUT" "$RESPAWN_CADENCE" "$RESPAWN_MERGE_RMSD" \
  "$CHI_NOISE_INITIAL_DEGREES" "$RECORD_STAGE1_TRAJECTORIES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
cadence = int(sys.argv[2])
threshold = float(sys.argv[3])
chi_noise = float(sys.argv[4])
record_trajectories = sys.argv[5] == "1"
manifest = json.loads((output / "run_manifest.json").read_text())
optimizer = Path("/home/dev/workspace/density_denoiser/five_site_optimizer.py")
if hashlib.sha256(optimizer.read_bytes()).hexdigest() != (
    manifest["source_sha256"][str(optimizer)]
):
    raise RuntimeError("optimizer source changed after controller launch")
configs = sorted((output / "shards").glob("*/*/run_config.json"))
if len(configs) != 20:
    raise RuntimeError(f"expected 20 shard configs, found {len(configs)}")
for path in configs:
    config = json.loads(path.read_text())
    expected = {
        "seed": 41,
        "K": 4,
        "n_starts": 50,
        "respawn_cadence": cadence,
        "respawn_merge_rmsd": threshold,
        "stage1_chi_noise_initial_degrees": chi_noise,
        "record_stage1_trajectories": record_trajectories,
        "fixed_occupancy_steps": 0,
        "density_mask_mode": "sphere",
        "density_weight_mode": "uniform",
        "grid_radius": 4.0,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(
                f"{key} mismatch in {path}: {config[key]} != {value}"
            )
PY

for panel in original5 expanded15; do
  if [[ "$panel" == original5 ]]; then
    selection=$ORIGINAL_SELECTION
  else
    selection=$EXPANDED_SELECTION
  fi
  write_status "geometry_audit_$panel"
  "$PYTHON" -m density_denoiser.audit_five_site_endpoints \
    --selection "$selection" \
    --results-root "$OUTPUT/shards/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --target synthetic \
    --merge-rmsd-threshold 0.5 \
    > "$OUTPUT/logs/geometry_audit_$panel.log" 2>&1

  write_status "tmol_calibration_$panel"
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --calibrate-only \
    > "$OUTPUT/logs/tmol_calibration_$panel.log" 2>&1

  write_status "tmol_audit_$panel"
  "$TMOL_PYTHON" five_site_tmol_audit.py \
    --input-root "$OUTPUT/audit/$panel" \
    --output "$OUTPUT/audit/$panel" \
    --resume \
    > "$OUTPUT/logs/tmol_audit_$panel.log" 2>&1

  write_status "summarizing_$panel"
  "$PYTHON" -m density_denoiser.summarize_endpoint_audit \
    --audit-root "$OUTPUT/audit/$panel" \
    --tmol-max-delta 0.44 \
    --found-occupancy 0.10 \
    > "$OUTPUT/logs/strict_summary_$panel.log" 2>&1
done

write_status complete
