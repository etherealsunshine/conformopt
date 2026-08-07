#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_containing_mask_sweep_v1}
FROZEN_METRIC=qfit-synth20-merge050-one-to-one-tmol044-v3
CONTROLLER=/home/dev/workspace/density_denoiser/run_heldout_twenty_synthetic_current_symmetry_shards.sh

write_atomic() {
  local path=$1
  local value=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$path"
}

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing sweep: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs"

"$PYTHON" - "$OUTPUT/run_manifest.json" "$FROZEN_METRIC" <<'PY'
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
        "run_heldout_twenty_synthetic_current_symmetry_shards.sh"
    ),
    Path(
        "/home/dev/workspace/density_denoiser/"
        "run_density_mask_sweep_shards.sh"
    ),
    Path("/home/dev/workspace/density_denoiser/residue_geometry.py"),
]
payload = {
    "experiment": "stage1_containing_density_mask_sweep_v1",
    "single_factor": "containing mask with uniform or variance voxel weights",
    "frozen_metric": sys.argv[2],
    "control": {
        "reused": True,
        "found": 742,
        "strict": 626,
        "minor_major_misses": [142, 45],
    },
    "seed": 41,
    "seed_per_start": "41 + start",
    "arms": {
        "containing_uniform": {
            "density_mask_mode": "containing_volume",
            "reachable_mask_padding_A": 1.0,
            "density_weight_mode": "uniform",
            "footprint": (
                "union of production canonical-center atom positions and "
                "deposited A/B atoms, padded by 1 A"
            ),
        },
        "containing_variance_weighted": {
            "density_mask_mode": "containing_volume",
            "reachable_mask_padding_A": 1.0,
            "density_weight_mode": "reachable_variance",
            "weight_normalization": "mean one over mask voxels",
            "weight_ground_truth_coordinates_used": False,
            "weight_state_source": "production marginal canonical center tuples",
        },
    },
    "common": {
        "starts_per_site": 50,
        "K": 4,
        "tmol_tolerance": 0.44,
        "found_occupancy_threshold": 0.10,
        "merge_rmsd_threshold_A": 0.5,
        "normalization": "masked z-score unchanged",
    },
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

run_arm() {
  local label=$1
  local mask_mode=$2
  local radius=$3
  local weight_mode=$4
  local arm="$OUTPUT/$label"
  write_atomic "$OUTPUT/status.txt" "${label}_running"
  env \
    OUTPUT="$arm" \
    DENSITY_MASK_MODE="$mask_mode" \
    GRID_RADIUS="$radius" \
    REACHABLE_MASK_PADDING=1.0 \
    DENSITY_WEIGHT_MODE="$weight_mode" \
    TMOL_MAX_DELTA=0.44 \
    FOUND_OCCUPANCY=0.10 \
    MERGE_RMSD_THRESHOLD=0.5 \
    SKIP_COMPOSITE_ANALYSIS=1 \
    EXPECTED_SHARDS=20 \
    bash "$CONTROLLER" \
    > "$OUTPUT/logs/$label.log" 2>&1
}

run_arm containing_uniform containing_volume 4.0 uniform
run_arm containing_variance_weighted containing_volume 4.0 reachable_variance

write_atomic "$OUTPUT/status.txt" complete
