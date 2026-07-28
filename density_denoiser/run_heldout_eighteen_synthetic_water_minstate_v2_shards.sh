#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/dev/qfit_unet_data/.venv/bin/python}
SOURCE_SELECTION=${SOURCE_SELECTION:-/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json}
export OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2}
SUBSET_SELECTION="$OUTPUT/expanded_selection_13.json"

if [[ -e "$OUTPUT/status.txt" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT"

"$PYTHON" - "$SOURCE_SELECTION" "$SUBSET_SELECTION" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
excluded = {"7UO8_A_GLN53", "2VFP_A_TYR417"}
payload = json.loads(source.read_text())
payload["sites"] = [
    row for row in payload["sites"] if row["key"] not in excluded
]
if len(payload["sites"]) != 13:
    raise RuntimeError("expected 13 expanded-panel sites after exclusion")
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, output)
PY

export EXPANDED_SELECTION="$SUBSET_SELECTION"
export EXCLUDE_SITES="7UO8_A_GLN53 2VFP_A_TYR417"
export EXPECTED_SHARDS=18
export SKIP_COMPOSITE_ANALYSIS=1
exec bash density_denoiser/run_heldout_twenty_synthetic_current_symmetry_shards.sh
