#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dev/qfit_unet_data/qfit_audit/clean_d1_mirror_sweep_v1
SRC=/home/dev/workspace
PY=/home/dev/qfit_unet_data/.venv-qfit-audit/bin/python
export LD_LIBRARY_PATH=/home/dev/qfit_unet_data/.venv-qfit-audit/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/dev/workspace:${PYTHONPATH:-}
MANIFEST=$SRC/data/clean_d1_benchmark_manifest_v4.json
STARTS=/home/dev/qfit_unet_data/qfit_audit/clean_d1_neutral_6zwk_v1
FLIP_ROOT=$SRC/data/qfit_2015_s004

write_status() {
    local value=$1
    printf '%s\n' "$value" > "$BASE/status.txt.tmp"
    mv "$BASE/status.txt.tmp" "$BASE/status.txt"
}

cd "$SRC"
mkdir -p "$BASE/logs"
write_status running

for spec in "0001 0.001" "0010 0.01" "0030 0.03"; do
    read -r tag eta <<< "$spec"
    out=/home/dev/qfit_unet_data/qfit_audit/clean_d1_benchmark_6zwk_mirror_eta${tag}_v1
    write_status "running_eta${tag}"
    "$PY" "$SRC/scripts/run_clean_d1_recovery.py" \
        --manifest "$MANIFEST" \
        --starts "$STARTS" \
        --output "$out" \
        --flip-root "$FLIP_ROOT" \
        --site 6ZWK_B_PHE47 \
        --device cpu \
        --occupancy-scheme mirror \
        --mirror-eta "$eta" \
        > "$BASE/logs/eta${tag}.log" 2>&1
    write_status "complete_eta${tag}"
done

write_status complete
