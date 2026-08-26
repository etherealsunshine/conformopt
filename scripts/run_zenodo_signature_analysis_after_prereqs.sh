#!/usr/bin/env bash
set -euo pipefail

AUDIT=/home/dev/qfit_unet_data/qfit_audit
QFIT_PHENIX="$AUDIT/zenodo_10936292_signature_qfit_direct_phenix_v1"
APRIME_PHENIX="$AUDIT/zenodo_10936292_signature_phenix_retry_v2"
PANEL="$AUDIT/zenodo_10936292_signature_panel_v5"
EXEC="$AUDIT/zenodo_10936292_signature_execution_v5"
OUT="$AUDIT/zenodo_10936292_signature_full_analysis_v1"
PY=/home/dev/qfit_unet_data/.venv-qfit-audit-gpu/bin/python
SCRIPT=/home/dev/workspace/scripts/run_zenodo_signature_full_analysis.py

while true; do
    q=$(/home/dev/qfit_unet_data/.venv/bin/python -c \
        'import json; print(json.load(open("/home/dev/qfit_unet_data/qfit_audit/zenodo_10936292_signature_qfit_direct_phenix_v1/progress.json"))["status"])' \
        2>/dev/null || true)
    a=$(/home/dev/qfit_unet_data/.venv/bin/python -c \
        'import json; print(json.load(open("/home/dev/qfit_unet_data/qfit_audit/zenodo_10936292_signature_phenix_retry_v2/progress.json"))["status"])' \
        2>/dev/null || true)
    if test "$q" = complete && test "$a" = complete; then
        break
    fi
    sleep 30
done

export PYTHONPATH=/home/dev/workspace:/home/dev/workspace/external/qfit-3.0/src:/home/dev/workspace/scripts
export CLEAN_D1_WIDER_ROOT="$PANEL/inputs/source"
export D1_MTZ_ROOT="$PANEL/inputs/map_mtz"
export LIBTBX_BUILD=/home/dev/qfit_unet_data/.venv-qfit-audit-gpu/lib/python3.11/site-packages/libtbx/core/share/cctbx
export LD_LIBRARY_PATH=/home/dev/qfit_unet_data/.venv-qfit-audit-gpu/lib:/home/dev/qfit_unet_data/.venv-qfit-audit-gpu/lib64:${LD_LIBRARY_PATH:-}

exec "$PY" -X faulthandler "$SCRIPT" \
    --panel-root "$PANEL" \
    --execution-root "$EXEC" \
    --qfit-phenix-root "$QFIT_PHENIX" \
    --aprime-phenix-root "$APRIME_PHENIX" \
    --output-root "$OUT" \
    --device cuda
