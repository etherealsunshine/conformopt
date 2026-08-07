#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/home/dev/qfit_unet_data/density_denoiser/sampleworks_3a1c_arg447_unet_vs_raw_v2}
RUNTIME=${RUNTIME:-/home/dev/qfit_unet_data/sampleworks_runtime}
PIXI="$RUNTIME/pixi/bin/pixi"
MANIFEST=/home/dev/workspace/external/sampleworks/pyproject.toml
CHECKPOINT="$RUNTIME/boltz/boltz2_conf.ckpt"
START="$ROOT/inputs/3A1C_B_ARG447_start_A_only.cif"
RESOLUTION=1.8500183413152225

mkdir -p "$ROOT/logs" "$ROOT/sampleworks" "$ROOT/pids"
export PIXI_HOME="$RUNTIME/pixi"
export TORCH_CUDA_ARCH_LIST=9.0

write_config() {
  /home/dev/qfit_unet_data/.venv/bin/python - "$ROOT/run_config.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "site": "3A1C_B_ARG447",
    "model": "boltz2",
    "guidance_type": "pure_guidance",
    "method": "X-RAY DIFFRACTION",
    "partial_diffusion_step": 120,
    "num_diffusion_steps": 200,
    "ensemble_size": 8,
    "recycling_steps": 3,
    "step_size": 0.1,
    "gradient_normalization": True,
    "augmentation": True,
    "align_to_input": True,
    "resolution_angstrom": 1.8500183413152225,
    "conditions": ["denoised", "raw"],
    "condition_order": ["denoised", "raw"],
    "checkpoint": "/home/dev/qfit_unet_data/sampleworks_runtime/boltz/boltz2_conf.ckpt",
    "input_manifest": str(path.parent / "inputs" / "input_manifest.json"),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

run_condition() {
  local condition=$1
  local density="$ROOT/inputs/3A1C_B_ARG447_${condition}_hybrid.ccp4"
  local output="$ROOT/sampleworks/$condition"
  local log="$ROOT/logs/sampleworks_${condition}.log"
  echo "sampling_${condition}" > "$ROOT/status.txt"
  mkdir -p "$output"
  "$PIXI" run --manifest-path "$MANIFEST" -e boltz sampleworks-guidance \
    --model boltz2 \
    --guidance-type pure_guidance \
    --protein "3A1C_B_ARG447_${condition}" \
    --model-checkpoint "$CHECKPOINT" \
    --structure "$START" \
    --density "$density" \
    --resolution "$RESOLUTION" \
    --output-dir "$output" \
    --log-path "$output/run.log" \
    --partial-diffusion-step 120 \
    --ensemble-size 8 \
    --recycling-steps 3 \
    --num-diffusion-steps 200 \
    --method "X-RAY DIFFRACTION" \
    --step-size 0.1 \
    --gradient-normalization \
    --augmentation \
    --align-to-input > "$log" 2>&1
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    echo complete > "$ROOT/pids/${condition}.status"
  else
    printf 'failed exit=%d\n' "$exit_code" > "$ROOT/pids/${condition}.status"
  fi
  return "$exit_code"
}

write_config
denoised_exit=0
raw_exit=0
run_condition denoised || denoised_exit=$?
run_condition raw || raw_exit=$?

if [[ $denoised_exit -eq 0 && $raw_exit -eq 0 ]]; then
  echo sampling_complete > "$ROOT/status.txt"
  exit 0
fi
printf 'sampling_failed denoised=%d raw=%d\n' "$denoised_exit" "$raw_exit" > "$ROOT/status.txt"
exit 1
