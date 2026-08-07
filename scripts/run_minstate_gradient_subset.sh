#!/usr/bin/env bash
set -euo pipefail

ROOT=${OUTPUT:?set OUTPUT to a new analysis directory}
PY=/home/dev/qfit_unet_data/.venv/bin/python
CHECKPOINT=/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt
WRAPPER=/home/dev/workspace/scripts/run_minstate_flip_trace.py

mkdir -p "$ROOT/logs" "$ROOT/pids" "$ROOT/shards"
printf 'launching\n' > "$ROOT/status.txt"

sites=(
  2VFP_A_TYR417
  5KWB_A_PHE591
  6H59_B_ARG144
  3GMI_A_GLU5
  7UO8_A_GLN53
  3A1C_B_ARG447
)

selection_for() {
  case "$1" in
    2VFP_A_TYR417|7UO8_A_GLN53)
      printf '%s\n' /home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1/selection.json
      ;;
    3A1C_B_ARG447|6H59_B_ARG144)
      printf '%s\n' /home/dev/qfit_unet_data/density_denoiser/heldout_five_site_selection_v3/selection.json
      ;;
    *)
      printf '%s\n' /home/dev/qfit_unet_data/density_denoiser/heldout_eighteen_synthetic_water_minstate_v2/expanded_selection_13.json
      ;;
  esac
}

four_chi_steps_for() {
  case "$1" in
    3A1C_B_ARG447|6H59_B_ARG144) printf '200\n' ;;
    *) printf '100\n' ;;
  esac
}

for site in "${sites[@]}"; do
  selection=$(selection_for "$site")
  four_chi_steps=$(four_chi_steps_for "$site")
  shard="$ROOT/shards/$site"
  trace="$ROOT/traces/$site"
  mkdir -p "$shard"
  nohup env PYTHONPATH=/home/dev/workspace "$PY" "$WRAPPER" \
    --trace-output "$trace" \
    --trace-site "$site" -- \
    --selection "$selection" \
    --checkpoint "$CHECKPOINT" \
    --output "$shard" \
    --site "$site" --target synthetic --frame crystal \
    --K 4 --n-starts 50 --n-steps 500 --lr 1.0 --seed 41 \
    --per-residue-class-schedule \
    --four-chi-stage-steps "$four_chi_steps" \
    --physics-refinement-steps 200 \
    --physics-refinement-lr-scale 0.1 \
    --lambda-vdw 1.0 --lambda-rot 0.5 --lambda-clash 5.0 \
    --vdw-threshold 3.0 --clash-threshold 2.5 \
    --symmetry-hard-threshold 2.0 \
    --symmetry-barrier-buffer 0.25 \
    --symmetry-barrier-scale 0.0 \
    > "$ROOT/logs/$site.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$ROOT/pids/$site.pid"
done

printf 'optimizing\n' > "$ROOT/status.txt"
failed=0
for site in "${sites[@]}"; do
  pid=$(<"$ROOT/pids/$site.pid")
  if wait "$pid"; then
    printf 'complete\n' > "$ROOT/pids/$site.status"
  else
    printf 'failed\n' > "$ROOT/pids/$site.status"
    failed=1
  fi
done

if (( failed )); then
  printf 'optimizer_failed\n' > "$ROOT/status.txt"
  exit 1
fi
printf 'optimization_complete\n' > "$ROOT/status.txt"
