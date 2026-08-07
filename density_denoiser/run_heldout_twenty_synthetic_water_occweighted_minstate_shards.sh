#!/usr/bin/env bash
set -euo pipefail

export OUTPUT=${OUTPUT:-/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_occweighted_minstate_v1}
exec bash density_denoiser/run_heldout_twenty_synthetic_current_symmetry_shards.sh
