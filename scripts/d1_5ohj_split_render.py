#!/usr/bin/env python3
"""Render exported 5OHJ model arrays with Torch/CUDA only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from density_denoiser.differentiable_renderer import render_cctbx_density


ROOT = Path(os.environ.get(
    "D1_SPLIT_ROOT",
    "/home/dev/qfit_unet_data/qfit_audit/d1_5ohj_aprime_phenix_closeout_v4_split",
))
MODE = os.environ.get("D1_SPLIT_MODE", "raw")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this renderer environment")
    inputs = np.load(ROOT / f"render_inputs_{MODE}.npz")
    device = torch.device("cuda")
    coords = torch.as_tensor(inputs["coordinates"], dtype=torch.float64, device=device)
    b_factors = torch.as_tensor(inputs["b_factors"], dtype=torch.float64, device=device)
    grid = torch.as_tensor(inputs["renderer_grid"], dtype=torch.float64, device=device)
    cell = torch.as_tensor(inputs["renderer_cell"], dtype=torch.float64, device=device)
    coefficients = torch.as_tensor(inputs["renderer_coefficients"], dtype=torch.float64, device=device)

    # This is the exact periodic-coordinate transform used by
    # SequentialBackbonePOC.model_density_torch.  The strict window mask uses
    # only the central image, so no CCTBX structure object is needed here.
    fractional = torch.linalg.solve(cell, coords.transpose(-1, -2)).transpose(-1, -2)
    fractional = fractional - torch.floor(fractional)
    atom_xyz = torch.matmul(fractional, cell.T)
    with torch.inference_mode():
        models = render_cctbx_density(
            grid, atom_xyz, b_factors, coefficients,
            u_base=float(inputs["renderer_u_base"][0]),
            exp_table_one_over_step_size=0.0,
            voxel_chunk=1024,
        ).detach().cpu().numpy()
    np.savez_compressed(ROOT / f"rendered_models_{MODE}.npz", models=models)
    manifest_path = ROOT / f"manifest_{MODE}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["render"] = {"status": "complete", "device": "cuda", "torch": torch.__version__, "models_shape": list(models.shape)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["render"], indent=2))


if __name__ == "__main__":
    main()
