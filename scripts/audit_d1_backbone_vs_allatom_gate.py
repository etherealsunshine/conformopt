#!/usr/bin/env python3
"""Read-only fixed deposited-geometry scope comparison for the six-site panel.

The all-atom arm restores the former objective exactly: all window atoms are
rendered, their density remains in the residual, and the mask is the union of
all-atom spheres.  The backbone arm renders only N/CA/C/O, subtracts deposited-A
window sidechains, and masks backbone spheres.  Both profile occupancies,
intercept, and one global ΔB at fixed deposited A/B coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from run_d1_8d_sequential_poc import SequentialBackbonePOC, atomic_json
from run_d1_supported_panel import site_name


SITES = (
    ("4HFS", "A", 200), ("4ZXG", "B", 13), ("6ZWK", "B", 47),
    ("7ZTL", "A", 257), ("8AJK", "A", 240), ("8R7O", "C", 1681),
)


def compact(base, profile: dict[str, object]) -> dict[str, object]:
    return {
        "mask_voxels": int(base.mask.sum()),
        "model_atom_count": int(len(base.model_atom_indices)),
        "subtracted_external_neighbour_atoms": int(base.subtracted_atom_count),
        "subtracted_A_window_sidechain_atoms": int(base.subtracted_window_sidechain_atom_count),
        "occupancies": np.asarray(profile["weights"], dtype=float).tolist(),
        "intercept": float(profile["intercept"]),
        "dB_A2": float(profile["b_offset"]),
        "RSS": float(profile["rss"]),
        "RMSE": float(np.sqrt(float(profile["rss"]) / int(base.mask.sum()))),
    }


def one_scope(root: Path, site: tuple[str, str, int], scope: str) -> dict[str, object]:
    base = SequentialBackbonePOC(
        *site, root / "scratch" / site_name(site) / scope, 0.25, 2.0, 0.0,
        residual_scale_mode="none", renderer_backend="torch",
        map_scaler_structure="full", mask_scope="window", device="cuda",
        density_atom_scope=scope,
    )
    windows = [base.initial_window, base.window_for_deposited_b()]
    return compact(base, base.profile_affine_b_offset(base.target, windows, [0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(args.output / "run_config.json", {
        "operation": "fixed deposited A/B geometry only; no torsion optimization",
        "all_atom": "legacy residual/rho_calc/mask scope: frozen sidechains retained",
        "backbone": "current residual/rho_calc/mask scope: N/CA/C/O only, deposited-A sidechains subtracted",
    })
    rows = []
    for site in SITES:
        path = args.output / "sites" / f"{site_name(site)}.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text())
        else:
            row = {"site": site_name(site), "all_atom": one_scope(args.output, site, "all"),
                   "backbone_only": one_scope(args.output, site, "backbone")}
            atomic_json(path, row)
        rows.append(row)
        atomic_json(args.output / "progress.json", {"status": "running", "completed": len(rows)})
    atomic_json(args.output / "summary.json", {"status": "complete", "rows": rows})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed": len(rows)})


if __name__ == "__main__":
    main()
