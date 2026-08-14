#!/usr/bin/env python3
"""Backbone-only fixed-geometry gate and two-site A'' recovery follow-up.

The target subtracts external neighbours and deposited-A sidechains from the
fitted seven-residue window.  rho_calc contains only N/CA/C/O.  The gate is
therefore the requested measurement of residual sidechain contamination after
the best available deposited-A sidechain subtraction; it does not claim to
remove density at deposited-B sidechain positions.
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
from run_d1_supported_panel import run_site, site_name


GATE_SITES = (
    ("4HFS", "A", 200), ("4ZXG", "B", 13), ("6ZWK", "B", 47),
    ("7ZTL", "A", 257), ("8AJK", "A", 240), ("8R7O", "C", 1681),
)
RECOVERY_SITES = (("8AJK", "A", 240), ("8R7O", "C", 1681))


def compact(profile: dict[str, object]) -> dict[str, object]:
    return {
        "occupancies": np.asarray(profile["weights"], dtype=float).tolist(),
        "intercept": float(profile["intercept"]), "dB_A2": float(profile["b_offset"]),
        "rss": float(profile["rss"]), "density_renders": int(profile["profile_density_renders"]),
        "B_offset_profile_interval_A2": list(profile["profile_interval_A2"]),
    }


def gate_site(root: Path, site: tuple[str, str, int]) -> dict[str, object]:
    base = SequentialBackbonePOC(
        *site, root / "gate_scratch" / site_name(site), 0.25, 2.0, 0.0,
        residual_scale_mode="none", renderer_backend="torch",
        map_scaler_structure="full", mask_scope="window", device="cuda",
    )
    deposited = [base.initial_window, base.window_for_deposited_b()]
    free = base.profile_affine_b_offset(base.target, deposited, [0, 1])
    fixed = base.profile_affine_b_offset(
        base.target, deposited, [0, 1], fixed_weights=base.deposited_occupancies,
    )
    return {
        "site": site_name(site),
        "configuration": {
            "target": "scaled map minus external neighbours and deposited-A window sidechains",
            "rho_calc": "N/CA/C/O only, both slots",
            "mask": "union of backbone spheres over deposited A and full deposited B seven-residue windows",
            "caveat": "deposited-B sidechain density is not removed; fixed geometry RSS quantifies its remaining contribution together with any other model mismatch",
        },
        "mask_voxels": int(base.mask.sum()), "model_backbone_atoms": int(len(base.model_atom_indices)),
        "subtracted_neighbour_atoms": int(base.subtracted_atom_count),
        "subtracted_deposited_A_window_sidechain_atoms": int(base.subtracted_window_sidechain_atom_count),
        "deposited_occupancies_A_B": base.deposited_occupancies.tolist(),
        "free_occupancy_intercept_dB": compact(free),
        "fixed_deposited_occupancy_intercept_dB": compact(fixed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover", action="store_true",
                        help="after the fixed deposited-geometry gate, run the fixed axis2-30deg A'' protocol at 8AJK and 8R7O")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(args.output / "run_config.json", {
        "operation": "backbone-only target/model gate, followed optionally by two fixed-initialization all-voxel A'' fits",
        "gate_sites": [site_name(site) for site in GATE_SITES],
        "recovery_sites": [site_name(site) for site in RECOVERY_SITES] if args.recover else [],
    })
    gate_rows = []
    for site in GATE_SITES:
        checkpoint = args.output / "gate" / f"{site_name(site)}.json"
        if args.resume and checkpoint.is_file():
            row = json.loads(checkpoint.read_text())
        else:
            row = gate_site(args.output, site)
            atomic_json(checkpoint, row)
        gate_rows.append(row)
        atomic_json(args.output / "progress.json", {"status": "gating", "completed_gate_sites": len(gate_rows)})
    atomic_json(args.output / "gate_summary.json", {"status": "complete", "rows": gate_rows})
    recoveries = []
    if args.recover:
        for site in RECOVERY_SITES:
            path = args.output / "recovery" / "sites" / site_name(site) / "site_result.json"
            if args.resume and path.is_file() and json.loads(path.read_text()).get("status") == "complete":
                row = json.loads(path.read_text())
            else:
                row = run_site(args.output / "recovery", site, include_cv=False)
            recoveries.append({"site": site_name(site), "result": row})
            atomic_json(args.output / "progress.json", {
                "status": "recovering", "completed_gate_sites": len(gate_rows),
                "completed_recovery_sites": len(recoveries),
            })
    atomic_json(args.output / "summary.json", {
        "status": "complete", "gate": gate_rows, "recoveries": recoveries,
    })
    atomic_json(args.output / "progress.json", {"status": "complete"})


if __name__ == "__main__":
    main()
