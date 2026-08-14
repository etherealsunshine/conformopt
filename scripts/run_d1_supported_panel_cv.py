#!/usr/bin/env python3
"""Leakage-free blocked-CV rerun for the fixed A'' nine-site protocol.

This controller deliberately does not reuse the earlier CV candidate trees:
those geometries were optimized over the full mask because the worker dropped
``training_indices``.  Every endpoint here is freshly optimized using only its
own training voxels.  The fixed axis-2, 30-degree initialization is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Match the panel controller's qFit/CCTBX plus CUDA-Torch bootstrap before
# importing any helper which imports mmtbx/qFit.
QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np  # noqa: F401
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]

from run_d1_8d_sequential_poc import atomic_json
from run_d1_supported_panel import evaluate_fold, site_name

# 5JBX is excluded here because its deposited A--B RMSD is 0.129 A and cannot
# support a meaningful recovery claim.  7SC4 and 7UTC are excluded because the
# full-model fitted |DeltaB| exceeded 15 A^2.  Keep their prior records, but do
# not spend the leakage-free recovery budget on them.
RANKED_SITES = (
    ("4HFS", "A", 200), ("4ZXG", "B", 13), ("6ZWK", "B", 47),
    ("7ZTL", "A", 257), ("8AJK", "A", 240), ("8R7O", "C", 1681),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-site", action="append", default=[],
                        help="site name(s) to shard, e.g. 4HFS_A_200; leaves per-site artifacts identical")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.output}; pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "run_config.json", {
        "protocol": "fresh geometry fit on each fold's training voxels only; one fixed axis2-30deg initialization; backbone-only rho_calc and target; separate fitted-ΔB and ΔB=0 geometry fits",
        "fits_per_site": 10,
        "dB_treatments": {
            "fitted": "A'' and deposited two-state both fit dB on training voxels",
            "zero": "A'' geometry is separately optimized with dB fixed at 0; deposited two-state also has dB fixed at 0",
        },
        "invalidated_prior_root": "/home/dev/qfit_unet_data/qfit_audit/d1_supported_panel_axis2_30deg_v2",
        "invalidity": "prior worker dropped training_indices and every preceding CV used all-atom rho_calc against a target retaining frozen-sidechain mismatch",
    })
    selected_sites = RANKED_SITES
    if args.only_site:
        requested = set(args.only_site)
        known = {site_name(site): site for site in RANKED_SITES}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown --only-site value(s): {sorted(unknown)}")
        selected_sites = tuple(site for site in RANKED_SITES if site_name(site) in requested)
    completed = []
    for site in selected_sites:
        site_root = args.output / "sites" / site_name(site)
        folds = []
        for fold in range(5):
            path = site_root / f"fold_{fold}.json"
            if args.resume and path.is_file():
                record = json.loads(path.read_text())
            else:
                record = evaluate_fold(site, site_root / f"fold_{fold}", fold)
                atomic_json(path, record)
            folds.append(record)
            atomic_json(site_root / "progress.json", {"status": "running", "completed_folds": len(folds)})
        deltas = [float(row["dB_fitted"]["deposited_minus_A_prime_heldout_rss"]) for row in folds]
        zero_deltas = [float(row["dB_zero"]["deposited_minus_A_prime_heldout_rss"])
                       for row in folds]
        result = {
            "status": "complete", "site": site_name(site), "folds": folds,
            "paired_differences": deltas, "mean_difference": sum(deltas) / len(deltas),
            "sign_pattern": ["A_prime_better" if value > 0.0 else "deposited_better_or_tie" for value in deltas],
            "dB_zero": {
                "paired_differences": zero_deltas, "mean_difference": sum(zero_deltas) / len(zero_deltas),
                "sign_pattern": ["A_prime_better" if value > 0.0 else "deposited_better_or_tie"
                                 for value in zero_deltas],
            },
        }
        atomic_json(site_root / "result.json", result)
        completed.append({"site": site_name(site), "status": "complete"})
        atomic_json(args.output / "progress.json", {"status": "running", "completed": completed})
    if not args.only_site:
        atomic_json(args.output / "summary.json", {"status": "complete", "sites": completed})
        atomic_json(args.output / "progress.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    main()
