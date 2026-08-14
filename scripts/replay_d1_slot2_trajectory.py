#!/usr/bin/env python3
"""Replay selected clean-D1 folds with slot-2 RMSD trajectory logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qfit  # noqa: F401
import numpy as np

from clean_d1_benchmark import site_key
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--inner-nfev", type=int, default=200)
    ap.add_argument("--outer-max", type=int, default=200)
    args = ap.parse_args()

    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    key = "6ZWK_B_PHE47"
    row = manifest[key]
    site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
    start = args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"
    args.output.mkdir(parents=True, exist_ok=False)
    base = APrimeSequential(
        args.output / "base", args.inner_nfev, args.outer_max, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=args.device,
        start_pdb=start, b_factor_mode="single_conformer",
    )
    folds = blocked_splits(base.base)
    for fold in (2, 4):
        train, _test, _direction = folds[fold]
        fold_root = args.output / f"fold_{fold}"
        specs = build_specs(
            fold_root / "templates", args.flip_root, site=site,
            mask_scope="window", rama_floor=0.02, start_pdb=start,
            b_factor_mode="single_conformer", device=args.device,
            occupancy_scheme="mirror", mirror_eta=0.001,
            amplitude_prior_lambda=0.008836284282618265,
            lambda_relative_tolerance=0.01,
            inner_nfev=args.inner_nfev, outer_updates=args.outer_max,
        )
        spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
        spec["output"] = str(fold_root / "D_null_axis2_30deg")
        spec["training_indices"] = np.asarray(train, dtype=int).tolist()
        spec["fixed_b_offset"] = 0.0
        result = worker(spec)
        (fold_root / "replay_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps({"status": "complete", "folds": [2, 4]}, indent=2))


if __name__ == "__main__":
    main()
