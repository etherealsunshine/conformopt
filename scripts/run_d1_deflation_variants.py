#!/usr/bin/env python3
"""Queue the two post-gate deflation variants on both clean-D1 sites."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import qfit  # noqa: F401
import numpy as np

from clean_d1_benchmark import site_key
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker


LAMBDA_AMP = 0.008836284282618265


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--diagnostic", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--inner-nfev", type=int, default=200)
    ap.add_argument("--outer-max", type=int, default=200)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    diagnostic = json.loads((args.diagnostic / "summary.json").read_text())
    d_perp = {row["site"]: np.asarray(row["d_perp_direction"], dtype=float)
              for row in diagnostic["sites"]}
    results = []

    for key in ("6ZWK_B_PHE47", "8R7O_C_THR1681"):
        row = manifest[key]
        site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
        start = args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"
        site_root = args.output / key
        base = APrimeSequential(
            site_root / "base", args.inner_nfev, args.outer_max, *site,
            renderer_backend="torch", residual_scale_mode="none",
            map_scaler_structure="full", mask_scope="window", device=args.device,
            start_pdb=start, b_factor_mode="single_conformer",
        )
        folds = blocked_splits(base.base)
        for fold, (train, _test, _direction) in enumerate(folds):
            specs = build_specs(
                site_root / f"fold_{fold}" / "templates", args.flip_root,
                site=site, mask_scope="window", rama_floor=0.02,
                start_pdb=start, b_factor_mode="single_conformer", device=args.device,
                occupancy_scheme="mirror", mirror_eta=0.001,
                amplitude_prior_lambda=LAMBDA_AMP,
                lambda_relative_tolerance=0.01,
                inner_nfev=args.inner_nfev, outer_updates=args.outer_max,
            )
            seed = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
            variants = [
                ("initial_orthogonal", "D_deflation_initial", "none",
                 (30.0 * d_perp[key]).tolist()),
                ("gradient_projected", "D_deflation_gradient", "slot2_gradient",
                 list(seed["p2"])),
            ]
            for name, label, mode, p2 in variants:
                spec = copy.deepcopy(seed)
                spec["label"] = label
                spec["output"] = str(site_root / f"fold_{fold}" / name / label)
                spec["p2"] = p2
                spec["training_indices"] = np.asarray(train, dtype=int).tolist()
                spec["fixed_b_offset"] = 0.0
                spec["deflation_mode"] = mode
                result = worker(spec)
                row_result = {
                    "site": key, "fold": fold, "variant": name,
                    "label": label, "deflation_mode": mode,
                    "result": result,
                }
                results.append(row_result)
                (args.output / "progress.json").write_text(
                    json.dumps({"status": "running", "completed": len(results), "total": 20})
                )

    report = {
        "status": "complete", "device": args.device,
        "inner_nfev": args.inner_nfev, "outer_max": args.outer_max,
        "mirror_eta": 0.001, "amplitude_prior_lambda": LAMBDA_AMP,
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "progress.json").write_text(
        json.dumps({"status": "complete", "completed": len(results), "total": 20})
    )
    print(json.dumps({"status": "complete", "runs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
