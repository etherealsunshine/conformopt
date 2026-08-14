#!/usr/bin/env python3
"""Time the legacy SciPy and Torch-native per-slot TRF paths on one fold."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", nargs=3, required=True)
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--fold", type=int, default=0, choices=range(5))
    ap.add_argument("--inner-nfev", type=int, default=8)
    args = ap.parse_args()

    pdb_id, chain, resnum = args.site[0], args.site[1], int(args.site[2])
    site = (pdb_id, chain, resnum)
    args.output.mkdir(parents=True, exist_ok=False)
    base = APrimeSequential(
        args.output / "base", 80, 6, *site, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device=args.device, start_pdb=args.start,
        b_factor_mode="single_conformer",
    )
    train, _, _ = blocked_splits(base.base)[args.fold]

    rows = []
    for name, torch_native in (("scipy", False), ("torch", True)):
        root = args.output / name
        built = build_specs(
            root / "templates", args.flip_root, site=site, mask_scope="window",
            rama_floor=0.02, start_pdb=args.start,
            b_factor_mode="single_conformer", device=args.device,
            occupancy_scheme="mirror", mirror_eta=0.001,
            inner_nfev=args.inner_nfev, outer_updates=1,
            lambda_relative_tolerance=0.01, lambda_damping_alpha=0.3,
            lambda_norm_cap=1.0, amplitude_prior_lambda=0.008836284282618265,
            per_slot_trust_radii=True, torch_native_trf=torch_native,
        )
        spec = next(item for item in built if item["label"] == "D_null_axis2_30deg")
        spec["output"] = str(root / "D_null_axis2_30deg")
        spec["training_indices"] = train.tolist()
        spec["fixed_b_offset"] = 0.0
        started = time.perf_counter()
        result = worker(spec)
        elapsed = time.perf_counter() - started
        rows.append({
            "backend": name, "torch_native_trf": torch_native,
            "elapsed_s": elapsed,
            "inner_nfev": args.inner_nfev,
            "outer_updates": 1,
            "result": result,
        })

    scipy_time = next(row["elapsed_s"] for row in rows if row["backend"] == "scipy")
    torch_time = next(row["elapsed_s"] for row in rows if row["backend"] == "torch")
    payload = {
        "status": "complete", "site": list(site), "fold": args.fold,
        "device": args.device, "rows": rows,
        "speedup_scipy_over_torch": scipy_time / torch_time,
        "torch_is_at_least_1_5x_faster": scipy_time / torch_time >= 1.5,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "site", "fold", "device", "speedup_scipy_over_torch",
        "torch_is_at_least_1_5x_faster",
    )}, indent=2))


if __name__ == "__main__":
    main()
