#!/usr/bin/env python3
"""Run the converged clean-D1 A-prime fits for the two benchmark sites."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import qfit  # noqa: F401  # keep qFit/CCTBX imports before CUDA Torch

from clean_d1_benchmark import site_key
from run_d1_slot_coordination import build_specs, worker


SITES = ("6ZWK_B_PHE47", "8R7O_C_THR1681")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--inner-nfev", type=int, default=40)
    ap.add_argument("--outer-max", type=int, default=50)
    ap.add_argument("--lambda-relative-tolerance", type=float, default=0.01)
    ap.add_argument("--mirror-eta", type=float, default=0.001)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    rows = {site_key(row): row for row in manifest}
    specs = []
    for key in SITES:
        row = rows[key]
        site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
        start = args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"
        site_root = args.output / key
        built = build_specs(
            site_root / "specs", args.flip_root, site=site, mask_scope="window",
            rama_floor=0.02, start_pdb=start, b_factor_mode="single_conformer",
            device=args.device, occupancy_scheme="mirror", mirror_eta=args.mirror_eta,
            inner_nfev=args.inner_nfev, outer_updates=args.outer_max,
            lambda_relative_tolerance=args.lambda_relative_tolerance,
        )
        spec = next(item for item in built if item["label"] == "D_null_axis2_30deg")
        spec["output"] = str(site_root / "D_null_axis2_30deg")
        spec["inner_nfev"] = args.inner_nfev
        spec["outer_updates"] = args.outer_max
        spec["lambda_relative_tolerance"] = args.lambda_relative_tolerance
        specs.append(spec)
    (args.output / "run_config.json").write_text(json.dumps({
        "sites": list(SITES), "device": args.device,
        "inner_evaluation_cap": args.inner_nfev,
        "outer_updates_max": args.outer_max,
        "lambda_relative_tolerance": args.lambda_relative_tolerance,
        "occupancy_scheme": "mirror", "mirror_eta": args.mirror_eta,
        "rama_floor": 0.02, "nullspace": "axis2,30 degrees",
        "b_factor_mode": "single_conformer_start_for_both_A-prime_slots",
        "per_slot_b_factor_refinement": False,
        "termination": "xtol/ftol/gtol or cap, with lambda norm relative change < 1%",
    }, indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(worker, copy.deepcopy(spec)): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append({"site": next(key for key in SITES if key in str(futures[future]["output"])),
                            "result": result})
    results.sort(key=lambda row: row["site"])
    (args.output / "summary.json").write_text(json.dumps({
        "status": "complete", "results": results,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "results": results}, indent=2))


if __name__ == "__main__":
    main()
